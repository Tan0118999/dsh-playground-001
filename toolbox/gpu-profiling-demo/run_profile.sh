#!/usr/bin/env bash
# Build and profile the GPU demo workload.  Outputs land in ./build and
# ./results.  Requires: NVIDIA GPU, CUDA toolkit (nvcc), Nsight Systems.
# Nsight Compute is optional: without counter permission the script skips it.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD="$HERE/build"
RESULTS="$HERE/results"
mkdir -p "$BUILD" "$RESULTS"

# ---------------------------------------------------------------- locate tools
NVCC="${NVCC:-$(command -v nvcc || true)}"
if [[ -z "$NVCC" ]]; then
  for c in /usr/local/cuda/bin/nvcc /usr/local/cuda-*/bin/nvcc; do
    [[ -x "$c" ]] && NVCC="$c" && break
  done
fi
if [[ -z "$NVCC" ]]; then
  echo "error: nvcc not found; set NVCC=/path/to/nvcc" >&2
  exit 1
fi
NSYS="${NSYS:-$(command -v nsys || true)}"
if [[ -z "$NSYS" ]]; then
  echo "error: nsys not found; install Nsight Systems and put it on PATH" >&2
  exit 1
fi
ARCH="${NVCC_ARCH:-native}"

# NVTX: CUDA 13 ships libnvtx3interop, older toolkits ship libnvToolsExt.
NVTX_DIR="${NVTX_DIR:-}"
NVTX_LIB="${NVTX_LIB:-}"
if [[ -n "$NVTX_DIR" ]]; then
  for name in libnvtx3interop.so libnvToolsExt.so; do
    [[ -e "$NVTX_DIR/$name" ]] && NVTX_LIB="$name" && break
  done
else
  for name in libnvtx3interop.so libnvToolsExt.so; do
    found=$(find /usr/local/cuda* /usr/lib /opt/nvidia -name "$name" 2>/dev/null | head -1 || true)
    if [[ -n "$found" ]]; then
      NVTX_LIB="$name"; NVTX_DIR="$(dirname "$found")"; break
    fi
  done
fi
if [[ -n "$NVTX_LIB" && -n "$NVTX_DIR" ]]; then
  linkname="${NVTX_LIB#lib}"; linkname="${linkname%.so}"
  NVTX_FLAGS=(-DUSE_NVTX "-L$NVTX_DIR" "-l$linkname" "-Xlinker" "-rpath" "-Xlinker" "$NVTX_DIR")
  export LD_LIBRARY_PATH="$NVTX_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "== NVTX: $NVTX_DIR/$NVTX_LIB"
else
  NVTX_FLAGS=()
  echo "== NVTX: not found; building without phase ranges (timeline will have no NVTX row)"
  echo "   install the CUDA nvtx package (libnvtx3interop / libnvToolsExt) to enable them"
fi

echo "== compiler: $NVCC ($($NVCC --version | tail -1))"
echo "== arch: $ARCH"

"$NVCC" -O3 -std=c++17 -arch="$ARCH" -lineinfo \
  "${NVTX_FLAGS[@]}" \
  "$HERE/gpu_demo.cu" -o "$BUILD/gpu_demo" || exit 1

echo
echo "== native run (5 repetitions, no profiler) =="
: > "$RESULTS/run-native.txt"
for i in 1 2 3 4 5; do
  echo "--- run $i ---" | tee -a "$RESULTS/run-native.txt"
  "$BUILD/gpu_demo" | tee -a "$RESULTS/run-native.txt"
done
echo

echo "== Nsight Systems capture =="
"$NSYS" profile \
  --trace=cuda,nvtx,osrt \
  --sample=process-tree \
  --cpuctxsw=process-tree \
  --force-overwrite=true \
  -o "$RESULTS/demo_profile" \
  "$BUILD/gpu_demo" > "$RESULTS/nsys-capture.log" 2>&1
# keep only the application's own output; nsys progress/paths stay in the log
grep -E '^(GPU:|vector_add|matmul_naive|transpose_|500 tiny|overlap phase|done)' \
  "$RESULTS/nsys-capture.log" > "$RESULTS/run-under-nsys.txt" || true
echo

echo "== SQLite export + stats =="
"$NSYS" export --force-overwrite=true --type sqlite -o "$RESULTS/demo_profile.sqlite" \
  "$RESULTS/demo_profile.nsys-rep" >/dev/null 2>&1 || true

for report in cuda_gpu_kern_sum cuda_gpu_mem_time_sum cuda_api_sum nvtx_sum; do
  case "$report" in
    cuda_gpu_kern_sum)    out="$RESULTS/nsys-kern-sum.txt" ;;
    cuda_gpu_mem_time_sum) out="$RESULTS/nsys-mem-sum.txt" ;;
    cuda_api_sum)         out="$RESULTS/nsys-api-sum.txt" ;;
    nvtx_sum)             out="$RESULTS/nsys-nvtx-sum.txt" ;;
  esac
  "$NSYS" stats --report "$report" "$RESULTS/demo_profile.sqlite" 2>&1 \
    | grep -vE '^(Generating SQLite|Processing )' \
    | sed "s#$RESULTS/##g" > "$out" || true
done
echo "   wrote results/nsys-*.txt"

echo
echo "== timeline figures =="
if [[ -f "$RESULTS/demo_profile.sqlite" ]] && python3 -c 'import matplotlib' 2>/dev/null; then
  python3 "$HERE/plot_timeline.py" "$RESULTS/demo_profile.sqlite" "$RESULTS/figures" || true
  echo "   wrote results/figures/*.png"
else
  echo "   skipped figures (need sqlite export + matplotlib)"
fi

echo
echo "== Nsight Compute (optional) =="
NCU="${NCU:-$(command -v ncu || true)}"
if [[ -z "$NCU" ]]; then
  for c in /opt/nvidia/nsight-compute/*/ncu; do
    [[ -x "$c" ]] && NCU="$c" && break
  done
fi
if [[ -z "$NCU" ]]; then
  echo "   ncu not found; skip kernel-level capture"
else
  out=$("$NCU" --set basic -k regex:matmul_naive -c 1 -o "$RESULTS/matmul_basic" \
        "$BUILD/gpu_demo" 2>&1)
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "   ncu failed (rc=$rc); this is expected without GPU counter permission:"
    echo "$out" | grep -E 'ERR_NVGPUCTRPERM|ERROR' | head -3 | sed 's/^/   /'
    echo "   see ../../experiences/gpu-profiling/gpu-profiling-permissions-and-containers.md for the permission setup"
  else
    echo "   wrote results/matmul_basic.ncu-rep"
    "$NCU" -i "$RESULTS/matmul_basic.ncu-rep" --page details --print-details header \
      > "$RESULTS/matmul-basic.txt" 2>&1 || true
  fi
fi

echo
echo "done. open $RESULTS/demo_profile.nsys-rep in nsys-ui, or read the text files."
