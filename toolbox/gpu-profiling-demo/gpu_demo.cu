// Small GPU workload covering five common bottleneck patterns.  It is the
// running example of experiences/gpu-profiling/.
//
// Build (see run_profile.sh for the portable version):
//   nvcc -O3 -std=c++17 -arch=native -lineinfo -DUSE_NVTX \
//        gpu_demo.cu -o gpu_demo -lnvtx3interop
// On CUDA <= 12, link -lnvToolsExt instead.  Without -DUSE_NVTX the program
// still runs, but the Nsight Systems timeline will not show phase ranges.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#if defined(USE_NVTX)
#include <nvtx3/nvToolsExt.h>
#define NVTX_PUSH(name) nvtxRangePushA(name)
#define NVTX_POP() nvtxRangePop()
#else
#define NVTX_PUSH(name) do {} while (0)
#define NVTX_POP() do {} while (0)
#endif

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
  std::exit(1); } } while (0)

static const int N = 1024;           // matmul / transpose side
static const int MATMUL_ITERS = 20;  // keep the naive kernel's runtime short
static const size_t VEC_N = 1u << 24; // 16M floats = 64 MB per buffer

// ---------------------------------------------------------------- kernels
__global__ void vector_add4(const float4* __restrict__ a, const float4* __restrict__ b,
                            float4* __restrict__ c, size_t n4) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n4) {
    float4 x = a[i], y = b[i];
    c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

__global__ void matmul_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < n && col < n) {
    float acc = 0.f;
    for (int k = 0; k < n; ++k) acc += A[row * n + k] * B[k * n + col];
    C[row * n + col] = acc;
  }
}

__global__ void transpose_naive(const float* __restrict__ in, float* __restrict__ out, int n) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < n && y < n) out[x * n + y] = in[y * n + x];   // strided write
}

__global__ void transpose_tiled(const float* __restrict__ in, float* __restrict__ out, int n) {
  __shared__ float tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x;
  int y = blockIdx.y * 32 + threadIdx.y;
  if (x < n && y < n) tile[threadIdx.y][threadIdx.x] = in[y * n + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x;
  y = blockIdx.x * 32 + threadIdx.y;
  if (x < n && y < n) out[y * n + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void tiny_kernel(float* x) { x[threadIdx.x] += 1.0f; }

// -------------------------------------------------------------- utilities
static float ms_between(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0.f; CK(cudaEventElapsedTime(&ms, a, b)); return ms;
}

static void init(float* p, size_t n, float seed) {
  for (size_t i = 0; i < n; ++i) p[i] = (float)((i * 131 + (size_t)seed) % 17) * 0.0625f;
}

int main() {
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s (sm_%d%d, %d SMs)\n",
         prop.name, prop.major, prop.minor, prop.multiProcessorCount);

  float *a, *b, *c, *A, *B, *C, *T, *T2, *dummy;
  CK(cudaMalloc(&a, VEC_N * sizeof(float)));
  CK(cudaMalloc(&b, VEC_N * sizeof(float)));
  CK(cudaMalloc(&c, VEC_N * sizeof(float)));
  CK(cudaMalloc(&A, (size_t)N * N * sizeof(float)));
  CK(cudaMalloc(&B, (size_t)N * N * sizeof(float)));
  CK(cudaMalloc(&C, (size_t)N * N * sizeof(float)));
  CK(cudaMalloc(&T, (size_t)N * N * sizeof(float)));
  CK(cudaMalloc(&T2, (size_t)N * N * sizeof(float)));
  CK(cudaMalloc(&dummy, 1024 * sizeof(float)));
  CK(cudaMemset(a, 0, VEC_N * sizeof(float)));
  CK(cudaMemset(b, 0, VEC_N * sizeof(float)));
  CK(cudaMemset(c, 0, VEC_N * sizeof(float)));

  float* h = (float*)malloc((size_t)N * N * sizeof(float));
  init(h, (size_t)N * N, 3);
  CK(cudaMemcpy(A, h, (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));
  init(h, (size_t)N * N, 5);
  CK(cudaMemcpy(B, h, (size_t)N * N * sizeof(float), cudaMemcpyHostToDevice));

  cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
  const size_t n4 = VEC_N / 4;
  const int threads = 256, blocks = (int)((n4 + threads - 1) / threads);
  dim3 blk(16, 16), grd((N + 15) / 16, (N + 15) / 16);
  dim3 tblk(32, 32), tgrd((N + 31) / 32, (N + 31) / 32);

  // warm-up: the first launch pays module load / JIT, never time it
  vector_add4<<<blocks, threads>>>((const float4*)a, (const float4*)b, (float4*)c, n4);
  matmul_naive<<<grd, blk>>>(A, B, C, N);
  transpose_naive<<<tgrd, tblk>>>(A, T, N);
  transpose_tiled<<<tgrd, tblk>>>(A, T2, N);
  tiny_kernel<<<1, 256>>>(dummy);
  CK(cudaDeviceSynchronize());

  // 1. memory-bound: vector add
  NVTX_PUSH("phase 1: vector_add");
  CK(cudaEventRecord(t0));
  vector_add4<<<blocks, threads>>>((const float4*)a, (const float4*)b, (float4*)c, n4);
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  double add_ms = ms_between(t0, t1);
  NVTX_POP();
  printf("vector_add      : %8.3f ms  %7.1f GB/s (read+write 3x64MB)\n", add_ms,
         3.0 * VEC_N * sizeof(float) / (add_ms * 1e6));

  // 2. compute-bound-ish: naive matmul, repeated
  NVTX_PUSH("phase 2: matmul_naive");
  CK(cudaEventRecord(t0));
  for (int it = 0; it < MATMUL_ITERS; ++it) matmul_naive<<<grd, blk>>>(A, B, C, N);
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  double mm_ms = ms_between(t0, t1);
  NVTX_POP();
  printf("matmul_naive    : %8.3f ms  %7.1f GFLOP/s (%.2f GFLOP)\n", mm_ms,
         2.0 * N * N * N * MATMUL_ITERS / (mm_ms * 1e6), 2.0 * N * N * N * MATMUL_ITERS / 1e9);

  // 3. uncoalesced write vs tiled transpose
  NVTX_PUSH("phase 3: transpose");
  CK(cudaEventRecord(t0));
  for (int it = 0; it < 20; ++it) transpose_naive<<<tgrd, tblk>>>(A, T, N);
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  double tr_ms = ms_between(t0, t1) / 20;
  CK(cudaEventRecord(t0));
  for (int it = 0; it < 20; ++it) transpose_tiled<<<tgrd, tblk>>>(A, T2, N);
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  double trt_ms = ms_between(t0, t1) / 20;
  NVTX_POP();
  printf("transpose_naive : %8.3f ms  %7.1f GB/s\n", tr_ms, 2.0 * N * N * sizeof(float) / (tr_ms * 1e6));
  printf("transpose_tiled : %8.3f ms  %7.1f GB/s\n", trt_ms, 2.0 * N * N * sizeof(float) / (trt_ms * 1e6));

  // 4. launch overhead: many tiny kernels
  NVTX_PUSH("phase 4: tiny_kernels");
  CK(cudaEventRecord(t0));
  for (int i = 0; i < 500; ++i) tiny_kernel<<<1, 256>>>(dummy);
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  double tiny_ms = ms_between(t0, t1);
  NVTX_POP();
  printf("500 tiny kernels: %8.3f ms  (%.1f us each, launch included)\n", tiny_ms, tiny_ms * 1000 / 500);

  // 5. async copy + compute overlap across two streams
  cudaStream_t s1, s2; CK(cudaStreamCreate(&s1)); CK(cudaStreamCreate(&s2));
  float* host_pinned = nullptr; CK(cudaMallocHost(&host_pinned, VEC_N * sizeof(float)));
  init(host_pinned, VEC_N, 7);
  NVTX_PUSH("phase 5: overlap");
  CK(cudaEventRecord(t0));
  CK(cudaMemcpyAsync(a, host_pinned, VEC_N * sizeof(float), cudaMemcpyHostToDevice, s2));
  for (int it = 0; it < 10; ++it) matmul_naive<<<grd, blk, 0, s1>>>(A, B, C, N);
  CK(cudaMemcpyAsync(host_pinned, c, VEC_N * sizeof(float), cudaMemcpyDeviceToHost, s2));
  CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
  double ov_ms = ms_between(t0, t1);
  NVTX_POP();
  printf("overlap phase   : %8.3f ms\n", ov_ms);

  CK(cudaFreeHost(host_pinned));
  free(h);
  CK(cudaStreamDestroy(s1)); CK(cudaStreamDestroy(s2));
  CK(cudaEventDestroy(t0)); CK(cudaEventDestroy(t1));
  CK(cudaFree(a)); CK(cudaFree(b)); CK(cudaFree(c));
  CK(cudaFree(A)); CK(cudaFree(B)); CK(cudaFree(C)); CK(cudaFree(T)); CK(cudaFree(T2));
  CK(cudaFree(dummy));
  printf("done\n");
  return 0;
}
