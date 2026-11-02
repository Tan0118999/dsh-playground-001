#!/usr/bin/env python3
"""Render the Nsight Systems SQLite export as a readable timeline figure.

Usage: python3 plot_timeline.py demo_profile.sqlite OUTDIR
Requires: matplotlib.  The SQLite file comes from:
    nsys export --type sqlite -o demo_profile.sqlite demo_profile.nsys-rep
"""
import os, sqlite3, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

COLORS = {
    "vector_add4": "#2f7ed8",
    "matmul_naive": "#e07b39",
    "transpose_naive": "#8c5bb8",
    "transpose_tiled": "#3f9e7a",
    "tiny_kernel": "#7a7a7a",
}
KIND = {1: "HtoD", 2: "DtoH", 3: "DtoD", 4: "HtoH"}
plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": "#bbbbbb", "axes.linewidth": 0.8,
    "xtick.color": "#333333", "ytick.color": "#333333",
})

def load(path):
    c = sqlite3.connect(path)
    nvtx = list(c.execute("SELECT start,end,text FROM NVTX_EVENTS "
                          "WHERE text LIKE 'phase %' ORDER BY start"))
    kernels = list(c.execute(
        "SELECT k.start,k.end,k.streamId,s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id=k.shortName ORDER BY k.start"))
    copies = list(c.execute(
        "SELECT start,end,copyKind,bytes,streamId FROM CUPTI_ACTIVITY_KIND_MEMCPY "
        "ORDER BY start"))
    return nvtx, kernels, copies

def full_figure(nvtx, kernels, copies, out):
    t0 = nvtx[0][0]
    fig, ax = plt.subplots(figsize=(12.4, 4.6), dpi=200)
    minx = (min(k[0] for k in kernels) - t0) / 1e6
    maxx = (max(nvtx[-1][1], max(k[1] for k in kernels)) - t0) / 1e6
    # phase bands
    for i, (s, e, text) in enumerate(nvtx):
        x0, x1 = (s - t0) / 1e6, (e - t0) / 1e6
        ax.axvspan(x0, x1, color="#f2f5f9" if i % 2 == 0 else "#e9eef5",
                   zorder=0, linewidth=0)
        ax.axvline(x0, color="#c8d0dd", lw=0.7, ls="--", zorder=1)
    # phase labels (staggered so short phases do not collide)
    p2_ms = (nvtx[1][1] - nvtx[1][0]) / 1e6
    labels = [
        (0, "P1", 3.62, 0.15),
        (1, f"P2 matmul_naive ({p2_ms:.1f} ms)", 3.62, 5.0),
        (2, "P3", 3.40, 0.20),
        (3, "P4 tiny_kernels", 3.22, 0.50),
        (4, "P5 overlap", 3.62, 0.40),
    ]
    for idx, label, yy, dx in labels:
        s, e, _ = nvtx[idx]
        ax.text((s - t0) / 1e6 + dx, yy, label, ha="left", va="bottom",
                fontsize=7.5, color="#33415c")
    # kernel row
    for s, e, stream, name in kernels:
        x0, x1 = (s - t0) / 1e6, (e - t0) / 1e6
        ax.barh(1.85, max(x1 - x0, 0.06), left=x0, height=0.42,
                color=COLORS.get(name, "#999999"), edgecolor="none", zorder=3)
    # memcpy row
    for s, e, kind, nbytes, stream in copies:
        x0, x1 = (s - t0) / 1e6, (e - t0) / 1e6
        if x1 < 0:
            continue
        ax.barh(0.85, max(x1 - x0, 0.10), left=x0, height=0.24,
                color="#55a0a8", edgecolor="none", zorder=3)
        if kind == 1:
            ax.text((x0 + x1) / 2, 1.10, "HtoD 64 MiB", ha="center", va="bottom",
                    fontsize=6.5, color="#2c6b72")
        else:
            ax.text(x1 + 0.4, 0.85, "DtoH 64 MiB", ha="left", va="center",
                    fontsize=6.5, color="#2c6b72")
    # NVTX row
    for i, (s, e, text) in enumerate(nvtx):
        x0, x1 = (s - t0) / 1e6, (e - t0) / 1e6
        ax.barh(2.75, x1 - x0, left=x0, height=0.22,
                color="#c9d4e4", edgecolor="#8a99b5", linewidth=0.5, zorder=3)
    # CPU-only gap
    p4_end = (nvtx[3][1] - t0) / 1e6
    p5_start = (nvtx[4][0] - t0) / 1e6
    ax.annotate("", xy=(p5_start, 3.40), xytext=(p4_end, 3.40),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.1))
    ax.text((p4_end + p5_start) / 2, 2.30,
            "CPU-only gap %.0f ms\n(host malloc / stream setup)\nGPU idle"
            % (p5_start - p4_end),
            ha="center", va="center", fontsize=7, color="#c0392b")
    p4_span = (nvtx[3][1] - nvtx[3][0]) / 1e6
    ax.annotate(f"500 tiny kernels:\nNVTX span {p4_span:.1f} ms,\nGPU busy only 0.25 ms",
                xy=((nvtx[3][0] + 5000 - t0) / 1e6, 1.85), xytext=(40, 2.6),
                fontsize=7, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
    ax.annotate("H2D + 10 matmuls + D2H:\ncopy engine works while SMs compute",
                xy=((nvtx[4][0] + 3000 - t0) / 1e6, 1.0),
                xytext=(58, 0.30), fontsize=7, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
    ax.set_yticks([2.75, 1.85, 0.85])
    ax.set_yticklabels(["NVTX ranges", "GPU kernels", "Memcpy"])
    ax.set_xlim(minx - 1.0, maxx + 1.0)
    ax.set_ylim(0.1, 3.85)
    ax.set_xlabel("time since first NVTX range (ms)")
    ax.set_title("Nsight Systems trace of the demo workload (real capture, SQLite export)",
                 fontsize=11, loc="left", color="#222222")
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=1)
    handles = [Patch(color=c, label=n) for n, c in COLORS.items() if n != "tiny_kernel"]
    handles.append(Patch(color=COLORS["tiny_kernel"], label="tiny_kernel (x500)"))
    handles.append(Patch(color="#55a0a8", label="memcpy"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7.2,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out)
    plt.close(fig)

def matmul_figure(nvtx, kernels, out):
    phase = next(p for p in nvtx if "matmul" in p[2])
    t0, t_end = phase[0], phase[1]
    ks = [k for k in kernels if t0 <= k[0] < t_end and k[3] == "matmul_naive"]
    fig, ax = plt.subplots(figsize=(11.6, 2.9), dpi=200)
    for i, (s, e, stream, name) in enumerate(ks):
        x0, x1 = (s - t0) / 1e6, (e - t0) / 1e6
        ax.barh(1.0, x1 - x0, left=x0, height=0.5,
                color=COLORS[name], edgecolor="#9c4f1d", linewidth=0.4)
    span_ms = (t_end - t0) / 1e6
    ax.axvspan(0.0, span_ms, color="#f2f5f9", zorder=0)
    ax.text(0.2, 1.74, f"NVTX range: {phase[2]}  ({span_ms:.2f} ms)", fontsize=8,
            color="#44506b")
    ax.annotate("most kernels ~1.03 ms (some 1.2–1.3 ms), gap 1–4 µs\n(back-to-back launches, GPU nearly always busy)",
                xy=(10.6, 1.0), xytext=(4.2, 0.36), fontsize=7.4, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
    ax.set_xlim(-0.2, span_ms + 0.4)
    ax.set_ylim(0.15, 1.95)
    ax.set_yticks([1.0])
    ax.set_yticklabels(["matmul_naive\n20 launches"])
    ax.set_xlabel("time since phase start (ms)")
    ax.set_title("Zoom: one NVTX phase, 20 identical kernels (real capture)", fontsize=10.5,
                 loc="left", color="#222222")
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

def overlap_figure(nvtx, kernels, copies, out):
    phase = next(p for p in nvtx if "overlap" in p[2])
    t0, t_end = phase[0], phase[1]
    span_ms = (t_end - t0) / 1e6
    fig, ax = plt.subplots(figsize=(11.6, 3.0), dpi=200)
    ks = [k for k in kernels if t0 - 1e6 <= k[0] <= t_end + 1e6]
    for s, e, stream, name in ks:
        x0, x1 = (s - t0) / 1e6, (e - t0) / 1e6
        ax.barh(1.55, x1 - x0, left=x0, height=0.4,
                color=COLORS.get(name, "#999999"), edgecolor="none")
        ax.text((x0 + x1) / 2, 1.82, f"{x1-x0:.2f} ms", ha="center", va="bottom",
                fontsize=6.4, color="#7a3c14")
    for s, e, kind, nbytes, stream in copies:
        if s < t0 - 1e6 or s > t_end + 1e6:
            continue
        x0, x1 = (s - t0) / 1e6, (e - t0) / 1e6
        ax.barh(0.62, x1 - x0, left=x0, height=0.26, color="#55a0a8")
        ax.text((x0 + x1) / 2, 0.9, f"{KIND.get(kind)} 64 MiB", ha="center",
                va="bottom", fontsize=6.6, color="#2c6b72")
    ax.axvspan(0.0, span_ms, color="#f2f5f9", zorder=0)
    ax.text(0.15, 2.2, f"NVTX range: {phase[2]}  ({span_ms:.2f} ms)", fontsize=8,
            color="#44506b")
    compute_ms = sum(e - s for s, e, _, _ in ks) / 1e6
    copy_ms = sum(e - s for s, e, _, _, _ in copies) / 1e6
    ax.annotate(f"copy engine (memcpy row) runs while SMs compute:\n"
                f"{len(ks)} matmuls = {compute_ms:.1f} ms, copies = {copy_ms:.1f} ms; total = {span_ms:.1f} ms",
                xy=(1.4, 0.62), xytext=(3.6, 1.25), fontsize=7.2, color="#444444",
                arrowprops=dict(arrowstyle="->", color="#666666", lw=0.9))
    ax.set_xlim(-0.1, span_ms + 0.3)
    ax.set_ylim(0.25, 2.4)
    ax.set_yticks([1.55, 0.62])
    ax.set_yticklabels(["SM kernels\n(stream 13)", "Memcpy\n(stream 14)"])
    ax.set_xlabel("time since phase start (ms)")
    ax.set_title("Zoom: asynchronous copies overlap compute in two streams (real capture)",
                 fontsize=10.5, loc="left", color="#222222")
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=1)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)

if __name__ == "__main__":
    db, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    nvtx, kernels, copies = load(db)
    full_figure(nvtx, kernels, copies, os.path.join(outdir, "timeline-full.png"))
    matmul_figure(nvtx, kernels, os.path.join(outdir, "timeline-matmul.png"))
    overlap_figure(nvtx, kernels, copies, os.path.join(outdir, "timeline-overlap.png"))
    print("wrote 3 figures to", outdir)
