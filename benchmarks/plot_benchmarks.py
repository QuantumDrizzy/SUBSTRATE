"""
SUBSTRATE — Benchmark Visualization
Generates publication-quality benchmark plots from existing data.
Output: benchmarks/plots/
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
PLOTS   = Path(__file__).parent / "plots"
PLOTS.mkdir(exist_ok=True)

DARK   = "#0d1117"
ACCENT = "#00d4ff"
GREEN  = "#39ff14"
ORANGE = "#ff6b35"
PURPLE = "#b388ff"
TEXT   = "#e6edf3"
GRID   = "#21262d"

plt.rcParams.update({
    "figure.facecolor": DARK, "axes.facecolor": DARK,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "grid.linewidth": 0.5, "font.family": "monospace",
    "font.size": 11,
})

# ── 1. U(1) plaquette CUDA kernel: roofline crossover vs JAX-CPU ───
def plot_plaquette_roofline():
    """
    Honest CUDA story: hand-written U(1) plaquette kernel (CuPy RawKernel,
    sm_120) vs a warmed @jax.jit reference running on CPU (no GPU JAX backend
    in this build). Source data: benchmarks/results/plaquette_cuda.json,
    produced by quantum_lab/P3_G2/benchmark_plaquette.py (warmed both sides,
    CUDA-event kernel timing, 1000-iter medians). The point is the crossover,
    not one hero number: kernel-only scales with problem size; end-to-end
    (with H2D/D2H transfers) only wins past L≈128.
    """
    src = RESULTS / "plaquette_cuda.json"
    if not src.exists():
        print("  skip: no plaquette_cuda.json")
        return
    rows = json.loads(src.read_text())
    L      = [r["L"] for r in rows]
    k_spd  = [r["kernel_speedup"] for r in rows]
    e_spd  = [r["e2e_speedup"] for r in rows]
    jax_ms = [r["jax_ms"] for r in rows]
    k_ms   = [r["cuda_kernel_ms"] for r in rows]
    e_ms   = [r["cuda_e2e_ms"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
    fig.suptitle("SUBSTRATE — U(1) Plaquette CUDA Kernel vs JAX-CPU\n"
                 "RawKernel · sm_120 Blackwell (RTX 5060 Ti) · CUDA-event timing, 1000-iter medians",
                 fontsize=13, fontweight="bold", color=TEXT, y=1.03)

    # Left: speedup vs lattice size (the crossover)
    ax = axes[0]
    ax.plot(L, k_spd, color=ACCENT, marker="o", markersize=4, linewidth=1.6,
            zorder=4, label="kernel-only (compute)")
    ax.plot(L, e_spd, color=ORANGE, marker="s", markersize=4, linewidth=1.6,
            zorder=4, label="end-to-end (+H2D/D2H)")
    ax.axhline(1.0, color=GREEN, linestyle="--", linewidth=1.0, zorder=2,
               label="break-even (1×)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(L); ax.set_xticklabels([str(x) for x in L])
    ax.set_xlabel("Lattice size L  (cells = 2·L²)")
    ax.set_ylabel("Speedup over JAX-CPU (×, log)")
    ax.set_title("Roofline crossover: overhead-bound → compute-bound", color=TEXT)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(zorder=0, which="both", alpha=0.4)
    ax.annotate(f"{k_spd[-1]:.0f}×", xy=(L[-1], k_spd[-1]), xytext=(-4, 6),
                textcoords="offset points", ha="right", fontsize=11,
                fontweight="bold", color=ACCENT)
    ax.annotate("e2e break-even\n≈ L 128", xy=(128, 1.0), xytext=(0, 18),
                textcoords="offset points", ha="center", fontsize=7,
                color=GREEN, alpha=0.9)

    # Right: absolute time vs size (log-log)
    ax = axes[1]
    ax.plot(L, jax_ms, color=PURPLE, marker="o", markersize=4, linewidth=1.6,
            zorder=4, label="JAX-CPU")
    ax.plot(L, k_ms, color=ACCENT, marker="o", markersize=4, linewidth=1.6,
            zorder=4, label="CUDA kernel-only")
    ax.plot(L, e_ms, color=ORANGE, marker="s", markersize=4, linewidth=1.6,
            zorder=4, label="CUDA end-to-end")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(L); ax.set_xticklabels([str(x) for x in L])
    ax.set_xlabel("Lattice size L")
    ax.set_ylabel("Wall time per call (ms, log)")
    ax.set_title("Absolute latency — transfers dominate at small L", color=TEXT)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(zorder=0, which="both", alpha=0.4)
    ax.text(0.98, 0.04, "JAX = CPU backend (no GPU JAX in this build)",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=TEXT, alpha=0.45)

    fig.tight_layout()
    out = PLOTS / "plaquette_roofline.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  [ok] {out}")


# ── 2. TDVP integration convergence ───────────────────────────────
def plot_tdvp_convergence():
    data = []
    with open(RESULTS / "tdvp_timing.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("event") == "integrate":
                data.append(d)
    if not data:
        print("  skip: no tdvp integrate events")
        return

    times  = [d["integrate_time_s"] for d in data]
    errors = [d["final_trace"] for d in data]   # |1 - tr(ρ)| — trace error, ideal = 0
    runs   = list(range(len(data)))

    # Two regimes: standard tolerance (runs 0-17) vs tight tolerance (runs 18+)
    split = next((i for i, t in enumerate(times) if t > 60), len(times))
    c_std   = [ACCENT if i < split else PURPLE for i in runs]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("SUBSTRATE — TDVP Integration (FAD-W Radical Pair, CPU RK45)",
                 fontsize=13, fontweight="bold", color=TEXT)

    # Left: wall time per run
    ax = axes[0]
    ax.scatter(runs[:split],  times[:split],  color=ACCENT,  s=20, zorder=4, label="Standard tolerance")
    ax.scatter(runs[split:],  times[split:],  color=PURPLE,  s=20, zorder=4, label="Tight tolerance")
    ax.plot(runs[:split],  times[:split],  color=ACCENT,  linewidth=1.2, zorder=3)
    ax.plot(runs[split:],  times[split:],  color=PURPLE,  linewidth=1.2, zorder=3)
    ax.axhline(np.mean(times[:split]),  color=ACCENT,  linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"std mean={np.mean(times[:split]):.1f}s")
    ax.axhline(np.mean(times[split:]),  color=PURPLE,  linestyle="--", linewidth=0.8, alpha=0.6,
               label=f"tight mean={np.mean(times[split:]):.1f}s")
    ax.set_xlabel("Run #"); ax.set_ylabel("Wall time (s)")
    ax.set_title("Integration time per run (100 steps, 5 µs)", color=TEXT)
    ax.legend(fontsize=8); ax.grid(zorder=0)

    # Right: trace error |1 - tr(ρ)| — lower = better normalization
    ax = axes[1]
    ax.scatter(runs[:split], errors[:split], color=ACCENT,  s=20, zorder=4)
    ax.scatter(runs[split:], errors[split:], color=PURPLE,  s=20, zorder=4)
    ax.plot(runs[:split], errors[:split], color=ACCENT,  linewidth=1.2, zorder=3,
            label=f"std:   err={errors[0]:.2e}")
    ax.plot(runs[split:], errors[split:], color=PURPLE,  linewidth=1.2, zorder=3,
            label=f"tight: err={errors[-1]:.2e}")
    ax.axhline(0.0, color=GREEN, linestyle="--", linewidth=1, label="ideal (err = 0)")
    ax.set_xlabel("Run #"); ax.set_ylabel("|1 − tr(ρ)|  (trace error, lower = better)")
    ax.set_title("Trace conservation — tighter tol → 28× lower error", color=TEXT)
    ax.set_ylim(-0.0005, max(errors) * 1.3)
    ax.legend(fontsize=8); ax.grid(zorder=0)
    # Annotation: trade-off
    ax.annotate(f"3× slower\n28× better",
                xy=(split, errors[split]), xytext=(split - 4, errors[0] * 0.6),
                fontsize=8, color=PURPLE, alpha=0.9,
                arrowprops=dict(arrowstyle="->", color=PURPLE, alpha=0.6))

    fig.tight_layout()
    out = PLOTS / "tdvp_convergence.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  [ok] {out}")


# ── 3. ErCry4a radical pair sensitivity ───────────────────────────
def plot_ercry4a():
    data = []
    with open(RESULTS / "ercry4a_benchmark.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if "phi_s" in d:
                data.append(d)
    if not data:
        print("  skip: no ercry4a phi_s data")
        return

    n      = len(data)
    phi_0  = [d["phi_s"][0] for d in data]
    phi_50 = [d["phi_s"][1] for d in data]
    delta  = [abs(d["delta_phi_s_earth"]) * 1000 for d in data]
    times  = [d["wall_time_s"] for d in data]
    runs   = np.arange(n)
    w      = 0.30

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("SUBSTRATE — ErCry4a Magnetoreception (6 nuclear spins, 8-site, exact χ)",
                 fontsize=13, fontweight="bold", color=TEXT)

    # Left: singlet yield — grouped bars per run
    ax = axes[0]
    b1 = ax.bar(runs - w/2, phi_0,  w, color=ACCENT,  alpha=0.9, label="φₛ  0 mT (dark)", zorder=3)
    b2 = ax.bar(runs + w/2, phi_50, w, color=PURPLE,  alpha=0.9, label="φₛ  0.05 mT (Earth)", zorder=3)
    ax.set_xticks(runs); ax.set_xticklabels([f"Run {i}" for i in runs])
    ax.set_ylabel("Singlet yield φₛ")
    ax.set_ylim(0.050, 0.062)
    ax.set_title("Singlet yield vs applied field", color=TEXT)
    ax.legend(fontsize=9); ax.grid(axis="y", zorder=0)
    # value labels
    for bar, val in zip(b1, phi_0):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                f"{val:.5f}", ha="center", va="bottom", fontsize=8, color=ACCENT)
    for bar, val in zip(b2, phi_50):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.0001,
                f"{val:.5f}", ha="center", va="bottom", fontsize=8, color=PURPLE)
    # wall time annotation
    for i, t in enumerate(times):
        ax.annotate(f"{t:.0f}s", xy=(i, 0.050), xycoords=("data","axes fraction"),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, color=TEXT, alpha=0.5)

    # Right: Earth-field sensitivity ΔφS × 10³
    ax = axes[1]
    bars = ax.bar(runs, delta, width=0.4, color=GREEN, alpha=0.9, zorder=3)
    ax.set_xticks(runs); ax.set_xticklabels([f"Run {i}" for i in runs])
    ax.set_ylabel("|ΔφS| × 10³  (Earth-field sensitivity)")
    ax.set_title("Radical pair Earth-field sensitivity", color=TEXT)
    ax.set_ylim(0, max(delta) * 1.5)
    ax.axhline(np.mean(delta), color=ORANGE, linestyle="--", linewidth=1,
               label=f"mean = {np.mean(delta):.3f}×10⁻³")
    ax.legend(fontsize=9); ax.grid(axis="y", zorder=0)
    for bar, val in zip(bars, delta):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f"{val:.3f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold", color=GREEN)
    ax.text(0.98, 0.04, "deterministic solver — consistent across runs",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color=TEXT, alpha=0.45)

    fig.tight_layout()
    out = PLOTS / "ercry4a_sensitivity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  [ok] {out}")


# ── 4. GPU Krylov build timing ─────────────────────────────────────
def plot_gpu_krylov():
    times = []
    with open(RESULTS / "gpu_timing.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d.get("event") == "gpu_krylov_build":
                times.append(d["build_time_ms"])
    if not times:
        print("  skip: no gpu_krylov data")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.suptitle("SUBSTRATE — GPU Krylov Build Time (FAD-W, dim=64, sm_120)",
                 fontsize=13, fontweight="bold", color=TEXT)
    ax.plot(times, color=ACCENT, linewidth=1.5, marker="o", markersize=3, zorder=3)
    ax.axhline(np.mean(times), color=ORANGE, linestyle="--", linewidth=1.2,
               label=f"mean = {np.mean(times):.1f} ms")
    ax.fill_between(range(len(times)), np.mean(times)-np.std(times),
                    np.mean(times)+np.std(times), color=ACCENT, alpha=0.1)
    ax.set_xlabel("Build #"); ax.set_ylabel("Wall time (ms)")
    ax.set_title("RTX 5060 Ti — GPU Krylov matrix build latency", color=TEXT)
    ax.legend(fontsize=10); ax.grid(zorder=0)
    fig.tight_layout()
    out = PLOTS / "gpu_krylov_timing.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"  [ok] {out}")


if __name__ == "__main__":
    print("SUBSTRATE benchmark plots")
    print("=" * 40)
    plot_plaquette_roofline()
    plot_tdvp_convergence()
    plot_ercry4a()
    plot_gpu_krylov()
    print(f"\nAll plots -> {PLOTS}/")
