"""
coupled_run.py — POLE_SHIFT_SIM: VADM-coupled LBM simulation

Couples the Sint-2000 VADM paleomagnetic record to the LBM body-force field.
Physics: weak geomagnetic field → reduced electromagnetic core-mantle coupling
         → lower drag on lithosphere → higher susceptibility to polar wander.

Coupling law:
    force_scale(t) = F0 * (VADM_ref / VADM(t))^alpha

where:
    F0        = baseline force magnitude (calibrated to produce ~cm/yr drift)
    VADM_ref  = long-term mean VADM (~8 × 10^22 A·m²)
    VADM(t)   = Sint-2000 value at geological time t
    alpha     = coupling exponent (default 1.0; try 0.5, 2.0 for sensitivity)

Time mapping:
    LBM step dt = 100 yr (matches the 100-yr aligned grid)
    Simulation: 0 → T_MAX_BP years before present
    Step k ↔ age_bp = T_MAX_BP - k * 100

Two runs are performed and compared:
    1. Coupled   — force_scale modulated by VADM(t)
    2. Control   — constant force_scale = F0 (VADM fixed at reference)

Expected result: displacement spike at Laschamp cluster (37-46 ka BP),
where VADM collapses to ~15-25% of reference.

Usage:
    python src/pole_shift_sim/coupled_run.py [--steps 800] [--alpha 1.0] [--gpu]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pole_shift_sim.lbm_core import LBMSolver, build_polar_wander_force, xp_backend

PROC    = ROOT / "data" / "processed"
OUT_DIR = PROC / "lbm_coupled"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIZ_DIR = PROC / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Physical constants
VADM_REF  = 8.0     # × 10^22 A·m²  (Holocene mean from Sint-2000)
KM_PER_LL = 111.0   # km per degree latitude


# ---------------------------------------------------------------------------
# Load + interpolate VADM
# ---------------------------------------------------------------------------

def load_vadm_series(t_max_bp: float, dt: float = 100.0) -> tuple:
    """
    Load Sint-2000 VADM from processed parquet.
    Returns (ages_bp array, vadm array) on a uniform dt grid from 0 to t_max_bp.
    """
    p = PROC / "sint2000_vadm.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Run fetch_data.py first: {p}")

    df = pd.read_parquet(p).sort_values("age_bp")
    ages_raw = df["age_bp"].values
    vadm_raw = df["vadm"].values

    # Interpolate to uniform grid
    ages_grid = np.arange(0, t_max_bp + dt, dt)
    vadm_grid = np.interp(ages_grid, ages_raw, vadm_raw,
                          left=vadm_raw[0], right=vadm_raw[-1])

    # Clip to physical range
    vadm_grid = np.clip(vadm_grid, 0.5, 20.0)
    return ages_grid, vadm_grid


def vadm_to_force_scale(vadm: float, F0: float, alpha: float) -> float:
    return F0 * (VADM_REF / (vadm + 1e-3)) ** alpha


# ---------------------------------------------------------------------------
# Single coupled run — returns displacement time series
# ---------------------------------------------------------------------------

def run_coupled(
    nx: int, ny: int,
    ages_bp: np.ndarray, vadm_series: np.ndarray,
    pole_shift_deg: float,
    F0: float, alpha: float,
    coupled: bool,         # True = VADM-modulated, False = control
    use_gpu: bool,
) -> np.ndarray:
    """
    Run LBM for each time step, updating force_scale from VADM.
    Returns mean displacement magnitude time series (length = n_steps).
    """
    xp = xp_backend(use_gpu)
    solver = LBMSolver(nx=nx, ny=ny, tau=0.6, xp=xp)
    solver.u_plate_x = xp.full(nx, 5e-4, dtype=np.float64)

    n_steps = len(ages_bp)
    T_MAX   = ages_bp[-1]

    # Geological time runs backward (T_MAX_BP → 0)
    # Step k corresponds to age_bp = T_MAX - k * dt
    ages_reversed = ages_bp[::-1]   # oldest first
    vadm_reversed = vadm_series[::-1]

    mean_disp = np.zeros(n_steps)
    disp_x    = np.zeros((ny, nx))
    disp_y    = np.zeros((ny, nx))

    t0 = time.time()
    for k in range(n_steps):
        age = ages_reversed[k]
        vadm_t = vadm_reversed[k]

        if coupled:
            fs = vadm_to_force_scale(vadm_t, F0, alpha)
        else:
            fs = F0   # constant — control run

        Fx, Fy = build_polar_wander_force(
            ny, nx,
            pole_shift_lat_deg=pole_shift_deg,
            force_scale=fs,
            ramp_steps=200,
            step=k,
            xp=xp,
        )
        solver.Fx = Fx
        solver.Fy = Fy
        solver.step()

        ux = solver.get_numpy(solver.ux)
        uy = solver.get_numpy(solver.uy)
        disp_x += ux
        disp_y += uy
        disp_mag = np.sqrt(disp_x**2 + disp_y**2)
        mean_disp[k] = disp_mag.mean()

    label = "coupled" if coupled else "control"
    elapsed = time.time() - t0
    print(f"  [{label}] {n_steps} steps in {elapsed:.1f}s | "
          f"peak mean disp = {mean_disp.max()*KM_PER_LL:.2f} km")
    return mean_disp


# ---------------------------------------------------------------------------
# Comparison figure — the key scientific result
# ---------------------------------------------------------------------------

EVENTS_BP = {
    "Laschamp": 41_000,
    "Mono Lake": 34_000,
    "LGM":       20_000,
    "YD":        12_900,
    "8.2 ka":     8_200,
}

def plot_comparison(
    ages_bp: np.ndarray,
    vadm_series: np.ndarray,
    disp_coupled: np.ndarray,
    disp_control: np.ndarray,
    alpha: float,
    out_path: Path,
):
    ages_reversed = ages_bp[::-1]   # oldest first (matches simulation order)

    fig, (ax_vadm, ax_force, ax_disp, ax_ratio) = plt.subplots(
        4, 1, figsize=(14, 12), sharex=True,
        gridspec_kw={"hspace": 0.06, "height_ratios": [1.5, 1, 2, 1]}
    )
    fig.patch.set_facecolor("#0F172A")
    BG = "#0F172A"

    x_ka = ages_reversed / 1000.0   # ka BP, oldest first

    # -- Panel 1: VADM (geomagnetic field strength)
    ax_vadm.set_facecolor(BG)
    ax_vadm.plot(x_ka, vadm_series[::-1], color="#FF9F0A", lw=1.2)
    ax_vadm.fill_between(x_ka, vadm_series[::-1], alpha=0.2, color="#FF9F0A")
    ax_vadm.axhline(VADM_REF, color="#475569", lw=0.7, ls="--",
                    label=f"VADM_ref = {VADM_REF}")
    ax_vadm.set_ylabel("VADM\n(10²² A·m²)", color="#94A3B8", fontsize=8)
    ax_vadm.legend(fontsize=7, framealpha=0.15, labelcolor="#E2E8F0", loc="upper right")
    ax_vadm.tick_params(colors="#64748B")
    for sp in ax_vadm.spines.values(): sp.set_edgecolor("#1E293B")

    # -- Panel 2: Derived force scale
    force_vals = np.array([vadm_to_force_scale(v, F0=5e-5, alpha=alpha)
                           for v in vadm_series[::-1]])
    ax_force.set_facecolor(BG)
    ax_force.plot(x_ka, force_vals * 1e5, color="#A78BFA", lw=1.0)
    ax_force.fill_between(x_ka, force_vals * 1e5, alpha=0.2, color="#A78BFA")
    ax_force.set_ylabel("Force scale\n(×10⁻⁵)", color="#94A3B8", fontsize=8)
    ax_force.tick_params(colors="#64748B")
    for sp in ax_force.spines.values(): sp.set_edgecolor("#1E293B")

    # -- Panel 3: Displacement comparison (main result)
    ax_disp.set_facecolor(BG)
    disp_coupled_km = disp_coupled * KM_PER_LL
    disp_control_km = disp_control * KM_PER_LL
    ax_disp.plot(x_ka, disp_coupled_km, color="#0A84FF", lw=1.4,
                 label="VADM-coupled (hypothesis)")
    ax_disp.plot(x_ka, disp_control_km, color="#64748B", lw=0.9, ls="--",
                 label="Control (constant force)", alpha=0.7)
    ax_disp.fill_between(x_ka, disp_coupled_km, disp_control_km,
                         where=disp_coupled_km > disp_control_km,
                         alpha=0.2, color="#0A84FF", label="Excess displacement")
    ax_disp.set_ylabel("Mean displacement\n(km)", color="#94A3B8", fontsize=8)
    ax_disp.legend(fontsize=8, framealpha=0.15, labelcolor="#E2E8F0", loc="upper left")
    ax_disp.tick_params(colors="#64748B")
    for sp in ax_disp.spines.values(): sp.set_edgecolor("#1E293B")

    # -- Panel 4: Ratio coupled / control (anomaly amplification)
    ratio = disp_coupled_km / (disp_control_km + 1e-9)
    ax_ratio.set_facecolor(BG)
    ax_ratio.plot(x_ka, ratio, color="#30D158", lw=1.0)
    ax_ratio.fill_between(x_ka, ratio, 1.0,
                          where=ratio > 1.0, alpha=0.25, color="#30D158")
    ax_ratio.axhline(1.0, color="#475569", lw=0.6, ls="--")
    ax_ratio.set_ylabel("Ratio\ncoupled/control", color="#94A3B8", fontsize=8)
    ax_ratio.set_xlabel("Age (ka BP)", color="#94A3B8", fontsize=9)
    ax_ratio.tick_params(colors="#64748B")
    for sp in ax_ratio.spines.values(): sp.set_edgecolor("#1E293B")

    # -- Event markers (all panels)
    for evt_name, evt_yr in EVENTS_BP.items():
        evt_ka = evt_yr / 1000.0
        if x_ka.min() <= evt_ka <= x_ka.max():
            for ax in (ax_vadm, ax_force, ax_disp, ax_ratio):
                ax.axvline(evt_ka, color="#F1FA8C", lw=0.8, ls=":", alpha=0.75)
            ax_vadm.text(evt_ka + 0.5,
                         ax_vadm.get_ylim()[1] * 0.85 if ax_vadm.get_ylim()[1] > 0 else 1,
                         evt_name, color="#F1FA8C", fontsize=6.5,
                         rotation=90, va="top")

    ax_disp.invert_xaxis()
    fig.suptitle(
        f"POLE_SHIFT_SIM — VADM-Coupled LBM  [α={alpha}, nx={nx_g}×{ny_g}]",
        color="#E2E8F0", fontsize=11, fontweight="bold", y=0.999
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] → {out_path}")


# ---------------------------------------------------------------------------
# Report: peak anomalies
# ---------------------------------------------------------------------------

def print_report(ages_reversed, disp_coupled, disp_control, alpha):
    ratio = (disp_coupled * KM_PER_LL) / (disp_control * KM_PER_LL + 1e-9)
    disp_km = disp_coupled * KM_PER_LL

    print(f"\n{'='*60}")
    print(f"POLE_SHIFT_SIM — Coupling Report  (alpha={alpha})")
    print(f"{'='*60}")
    print(f"{'Age (ka BP)':>12}  {'Disp_coupled (km)':>18}  {'Ratio vs ctrl':>14}  Event")
    print("-" * 60)

    # Sort by ratio descending
    top_idx = np.argsort(ratio)[::-1][:15]
    printed = set()
    for idx in top_idx:
        age = ages_reversed[idx]
        age_ka = age / 1000
        tag = ""
        for evt_name, evt_yr in EVENTS_BP.items():
            if abs(age - evt_yr) <= 3000:
                tag = f"← {evt_name}"
        key = round(age / 2000)    # deduplicate nearby windows
        if key in printed:
            continue
        printed.add(key)
        print(f"  {age_ka:8.1f} ka    {disp_km[idx]:12.3f} km    "
              f"  {ratio[idx]:8.3f}x    {tag}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Module-level grid size for plot title (set in main)
nx_g, ny_g = 90, 45

def main():
    global nx_g, ny_g
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx",          type=int,   default=90)
    parser.add_argument("--ny",          type=int,   default=45)
    parser.add_argument("--t-max-bp",    type=float, default=80_000.0)
    parser.add_argument("--dt",          type=float, default=100.0,
                        help="LBM time step in years (default 100)")
    parser.add_argument("--pole-shift",  type=float, default=30.0)
    parser.add_argument("--F0",          type=float, default=5e-5)
    parser.add_argument("--alpha",       type=float, default=1.0,
                        help="VADM coupling exponent (1.0 = linear)")
    parser.add_argument("--gpu",         action="store_true")
    args = parser.parse_args()

    nx_g, ny_g = args.nx, args.ny

    print(f"[config] grid {args.nx}×{args.ny} | "
          f"t_max={args.t_max_bp/1000:.0f} ka | "
          f"dt={args.dt:.0f} yr | α={args.alpha}")

    # Load VADM
    print("[VADM]  Loading Sint-2000 …")
    ages_bp, vadm_series = load_vadm_series(args.t_max_bp, args.dt)
    n_steps = len(ages_bp)
    print(f"        {n_steps} time steps | "
          f"VADM range [{vadm_series.min():.2f}, {vadm_series.max():.2f}] × 10²² A·m²")

    # Laschamp: VADM minimum
    laschamp_idx = np.argmin(np.abs(ages_bp - 41_000))
    print(f"        VADM at Laschamp (~41 ka): {vadm_series[laschamp_idx]:.2f} × 10²² A·m²")
    fs_laschamp = vadm_to_force_scale(vadm_series[laschamp_idx], args.F0, args.alpha)
    print(f"        Force scale at Laschamp:   {fs_laschamp:.2e}  "
          f"(×{fs_laschamp/args.F0:.2f} vs reference)")

    # Run coupled
    print("\n[RUN 1] VADM-coupled simulation …")
    disp_coupled = run_coupled(
        args.nx, args.ny, ages_bp, vadm_series,
        args.pole_shift, args.F0, args.alpha,
        coupled=True, use_gpu=args.gpu,
    )

    # Run control
    print("[RUN 2] Control (constant force) …")
    disp_control = run_coupled(
        args.nx, args.ny, ages_bp, vadm_series,
        args.pole_shift, args.F0, args.alpha,
        coupled=False, use_gpu=args.gpu,
    )

    # Save
    np.save(OUT_DIR / "ages_reversed.npy", ages_bp[::-1])
    np.save(OUT_DIR / "disp_coupled.npy",  disp_coupled)
    np.save(OUT_DIR / "disp_control.npy",  disp_control)
    np.save(OUT_DIR / "vadm_reversed.npy", vadm_series[::-1])

    # Report
    print_report(ages_bp[::-1], disp_coupled, disp_control, args.alpha)

    # Plot
    print("\n[plot] Comparison figure …")
    plot_comparison(
        ages_bp, vadm_series,
        disp_coupled, disp_control,
        alpha=args.alpha,
        out_path=VIZ_DIR / "vadm_coupled_comparison.png",
    )

    print(f"\n[done] Results in {OUT_DIR}")
    print("       Increase --nx 360 --ny 180 --gpu for full resolution.")


if __name__ == "__main__":
    main()
