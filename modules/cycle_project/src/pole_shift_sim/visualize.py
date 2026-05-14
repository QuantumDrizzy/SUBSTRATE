"""
visualize.py — POLE_SHIFT_SIM: 2D + 3D visualization

2D:  matplotlib — lat/lon displacement maps, velocity quivers, time series
3D:  PyVista    — displacement field projected on a sphere (if pyvista installed)

Physical scaling:
  1 lattice length = ~111 km (1 degree of arc at Earth's surface)
  1 time step      = ~1000 years (geological LBM)
  velocity in lattice units → multiply by 111 km / 1000 yr = 0.111 m/yr

Usage:
  python src/pole_shift_sim/visualize.py [--no-3d] [--animate]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

ROOT    = Path(__file__).resolve().parent.parent.parent
LBM_DIR = ROOT / "data" / "processed" / "lbm_output"
VIZ_DIR = ROOT / "data" / "processed" / "viz"
VIZ_DIR.mkdir(parents=True, exist_ok=True)

# Physical scaling factors
KM_PER_LATTICE    = 111.0          # km per degree
YR_PER_STEP       = 1_000.0        # years per LBM step
M_PER_KM          = 1_000.0
VEL_SCALE         = KM_PER_LATTICE * M_PER_KM / YR_PER_STEP  # m/yr per lattice vel unit


def load_fields(lbm_dir: Path) -> dict:
    fields = {}
    expected = ["disp_x", "disp_y", "disp_mag", "ux_final", "uy_final"]
    for name in expected:
        p = lbm_dir / f"{name}.npy"
        if p.exists():
            fields[name] = np.load(p)
        else:
            print(f"  [warn] {name}.npy not found")
    # History
    for name in ["history_ux", "history_disp_x"]:
        p = lbm_dir / f"{name}.npy"
        if p.exists():
            fields[name] = np.load(p)
    return fields


# ---------------------------------------------------------------------------
# 2D: Displacement map
# ---------------------------------------------------------------------------

def plot_displacement_map(fields: dict, out_path: Path):
    disp_mag  = fields.get("disp_mag")
    disp_x    = fields.get("disp_x")
    disp_y    = fields.get("disp_y")
    ux_final  = fields.get("ux_final")
    uy_final  = fields.get("uy_final")

    if disp_mag is None:
        print("[skip] displacement map — no data")
        return

    ny, nx = disp_mag.shape
    lons = np.linspace(0, 360, nx)
    lats = np.linspace(-90, 90, ny)
    LON, LAT = np.meshgrid(lons, lats)

    # Physical scale
    disp_km = disp_mag * KM_PER_LATTICE

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor("#0F172A")

    # -- Left: accumulated displacement magnitude
    ax = axes[0]
    ax.set_facecolor("#0F172A")
    im = ax.contourf(LON, LAT, disp_km, levels=50, cmap="plasma")
    plt.colorbar(im, ax=ax, label="Displacement (km)", fraction=0.03)

    # Velocity quiver (subsample)
    step_q = max(1, nx // 20)
    if ux_final is not None and uy_final is not None:
        ax.quiver(
            LON[::step_q, ::step_q],
            LAT[::step_q, ::step_q],
            ux_final[::step_q, ::step_q],
            uy_final[::step_q, ::step_q],
            color="#38BDF8", alpha=0.7, scale=0.05, width=0.003,
        )
    ax.set_xlabel("Longitude (°)", color="#94A3B8")
    ax.set_ylabel("Latitude (°)",  color="#94A3B8")
    ax.set_title("Accumulated Displacement + Velocity Field",
                 color="#E2E8F0", fontsize=10)
    ax.tick_params(colors="#64748B")
    for sp in ax.spines.values(): sp.set_edgecolor("#1E293B")

    # Grid lines
    ax.axhline(0,  color="#334155", lw=0.5, ls="--")
    ax.axhline(23.5,  color="#334155", lw=0.4, ls=":")
    ax.axhline(-23.5, color="#334155", lw=0.4, ls=":")

    # -- Right: X-component (longitude shift)
    ax2 = axes[1]
    ax2.set_facecolor("#0F172A")
    norm_x = Normalize(vmin=disp_x.min() if disp_x is not None else 0,
                       vmax=disp_x.max() if disp_x is not None else 1)
    if disp_x is not None:
        im2 = ax2.contourf(LON, LAT, disp_x * KM_PER_LATTICE,
                           levels=50, cmap="RdBu_r")
        plt.colorbar(im2, ax=ax2, label="E–W Displacement (km)", fraction=0.03)
    ax2.set_xlabel("Longitude (°)", color="#94A3B8")
    ax2.set_title("E–W Component (Lithosphere Zonal Drift)",
                  color="#E2E8F0", fontsize=10)
    ax2.tick_params(colors="#64748B")
    for sp in ax2.spines.values(): sp.set_edgecolor("#1E293B")

    fig.suptitle("POLE_SHIFT_SIM — Lithosphere Displacement [D2Q9 LBM]",
                 color="#E2E8F0", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [2D] → {out_path}")


# ---------------------------------------------------------------------------
# 2D: Time evolution
# ---------------------------------------------------------------------------

def plot_time_evolution(fields: dict, out_path: Path):
    hist = fields.get("history_ux")
    hist_d = fields.get("history_disp_x")
    if hist is None:
        print("[skip] time evolution — no history")
        return

    n_frames, ny, nx = hist.shape
    lats = np.linspace(-90, 90, ny)

    # Mean velocity at each latitude, over time
    mean_lat_vel = hist.mean(axis=2)   # (n_frames, ny)
    mean_lat_disp = hist_d.mean(axis=2) if hist_d is not None else None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0F172A")

    t_axis = np.arange(n_frames)

    # Hovmöller diagram: lat vs time, color = mean zonal velocity
    ax1.set_facecolor("#0F172A")
    im = ax1.contourf(t_axis, lats, mean_lat_vel.T, levels=40, cmap="seismic")
    plt.colorbar(im, ax=ax1, label="Mean zonal vel (lattice)", fraction=0.03)
    ax1.set_xlabel("Time step (×record_interval)", color="#94A3B8")
    ax1.set_ylabel("Latitude (°)", color="#94A3B8")
    ax1.set_title("Hovmöller: Zonal Velocity", color="#E2E8F0", fontsize=10)
    ax1.tick_params(colors="#64748B")
    for sp in ax1.spines.values(): sp.set_edgecolor("#1E293B")

    # Velocity profile snapshots
    ax2.set_facecolor("#0F172A")
    cmap = plt.get_cmap("viridis")
    snap_idx = np.linspace(0, n_frames - 1, min(8, n_frames), dtype=int)
    for i, idx in enumerate(snap_idx):
        color = cmap(i / len(snap_idx))
        ax2.plot(mean_lat_vel[idx] * VEL_SCALE, lats,
                 color=color, lw=1.2, alpha=0.85,
                 label=f"t={idx}")
    ax2.set_xlabel("Zonal velocity (m/yr)", color="#94A3B8")
    ax2.set_ylabel("Latitude (°)", color="#94A3B8")
    ax2.set_title("Velocity Profile Snapshots", color="#E2E8F0", fontsize=10)
    ax2.axvline(0, color="#334155", lw=0.5, ls="--")
    ax2.legend(fontsize=7, framealpha=0.15, labelcolor="#E2E8F0", loc="lower right")
    ax2.tick_params(colors="#64748B")
    for sp in ax2.spines.values(): sp.set_edgecolor("#1E293B")

    fig.suptitle("POLE_SHIFT_SIM — Temporal Evolution",
                 color="#E2E8F0", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [2D] → {out_path}")


# ---------------------------------------------------------------------------
# 3D: Sphere (PyVista)
# ---------------------------------------------------------------------------

def plot_sphere_3d(fields: dict, out_path: Path):
    try:
        import pyvista as pv
    except ImportError:
        print("[skip] 3D sphere — pyvista not installed "
              "(pip install pyvista vtk)")
        return

    disp_mag = fields.get("disp_mag")
    if disp_mag is None:
        return

    ny, nx = disp_mag.shape

    # Build sphere mesh
    sphere = pv.Sphere(
        radius=1.0,
        theta_resolution=nx,
        phi_resolution=ny,
        start_theta=0, end_theta=360,
        start_phi=0, end_phi=180,
    )

    # Map displacement onto sphere surface
    # PyVista sphere points: (phi=lat from N, theta=lon)
    pts = sphere.points                  # (N_pts, 3)
    N_pts = pts.shape[0]

    # Compute lat/lon for each point
    r   = np.linalg.norm(pts, axis=1)
    lat = np.degrees(np.arcsin(pts[:, 2] / (r + 1e-9)))   # -90..90
    lon = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 360

    # Interpolate disp_mag onto sphere points
    lat_idx = ((lat + 90) / 180 * (ny - 1)).astype(int).clip(0, ny - 1)
    lon_idx = (lon / 360 * (nx - 1)).astype(int).clip(0, nx - 1)
    disp_on_sphere = disp_mag[lat_idx, lon_idx] * KM_PER_LATTICE

    sphere["displacement_km"] = disp_on_sphere

    # Warp sphere radially by displacement
    warp_scale = 0.1 / (disp_on_sphere.max() + 1e-9)
    normals = pts / (r[:, None] + 1e-9)
    warped_pts = pts + normals * disp_on_sphere[:, None] * warp_scale
    warped = sphere.copy()
    warped.points = warped_pts
    warped["displacement_km"] = disp_on_sphere

    pl = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    pl.set_background("#0F172A")
    pl.add_mesh(
        warped,
        scalars="displacement_km",
        cmap="plasma",
        smooth_shading=True,
        show_scalar_bar=True,
        scalar_bar_args={
            "title": "Displacement (km)",
            "color": "#E2E8F0",
            "fmt": "%.1f",
        },
    )
    # Axis indicator
    pl.add_axes(color="#94A3B8")
    pl.camera.position = (3.5, 1.5, 1.5)
    pl.camera.focal_point = (0, 0, 0)

    pl.screenshot(str(out_path), transparent_background=False)
    pl.close()
    print(f"  [3D] → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-3d",   action="store_true")
    parser.add_argument("--lbm-dir", type=str, default=str(LBM_DIR))
    args = parser.parse_args()

    lbm_dir = Path(args.lbm_dir)
    print(f"[load] {lbm_dir}")
    fields = load_fields(lbm_dir)

    if not fields:
        print("[ERROR] No LBM output found. Run lbm_core.py first.")
        sys.exit(1)

    print("[2D]  Displacement map …")
    plot_displacement_map(fields, VIZ_DIR / "displacement_map.png")

    print("[2D]  Time evolution …")
    plot_time_evolution(fields, VIZ_DIR / "time_evolution.png")

    if not args.no_3d:
        print("[3D]  Sphere projection …")
        plot_sphere_3d(fields, VIZ_DIR / "sphere_3d.png")

    print(f"\n[done] viz/ → {VIZ_DIR}")


if __name__ == "__main__":
    main()
