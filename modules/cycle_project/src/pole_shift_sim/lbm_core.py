"""
lbm_core.py — POLE_SHIFT_SIM: D2Q9 Lattice Boltzmann Engine

Models viscous flow in the asthenosphere driven by lithosphere displacement.
The lithosphere is treated as a moving top boundary; the asthenosphere is
the fluid domain (η ~ 10^19–10^21 Pa·s at geological timescales).

Coordinate system:
  x → longitude index  (0..NX-1),  periodic BC
  y → latitude index   (0..NY-1),  no-slip at poles

D2Q9 velocity set (lattice units):
  e_0 = (0,0), rest
  e_1..4 = cardinal  (±1,0), (0,±1)
  e_5..8 = diagonal  (±1,±1)

BGK collision operator:
  Ω_i = -(f_i - f_eq_i) / τ
  where τ controls viscosity: ν = cs² (τ - 0.5)

External body force (polar wander trigger):
  Applied as Guo body-force correction to f_eq.
  Direction and magnitude configurable — mimics centrifugal imbalance
  when the ice cap is displaced from the rotational pole.

Units: lattice units throughout. Physical scaling in visualize.py.

Usage (standalone test):
  python src/pole_shift_sim/lbm_core.py --steps 500 --nx 90 --ny 45
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Try CuPy; fall back to numpy transparently
try:
    import cupy as cp
    _HAS_CUPY = True
except ImportError:
    cp = None
    _HAS_CUPY = False


def xp_backend(use_gpu: bool):
    """Return cupy if available and requested, else numpy."""
    if use_gpu and _HAS_CUPY:
        print("[backend] CuPy / CUDA")
        return cp
    print(f"[backend] NumPy (cupy={'not installed' if not _HAS_CUPY else 'disabled'})")
    return np


# ---------------------------------------------------------------------------
# D2Q9 constants
# ---------------------------------------------------------------------------
# Velocity vectors e_i  (shape: 9 × 2)
E = np.array([
    [0,  0],   # 0 rest
    [1,  0],   # 1 E
    [0,  1],   # 2 N
    [-1, 0],   # 3 W
    [0, -1],   # 4 S
    [1,  1],   # 5 NE
    [-1, 1],   # 6 NW
    [-1,-1],   # 7 SW
    [1, -1],   # 8 SE
], dtype=np.float64)

W = np.array([
    4/9,                        # 0
    1/9, 1/9, 1/9, 1/9,         # 1-4
    1/36, 1/36, 1/36, 1/36,     # 5-8
], dtype=np.float64)

# Opposite direction index (for bounce-back BC)
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

CS2 = 1.0 / 3.0   # speed of sound squared (lattice units)


# ---------------------------------------------------------------------------
# LBM core
# ---------------------------------------------------------------------------

class LBMSolver:
    """
    D2Q9 BGK LBM solver with:
      - Periodic BC in x (longitude)
      - No-slip (bounce-back) at y=0 (S pole) and y=NY-1 (N pole)
      - Moving top boundary (lithosphere plate)
      - Guo body-force scheme for external forcing

    Grid layout: f[q, y, x]  — q=9 populations, y=lat, x=lon
    """

    def __init__(
        self,
        nx: int,
        ny: int,
        tau: float = 0.6,
        xp=np,
    ):
        self.nx  = nx
        self.ny  = ny
        self.tau = tau
        self.xp  = xp

        # Population array f[q, y, x]
        self.f = xp.ones((9, ny, nx), dtype=np.float64) * xp.array(W)[:, None, None]

        # Macroscopic fields
        self.rho = xp.ones((ny, nx), dtype=np.float64)
        self.ux  = xp.zeros((ny, nx), dtype=np.float64)
        self.uy  = xp.zeros((ny, nx), dtype=np.float64)

        # Body force (lat, lon) — set externally
        self.Fx  = xp.zeros((ny, nx), dtype=np.float64)
        self.Fy  = xp.zeros((ny, nx), dtype=np.float64)

        # Lithosphere plate velocity (top boundary, y=NY-1)
        self.u_plate_x = xp.zeros(nx, dtype=np.float64)
        self.u_plate_y = xp.zeros(nx, dtype=np.float64)

        # History buffers (stored as numpy for plotting)
        self.history_ux  = []
        self.history_uy  = []
        self.history_disp_x = []
        self.history_disp_y = []

        # Accumulated displacement (lattice units)
        self.disp_x = np.zeros((ny, nx), dtype=np.float64)
        self.disp_y = np.zeros((ny, nx), dtype=np.float64)

        # Lattice E and W tensors on device
        self._E  = xp.array(E)      # (9, 2)
        self._W  = xp.array(W)      # (9,)

    # ------------------------------------------------------------------
    # Macroscopic update
    # ------------------------------------------------------------------

    def _macro(self):
        xp = self.xp
        self.rho = self.f.sum(axis=0)
        # Guo: effective velocity = (Σ e_i f_i + F/2) / ρ
        self.ux = (
            (self.f * self._E[:, 0, None, None]).sum(axis=0)
            + 0.5 * self.Fx
        ) / (self.rho + 1e-12)
        self.uy = (
            (self.f * self._E[:, 1, None, None]).sum(axis=0)
            + 0.5 * self.Fy
        ) / (self.rho + 1e-12)

    # ------------------------------------------------------------------
    # Equilibrium
    # ------------------------------------------------------------------

    def _feq(self, rho, ux, uy):
        xp = self.xp
        # eu[q, y, x] = e_i · u
        eu = (self._E[:, 0, None, None] * ux[None] +
              self._E[:, 1, None, None] * uy[None])
        u2 = ux**2 + uy**2
        feq = self._W[:, None, None] * rho[None] * (
            1.0 + eu / CS2
            + eu**2 / (2 * CS2**2)
            - u2[None] / (2 * CS2)
        )
        return feq

    # ------------------------------------------------------------------
    # Guo body-force correction
    # ------------------------------------------------------------------

    def _guo_force(self):
        xp = self.xp
        # S_i = (1 - 1/(2τ)) * w_i * [(e_i - u)/cs² + (e_i·u)/cs⁴ * e_i] · F
        eu = (self._E[:, 0, None, None] * self.ux[None] +
              self._E[:, 1, None, None] * self.uy[None])
        ex_F = self._E[:, 0, None, None] * self.Fx[None]
        ey_F = self._E[:, 1, None, None] * self.Fy[None]
        S = (1.0 - 1.0 / (2 * self.tau)) * self._W[:, None, None] * (
            (self._E[:, 0, None, None] - self.ux[None]) * self.Fx[None] / CS2
            + (self._E[:, 1, None, None] - self.uy[None]) * self.Fy[None] / CS2
            + eu / (CS2**2) * (ex_F + ey_F)
        )
        return S

    # ------------------------------------------------------------------
    # Collision
    # ------------------------------------------------------------------

    def _collide(self):
        feq = self._feq(self.rho, self.ux, self.uy)
        S   = self._guo_force()
        self.f = self.f - (self.f - feq) / self.tau + S

    # ------------------------------------------------------------------
    # Streaming (periodic in x, walls at y=0 and y=NY-1)
    # ------------------------------------------------------------------

    def _stream(self):
        xp = self.xp
        f_new = xp.zeros_like(self.f)
        for q in range(9):
            ex, ey = int(E[q, 0]), int(E[q, 1])
            # Roll in both directions: periodic in x
            f_new[q] = xp.roll(xp.roll(self.f[q], ex, axis=1), ey, axis=0)
        self.f = f_new

    # ------------------------------------------------------------------
    # Boundary conditions
    # ------------------------------------------------------------------

    def _apply_bc(self):
        xp = self.xp
        # South pole (y=0): full bounce-back
        for q in range(9):
            self.f[q, 0, :] = self.f[OPP[q], 1, :]

        # North pole (y=NY-1): full bounce-back
        for q in range(9):
            self.f[q, -1, :] = self.f[OPP[q], -2, :]

        # Moving lithosphere plate at top: Zou-He moving wall
        # Simplified: half-way bounce-back with plate velocity correction
        # Only for populations that hit the top wall (ey = -1: q=4,7,8 going S)
        # Target: f going back up (q=2,5,6) with plate velocity contribution
        rho_w = (
            self.f[0, -2, :] + self.f[1, -2, :] + self.f[3, -2, :]
            + 2.0 * (self.f[2, -2, :] + self.f[5, -2, :] + self.f[6, -2, :])
        ) / (1.0 + self.uy[-2, :] + 1e-12)
        # North-going populations at wall row:
        self.f[4, -1, :] = self.f[2, -2, :] - (2.0/3.0) * rho_w * self.u_plate_y
        self.f[7, -1, :] = (self.f[5, -2, :]
                             - 0.5 * (self.f[1, -2, :] - self.f[3, -2, :])
                             - (1.0/6.0) * rho_w * self.u_plate_x
                             - (1.0/6.0) * rho_w * self.u_plate_y)
        self.f[8, -1, :] = (self.f[6, -2, :]
                             + 0.5 * (self.f[1, -2, :] - self.f[3, -2, :])
                             + (1.0/6.0) * rho_w * self.u_plate_x
                             - (1.0/6.0) * rho_w * self.u_plate_y)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(self):
        self._macro()
        self._collide()
        self._stream()
        self._apply_bc()

    def get_numpy(self, arr):
        """Transfer device array to host numpy."""
        if _HAS_CUPY and self.xp is cp:
            return cp.asnumpy(arr)
        return np.asarray(arr)

    def record(self):
        """Save snapshot of macroscopic fields (numpy)."""
        ux = self.get_numpy(self.ux)
        uy = self.get_numpy(self.uy)
        self.disp_x += ux
        self.disp_y += uy
        self.history_ux.append(ux.copy())
        self.history_uy.append(uy.copy())
        self.history_disp_x.append(self.disp_x.copy())
        self.history_disp_y.append(self.disp_y.copy())


# ---------------------------------------------------------------------------
# Physical setup: crustal displacement scenario
# ---------------------------------------------------------------------------

def build_polar_wander_force(
    ny: int,
    nx: int,
    pole_shift_lat_deg: float = 30.0,
    pole_shift_lon_deg: float = 0.0,
    force_scale: float = 1e-4,
    ramp_steps: int = 200,
    step: int = 0,
    xp=np,
) -> tuple:
    """
    Centrifugal body force field for a pole shifted by `pole_shift_lat_deg`.
    Direction: toward the old equatorial bulge (poleward in lat, fixed lon).
    Magnitude ramps up over `ramp_steps` steps (simulating gradual trigger).

    Returns Fx, Fy arrays on the compute device.
    """
    lats = np.linspace(-90, 90, ny)    # degrees
    lons = np.linspace(0, 360, nx)

    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    LON, LAT = np.meshgrid(lon_rad, lat_rad)

    # Force directed toward rotated pole
    # Simple model: Fx ∝ sin(lon - shift_lon), Fy ∝ -sin(lat - shift_lat)
    shift_lat = np.deg2rad(pole_shift_lat_deg)
    shift_lon = np.deg2rad(pole_shift_lon_deg)

    Fx_np = force_scale * np.sin(LON - shift_lon) * np.cos(LAT)
    Fy_np = force_scale * (-np.sin(LAT - shift_lat))

    # Ramp factor
    ramp = min(1.0, step / max(ramp_steps, 1))

    return xp.array(Fx_np * ramp), xp.array(Fy_np * ramp)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def run(nx, ny, steps, tau, record_every, use_gpu, pole_shift_deg, out_dir):
    xp = xp_backend(use_gpu)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[grid]   {nx} × {ny}  (lon × lat)")
    print(f"[tau]    {tau}  →  ν = {CS2*(tau-0.5):.4f} (lattice units)")
    print(f"[steps]  {steps}  (record every {record_every})")
    print(f"[pole]   shift = {pole_shift_deg}°")

    solver = LBMSolver(nx=nx, ny=ny, tau=tau, xp=xp)

    # Plate moves eastward at small velocity (initial trigger)
    solver.u_plate_x = xp.full(nx, 1e-3, dtype=np.float64)

    t0 = time.time()
    for step in range(steps):
        # Update body force (ramps up over first 200 steps)
        Fx, Fy = build_polar_wander_force(
            ny, nx,
            pole_shift_lat_deg=pole_shift_deg,
            force_scale=5e-5,
            ramp_steps=200,
            step=step,
            xp=xp,
        )
        solver.Fx = Fx
        solver.Fy = Fy
        solver.step()

        if step % record_every == 0:
            solver.record()
            elapsed = time.time() - t0
            mlups = (nx * ny * step + 1) / elapsed / 1e6
            ux_max = float(xp.abs(solver.ux).max())
            print(f"  step {step:5d}/{steps}  |ux|_max={ux_max:.2e}"
                  f"  {mlups:.1f} MLUPS")

    # Save final displacement field
    disp_mag = np.sqrt(solver.disp_x**2 + solver.disp_y**2)
    np.save(out_dir / "disp_x.npy",   solver.disp_x)
    np.save(out_dir / "disp_y.npy",   solver.disp_y)
    np.save(out_dir / "disp_mag.npy", disp_mag)
    np.save(out_dir / "ux_final.npy", solver.get_numpy(solver.ux))
    np.save(out_dir / "uy_final.npy", solver.get_numpy(solver.uy))
    np.save(out_dir / "history_ux.npy",    np.array(solver.history_ux))
    np.save(out_dir / "history_disp_x.npy", np.array(solver.history_disp_x))

    total_time = time.time() - t0
    print(f"\n[done]  {total_time:.1f}s | "
          f"peak |u| = {float(xp.abs(solver.ux).max()):.3e}")
    print(f"[saved] {out_dir}/")
    return solver


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="POLE_SHIFT_SIM: D2Q9 LBM")
    parser.add_argument("--nx",           type=int,   default=90)
    parser.add_argument("--ny",           type=int,   default=45)
    parser.add_argument("--steps",        type=int,   default=500)
    parser.add_argument("--tau",          type=float, default=0.6)
    parser.add_argument("--record-every", type=int,   default=50)
    parser.add_argument("--pole-shift",   type=float, default=30.0,
                        help="Pole shift angle in degrees")
    parser.add_argument("--gpu",          action="store_true")
    args = parser.parse_args()

    out_dir = (Path(__file__).resolve().parent.parent.parent
               / "data" / "processed" / "lbm_output")
    run(
        nx=args.nx, ny=args.ny,
        steps=args.steps, tau=args.tau,
        record_every=args.record_every,
        use_gpu=args.gpu,
        pole_shift_deg=args.pole_shift,
        out_dir=out_dir,
    )
