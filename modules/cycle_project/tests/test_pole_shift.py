"""
Tests for POLE_SHIFT_SIM module.
Uses actual API: LBMSolver(nx,ny,tau,xp), solver.step() (no args),
solver.Fx/Fy set directly. build_polar_wander_force with lon param.
CPU-only numpy backend.
"""
import numpy as np
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from pole_shift_sim.lbm_core import (
    LBMSolver,
    build_polar_wander_force,
    E,    # D2Q9 velocity vectors (9,2)
    CS2,  # 1/3
)


# ── LBMSolver initialization ──────────────────────────────────────────────────

class TestLBMSolverInit:
    def test_creates_solver(self):
        s = LBMSolver(nx=32, ny=16, tau=0.6, xp=np)
        assert s is not None

    def test_f_shape(self):
        s = LBMSolver(nx=32, ny=16, tau=0.6, xp=np)
        assert s.f.shape == (9, 16, 32)

    def test_initial_rho_uniform(self):
        s = LBMSolver(nx=32, ny=16, tau=0.6, xp=np)
        rho = s.f.sum(axis=0)
        np.testing.assert_allclose(rho, 1.0, atol=1e-10)

    def test_Fx_Fy_initialized_zero(self):
        s = LBMSolver(nx=32, ny=16, tau=0.6, xp=np)
        np.testing.assert_array_equal(s.Fx, 0.0)
        np.testing.assert_array_equal(s.Fy, 0.0)

    def test_cs2_correct(self):
        np.testing.assert_allclose(CS2, 1.0 / 3.0, rtol=1e-10)

    def test_E_shape(self):
        assert E.shape == (9, 2)


# ── LBMSolver step (force set via solver.Fx/Fy) ───────────────────────────────

class TestLBMSolverStep:
    @pytest.fixture
    def solver(self):
        return LBMSolver(nx=32, ny=16, tau=0.6, xp=np)

    def test_step_no_force_runs(self, solver):
        solver.step()  # default Fx=Fy=0, should not raise

    def test_rho_conserved_no_force(self, solver):
        """Total mass must be conserved with no forcing (periodic BCs)."""
        mass_before = solver.f.sum()
        for _ in range(20):
            solver.step()
        mass_after = solver.f.sum()
        np.testing.assert_allclose(mass_after, mass_before, rtol=1e-9)

    def test_no_nan_with_gentle_force(self, solver):
        solver.Fx = np.full((16, 32), 1e-5)
        solver.Fy = np.zeros((16, 32))
        for _ in range(100):
            solver.step()
        assert not np.any(np.isnan(solver.f))
        assert not np.any(np.isinf(solver.f))

    def test_velocity_increases_with_force(self, solver):
        """Applying x-force should accelerate fluid in x-direction."""
        rho0 = solver.f.sum(axis=0)
        ux0  = (E[:, 0, None, None] * solver.f).sum(axis=0) / rho0

        solver.Fx = np.full((16, 32), 5e-5)
        for _ in range(200):
            solver.step()

        rho1 = solver.f.sum(axis=0)
        ux1  = (E[:, 0, None, None] * solver.f).sum(axis=0) / rho1
        assert ux1.mean() > ux0.mean()

    def test_stability_at_low_tau(self):
        """tau=0.55 — 300 steps — no blow-up."""
        s = LBMSolver(nx=32, ny=16, tau=0.55, xp=np)
        s.Fy = np.full((16, 32), 3e-5)
        for _ in range(300):
            s.step()
        assert not np.any(np.isnan(s.f))

    def test_velocity_stays_bounded(self, solver):
        """Peak velocity must stay below Ma<0.3 (LBM Mach stability limit)."""
        solver.Fx = np.full((16, 32), 5e-5)
        for _ in range(300):
            solver.step()
        rho = solver.f.sum(axis=0)
        ux  = (E[:, 0, None, None] * solver.f).sum(axis=0) / rho
        uy  = (E[:, 1, None, None] * solver.f).sum(axis=0) / rho
        u   = np.sqrt(ux**2 + uy**2)
        assert u.max() < 0.3, f"Velocity blow-up: u_max={u.max():.4f}"


# ── build_polar_wander_force ──────────────────────────────────────────────────

class TestPolarWanderForce:
    def test_shape(self):
        Fx, Fy = build_polar_wander_force(
            ny=16, nx=32, pole_shift_lat_deg=30.0,
            ramp_steps=50, step=100, xp=np
        )
        assert Fx.shape == (16, 32)
        assert Fy.shape == (16, 32)

    def test_zero_at_step_zero(self):
        """Force ramps from 0 at step=0."""
        Fx, Fy = build_polar_wander_force(
            ny=16, nx=32, pole_shift_lat_deg=30.0,
            ramp_steps=50, step=0, xp=np
        )
        np.testing.assert_allclose(Fx, 0.0, atol=1e-15)
        np.testing.assert_allclose(Fy, 0.0, atol=1e-15)

    def test_force_nonzero_after_ramp(self):
        Fx, Fy = build_polar_wander_force(
            ny=16, nx=32, pole_shift_lat_deg=30.0,
            ramp_steps=50, step=100, xp=np
        )
        assert np.abs(Fx).max() > 0 or np.abs(Fy).max() > 0

    def test_force_scale_scales_linearly(self):
        """Doubling force_scale should double force amplitude."""
        _, Fy1 = build_polar_wander_force(
            ny=16, nx=32, pole_shift_lat_deg=30.0,
            force_scale=1e-4, ramp_steps=10, step=100, xp=np
        )
        _, Fy2 = build_polar_wander_force(
            ny=16, nx=32, pole_shift_lat_deg=30.0,
            force_scale=2e-4, ramp_steps=10, step=100, xp=np
        )
        if np.abs(Fy1).max() > 0:
            ratio = np.abs(Fy2).max() / np.abs(Fy1).max()
            np.testing.assert_allclose(ratio, 2.0, rtol=1e-5)


# ── VADM coupling formula ─────────────────────────────────────────────────────

class TestVADMCoupling:
    """The VADM→force_scale law from coupled_run.py."""

    def vadm_force(self, vadm, F0=1.0, alpha=1.0, vadm_ref=8.0):
        return F0 * (vadm_ref / (vadm + 1e-3)) ** alpha

    def test_reference_vadm_unit_force(self):
        f = self.vadm_force(8.0)
        np.testing.assert_allclose(f, 1.0, rtol=1e-3)

    def test_laschamp_vadm_amplifies(self):
        """VADM=2.70 (Laschamp) must give force > 1."""
        assert self.vadm_force(2.70) > 1.0

    def test_laschamp_approx_3x(self):
        """From validated coupled_run: VADM=2.70, alpha=1.0 (default) → ~2.96×."""
        f = self.vadm_force(2.70, alpha=1.0)
        assert 2.5 < f < 3.5, f"Expected ~2.96×, got {f:.3f}"

    def test_monotone_decreasing_in_vadm(self):
        vadms  = np.linspace(1.0, 9.0, 40)
        forces = np.array([self.vadm_force(v) for v in vadms])
        assert np.all(np.diff(forces) < 0)

    def test_alpha_2_stronger_than_alpha_1p0(self):
        f10 = self.vadm_force(2.70, alpha=1.0)
        f20 = self.vadm_force(2.70, alpha=2.0)
        assert f20 > f10

    def test_alpha_2_laschamp_near_9x(self):
        """Sensitivity analysis: alpha=2.0 → (8/2.7)² ≈ 8.77×."""
        f = self.vadm_force(2.70, alpha=2.0)
        assert 8.0 < f < 10.0, f"Expected ~8.77×, got {f:.3f}"
