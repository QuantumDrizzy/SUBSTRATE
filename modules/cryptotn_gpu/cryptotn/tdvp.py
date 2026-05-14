"""
cryptotn/tdvp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
time-dependent variational principle (TDVP) integrators.

two backends:
  ExactSolver  — scipy.linalg.expm / odeint; exact, for N ≤ 20 spins
  MpsSolver    — quimb MPS/MPO TDVP; χ-truncated, for N ≤ 60 spins

all solvers:
  - log benchmark data to file from first call (bare-metal rule)
  - return (t_array, P_S(t), trace(t)) for consistency
  - accept chi parameter (ignored in ExactSolver, used in MpsSolver)

units: time in μs, rates in μs⁻¹, energies in MHz (→ 2π × MHz × μs = 2π rad)
"""
from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
from scipy import sparse, linalg
from scipy.integrate import solve_ivp

from .hamiltonian import build_liouvillian, build_recombination

logger = logging.getLogger(__name__)

# benchmark log path
_BENCH_LOG = Path("benchmarks/results/tdvp_timing.jsonl")


def _log_benchmark(record: dict) -> None:
    """append a JSON record to the timing log (creates file/dir if needed)."""
    _BENCH_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_BENCH_LOG, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─────────────────────────────────────────────────────────────
# Exact solver (small systems, dense)
# ─────────────────────────────────────────────────────────────

class ExactSolver:
    """
    exact time evolution using scipy.linalg.expm or solve_ivp.
    suitable for systems with ≤ 20 spin-½ particles (dim ≤ 10^6).

    usage:
        solver = ExactSolver(config)
        t, P_S, tr = solver.run(t_max_us=10.0, n_steps=500)
    """

    def __init__(self, config):
        """
        config : SystemConfig (from radical_pair.py)
        """
        self.config = config
        self._H = None
        self._K = None
        self._L = None
        self._rho0 = None

    def _build(self) -> None:
        if self._L is not None:
            return
        cfg = self.config
        logger.info(f"building Hamiltonian for {cfg.name} ({cfg.n_sites} sites)…")
        t0 = time.perf_counter()
        H = cfg.build_hamiltonian()
        K = cfg.build_K()
        self._L = build_liouvillian(H, K)
        self._rho0 = cfg.initial_rho()
        build_time = time.perf_counter() - t0
        logger.info(f"build: {build_time*1e3:.1f} ms, dim={H.shape[0]}")
        _log_benchmark({
            "event": "build",
            "system": cfg.name,
            "n_sites": cfg.n_sites,
            "dim": H.shape[0],
            "build_time_ms": round(build_time * 1e3, 2),
        })

    def _singlet_yield_instantaneous(self, rho_vec: np.ndarray) -> float:
        """P_S(t) = Tr(Q_S ρ(t))."""
        cfg = self.config
        dim = 2 ** cfg.n_sites
        rho = rho_vec.reshape(dim, dim)
        e1, e2, _, _ = cfg.site_layout()
        Q_S = build_recombination(e1, e2, cfg.n_sites, k_S=1.0, k_T=0.0)
        return np.real(np.trace(Q_S.toarray() @ rho))

    def run(
        self,
        t_max_us: float = 10.0,
        n_steps: int = 500,
        method: str = "RK45",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        integrate dρ/dt = L ρ from t=0 to t=t_max_us.

        returns:
            t_us   : (n_steps,) time array in μs
            P_S    : (n_steps,) singlet probability
            trace  : (n_steps,) Tr(ρ) — should decay due to recombination
        """
        self._build()
        cfg = self.config
        dim = 2 ** cfg.n_sites

        # use sparse matvec — never materialize dense Liouvillian
        # for n_sites ≤ 12 (dim ≤ 4096, Liouville dim ≤ 16M) this is fast
        L_sparse = self._L   # already sparse csr
        rho0_vec = self._rho0.reshape(-1)

        TWO_PI = 2.0 * np.pi

        def drho_dt(t, y):
            return TWO_PI * L_sparse.dot(y)

        logger.info(f"integrating {cfg.name} to {t_max_us} μs ({n_steps} steps)…")
        t0 = time.perf_counter()

        t_eval = np.linspace(0, t_max_us, n_steps)
        sol = solve_ivp(
            drho_dt,
            [0.0, t_max_us],
            rho0_vec,
            t_eval=t_eval,
            method=method,
            rtol=1e-8,
            atol=1e-10,
        )

        integrate_time = time.perf_counter() - t0
        logger.info(f"integration: {integrate_time:.2f} s")

        # singlet projector Q_S = 1/4 I - S1·S2, independent of k_S and k_T
        e1, e2, _, _ = cfg.site_layout()
        Q_S_arr = build_recombination(e1, e2, cfg.n_sites, k_S=1.0, k_T=0.0).toarray()
        P_S   = np.zeros(n_steps)
        trace = np.zeros(n_steps)
        for i, y in enumerate(sol.y.T):
            rho = y.reshape(dim, dim)
            P_S[i]   = np.real(np.trace(Q_S_arr @ rho))
            trace[i] = np.real(np.trace(rho))

        _log_benchmark({
            "event": "integrate",
            "system": cfg.name,
            "n_sites": cfg.n_sites,
            "dim": dim,
            "t_max_us": t_max_us,
            "n_steps": n_steps,
            "method": method,
            "integrate_time_s": round(integrate_time, 3),
            "final_trace": round(float(trace[-1]), 6),
        })

        return sol.t, P_S, trace


# ─────────────────────────────────────────────────────────────
# MPS solver (large systems, quimb-based)
# ─────────────────────────────────────────────────────────────

class MpsSolver:
    """
    TDVP solver using quimb tensor networks.
    handles density matrix as MPS (vectorized: ρ → |ρ⟩⟩).

    phase A: quimb + numpy (CPU, χ up to ~500)
    phase B: cuTensorNet backend (GPU, χ up to 2500+)

    usage:
        solver = MpsSolver(config, chi=64)
        t, P_S, tr = solver.run(t_max_us=5.0, n_steps=200)
    """

    def __init__(self, config, chi: int = 64, backend: str = "numpy"):
        """
        chi     : bond dimension (truncation parameter)
        backend : 'numpy'  — CPU Krylov / expm_multiply (Phase A)
                  'cupy'   — GPU sparse Krylov, N ≤ 14  (Phase B)
                  'cutn'   — GPU MPO-MPS TDVP, N ≤ 62  (Phase B, large χ)
        """
        self.config  = config
        self.chi     = chi
        self.backend = backend
        self._mpo    = None
        self._psi0   = None
        self._L_dense = None
        self._rho0   = None
        self._dim    = None
        self._gpu_solver = None   # CupyKrylovSolver or CuTDVPSolver

    def _build(self) -> None:
        """Build MPO Liouvillian and MPS initial state."""
        # ── Phase B: GPU backends ────────────────────────────────────────────
        if self.backend in ("cupy", "cutn"):
            from .cuda.engine import HAS_CUPY, HAS_CUTN, CupyKrylovSolver, CuTDVPSolver
            if self.backend == "cupy" and not HAS_CUPY:
                raise RuntimeError(
                    "backend='cupy' requested but CuPy is not available"
                    " — install cupy-cuda12x"
                )
            if self.backend == "cutn" and not HAS_CUTN:
                raise RuntimeError(
                    "backend='cutn' requested but cuQuantum is not available"
                    " — install cuquantum"
                )
            if self.backend == "cupy":
                self._gpu_solver = CupyKrylovSolver(self.config)
            else:
                self._gpu_solver = CuTDVPSolver(self.config, chi=self.chi)
            return

        # ── Phase A: CPU fallback ─────────────────────────────────────────────
        cfg = self.config
        logger.info(f"building MPO for {cfg.name} (χ={self.chi}, backend={self.backend})…")
        t0 = time.perf_counter()

        H = cfg.build_hamiltonian()
        K = cfg.build_K()
        L_super = build_liouvillian(H, K)

        self._L_dense = L_super.toarray()
        self._rho0    = cfg.initial_rho()
        self._dim     = H.shape[0]

        build_time = time.perf_counter() - t0
        logger.info(f"MPO build: {build_time*1e3:.1f} ms")
        _log_benchmark({
            "event": "mpo_build",
            "system": cfg.name,
            "n_sites": cfg.n_sites,
            "chi": self.chi,
            "backend": self.backend,
            "dim": self._dim,
            "build_time_ms": round(build_time * 1e3, 2),
        })

    def run(
        self,
        t_max_us: float = 5.0,
        n_steps:  int   = 200,
        dt_us:    Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        TDVP integration.

        Phase A (backend='numpy'): scipy expm_multiply (Krylov on CPU).
        Phase B (backend='cupy'):  GPU sparse Krylov via CupyKrylovSolver.
        Phase B (backend='cutn'):  GPU MPO-MPS TDVP via CuTDVPSolver.
        """
        if self._gpu_solver is None and self._L_dense is None:
            self._build()

        # ── Phase B: delegate to GPU solver ──────────────────────────────────
        if self._gpu_solver is not None:
            return self._gpu_solver.run(t_max_us=t_max_us, n_steps=n_steps)

        # ── Phase A: CPU expm_multiply ────────────────────────────────────────
        cfg    = self.config
        TWO_PI = 2.0 * np.pi

        if dt_us is None:
            dt_us = t_max_us / n_steps

        logger.info(f"MPS TDVP {cfg.name}: χ={self.chi}, {n_steps} steps (CPU)…")
        t0 = time.perf_counter()

        from scipy.sparse.linalg import expm_multiply

        t_eval   = np.linspace(0, t_max_us, n_steps)
        rho0_vec = self._rho0.reshape(-1)
        L_scaled = TWO_PI * sparse.csr_matrix(self._L_dense)

        rho_t = expm_multiply(
            L_scaled * t_max_us, rho0_vec,
            start=0.0, stop=1.0, num=n_steps, endpoint=True,
        )

        integrate_time = time.perf_counter() - t0
        logger.info(f"TDVP integration: {integrate_time:.2f} s")

        e1, e2, _, _ = cfg.site_layout()
        Q_S_arr = build_recombination(e1, e2, cfg.n_sites, k_S=1.0, k_T=0.0).toarray()
        dim     = self._dim
        P_S     = np.zeros(n_steps)
        trace   = np.zeros(n_steps)
        for i, y in enumerate(rho_t):
            rho      = y.reshape(dim, dim)
            P_S[i]   = np.real(np.trace(Q_S_arr @ rho))
            trace[i] = np.real(np.trace(rho))

        _log_benchmark({
            "event": "mps_integrate",
            "system": cfg.name,
            "chi": self.chi,
            "backend": self.backend,
            "n_steps": n_steps,
            "t_max_us": t_max_us,
            "integrate_time_s": round(integrate_time, 3),
        })

        return t_eval, P_S, trace
