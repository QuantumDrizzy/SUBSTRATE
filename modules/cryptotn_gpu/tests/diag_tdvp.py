"""
tests/diag_tdvp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Diagnostic script for CuTDVPSolver — isolates exactly where
the multi-step trace instability comes from.

Three focused tests:

  TEST 1 — Trace conservation (k_S=k_T=0, pure Hamiltonian)
    Trace must be exactly 1 at every step.  Any drift reveals
    accumulative error in the LR sweep itself.

  TEST 2 — Single-step accuracy vs ExactSolver
    n_nuc=1 (3 sites), dt=0.001 μs, 1 step.
    Compare tr and P_S against ExactSolver.

  TEST 3 — Multi-step trajectory vs ExactSolver
    n_nuc=1, 50 steps, dt=0.001 μs.
    Side-by-side tr(t) and P_S(t).  Shows when/how TDVP diverges.

Run:
    cd /mnt/c/Users/drizzy/Desktop/cryptotn-gpu
    python tests/diag_tdvp.py
"""
from __future__ import annotations

import sys, copy
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptotn.radical_pair import ercry4a_config, SystemConfig, RadicalConfig, NuclearSpin
from cryptotn.tdvp       import ExactSolver
from cryptotn.cuda.engine import CuTDVPSolver

SEP  = "=" * 60
SEP2 = "-" * 60


def _clone_with_rates(cfg, k_S: float, k_T: float) -> SystemConfig:
    """Return a copy of cfg with modified recombination rates."""
    return SystemConfig(
        name        = cfg.name + f"_kS{k_S}",
        radical_1   = cfg.radical_1,
        radical_2   = cfg.radical_2,
        k_S_us      = k_S,
        k_T_us      = k_T,
        J_MHz       = cfg.J_MHz,
        B_mT        = cfg.B_mT,
        B_axis      = cfg.B_axis,
        description = cfg.description,
    )


# ─────────────────────────────────────────────────────────────
# TEST 1: trace conservation under pure Hamiltonian (k_S=k_T=0)
# ─────────────────────────────────────────────────────────────

def test1_trace_conservation():
    print(SEP)
    print("TEST 1 — Trace conservation  (k_S=k_T=0, pure H)")
    print(SEP)

    cfg0 = ercry4a_config(n_nuc=1)                # 3 sites
    cfg  = _clone_with_rates(cfg0, k_S=0.0, k_T=0.0)
    print(f"  System: {cfg.n_sites} sites, k_S=0, k_T=0")

    solver = CuTDVPSolver(cfg, chi=16, _numpy_mode=True)
    N_STEPS = 20
    DT      = 0.01       # μs

    t_eval = np.linspace(0.0, N_STEPS * DT, N_STEPS + 1)
    # run manually step by step so we can inspect each step
    solver._build()

    from cryptotn.cuda.engine import _init_boundary_envs
    traces = [solver._observables()[1]]

    for step in range(N_STEPS):
        right_envs = solver._build_right_envs()
        L0, _ = _init_boundary_envs(cfg.n_sites, xp=solver._xp)
        left_envs = [None] * (cfg.n_sites + 1)
        left_envs[0] = L0
        solver._sweep_lr_2site(left_envs, right_envs, DT)
        _, tr = solver._observables()
        traces.append(tr)

    traces = np.array(traces)
    max_drift = np.max(np.abs(traces - 1.0))
    print(f"  {'step':>5}  {'trace':>10}  {'|tr-1|':>10}")
    print(SEP2)
    for i, tr in enumerate(traces):
        flag = "  ← DRIFT" if abs(tr - 1.0) > 1e-4 else ""
        print(f"  {i:5d}  {tr:10.6f}  {abs(tr-1.0):10.2e}{flag}")
    print(SEP2)
    print(f"  Max |tr - 1| = {max_drift:.2e}")
    status = "PASS" if max_drift < 1e-3 else "FAIL"
    print(f"  Result: {status}")
    print()
    return max_drift


# ─────────────────────────────────────────────────────────────
# TEST 2: single-step accuracy vs ExactSolver
# ─────────────────────────────────────────────────────────────

def test2_single_step():
    print(SEP)
    print("TEST 2 — Single-step accuracy vs ExactSolver  (n_nuc=1, dt=0.001 μs)")
    print(SEP)

    cfg = ercry4a_config(n_nuc=1)
    DT  = 0.001
    print(f"  System: {cfg.n_sites} sites, k_S={cfg.k_S_us}, k_T={cfg.k_T_us}")

    # Exact
    exact = ExactSolver(cfg)
    t_ex, P_ex, tr_ex = exact.run(t_max_us=DT, n_steps=2)   # step 0 and step 1
    print(f"  ExactSolver:  tr[1]={tr_ex[1]:.8f}  P_S[1]={P_ex[1]:.8f}")

    # TDVP
    solver = CuTDVPSolver(cfg, chi=16, _numpy_mode=True)
    solver._build()
    from cryptotn.cuda.engine import _init_boundary_envs
    right_envs = solver._build_right_envs()
    L0, _ = _init_boundary_envs(cfg.n_sites, xp=solver._xp)
    left_envs = [None] * (cfg.n_sites + 1)
    left_envs[0] = L0
    solver._sweep_lr_2site(left_envs, right_envs, DT)
    P_tdvp, tr_tdvp = solver._observables()
    print(f"  CuTDVPSolver: tr[1]={tr_tdvp:.8f}  P_S[1]={P_tdvp:.8f}")

    err_tr = abs(tr_tdvp - tr_ex[1])
    err_ps = abs(P_tdvp - P_ex[1])
    print(f"  |Δtr|   = {err_tr:.2e}")
    print(f"  |ΔP_S|  = {err_ps:.2e}")
    status = "PASS" if err_tr < 1e-3 and err_ps < 1e-3 else "FAIL"
    print(f"  Result: {status}")
    print()
    return err_tr, err_ps


# ─────────────────────────────────────────────────────────────
# TEST 3: multi-step trajectory vs ExactSolver
# ─────────────────────────────────────────────────────────────

def test3_trajectory():
    print(SEP)
    print("TEST 3 — Multi-step trajectory vs ExactSolver  (n_nuc=1, 50 steps, dt=0.001 μs)")
    print(SEP)

    cfg     = ercry4a_config(n_nuc=1)
    N_STEPS = 50
    DT      = 0.001
    print(f"  System: {cfg.n_sites} sites, t_max={N_STEPS*DT:.3f} μs")

    # Exact reference
    exact = ExactSolver(cfg)
    t_ex, P_ex, tr_ex = exact.run(t_max_us=N_STEPS * DT, n_steps=N_STEPS + 1)

    # TDVP
    from cryptotn.cuda.engine import _init_boundary_envs
    solver = CuTDVPSolver(cfg, chi=16, _numpy_mode=True)
    solver._build()

    tr_tdvp = np.zeros(N_STEPS + 1)
    ps_tdvp = np.zeros(N_STEPS + 1)
    ps_tdvp[0], tr_tdvp[0] = solver._observables()

    for step in range(N_STEPS):
        right_envs = solver._build_right_envs()
        L0, _ = _init_boundary_envs(cfg.n_sites, xp=solver._xp)
        left_envs = [None] * (cfg.n_sites + 1)
        left_envs[0] = L0
        solver._sweep_lr_2site(left_envs, right_envs, DT)
        ps_tdvp[step + 1], tr_tdvp[step + 1] = solver._observables()

    print(f"  {'step':>5}  {'t(μs)':>7}  {'tr_exact':>10}  {'tr_tdvp':>10}  {'|Δtr|':>10}  {'P_S_exact':>10}  {'P_S_tdvp':>10}")
    print(SEP2)
    # Print every 5th step to keep output manageable
    for i in range(0, N_STEPS + 1, 5):
        dt_tr = abs(tr_tdvp[i] - tr_ex[i])
        flag  = " ← UNPHYS" if tr_tdvp[i] < -1e-4 else ""
        print(f"  {i:5d}  {t_ex[i]:7.3f}  {tr_ex[i]:10.6f}  {tr_tdvp[i]:10.6f}  {dt_tr:10.2e}  {P_ex[i]:10.6f}  {ps_tdvp[i]:10.6f}{flag}")

    max_err_tr = np.max(np.abs(tr_tdvp - tr_ex))
    print(SEP2)
    print(f"  Max |Δtr| over all steps = {max_err_tr:.2e}")
    neg_steps = np.sum(tr_tdvp < -1e-4)
    print(f"  Steps with tr < 0        = {neg_steps}")
    status = "PASS" if max_err_tr < 1e-2 and neg_steps == 0 else "FAIL"
    print(f"  Result: {status}")
    print()
    return max_err_tr, neg_steps


# ─────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(SEP)
    print("CuTDVPSolver diagnostic suite")
    print(SEP)
    print()

    drift      = test1_trace_conservation()
    err_tr, _  = test2_single_step()
    max_err, neg = test3_trajectory()

    print(SEP)
    print("SUMMARY")
    print(SEP)
    print(f"  Test 1  trace drift (k=0):      {drift:.2e}  {'OK' if drift < 1e-3 else 'FAIL'}")
    print(f"  Test 2  single-step |Δtr|:       {err_tr:.2e}  {'OK' if err_tr < 1e-3 else 'FAIL'}")
    print(f"  Test 3  max traj |Δtr|:          {max_err:.2e}  {'OK' if max_err < 1e-2 else 'FAIL'}")
    print(f"  Test 3  unphysical steps (tr<0): {neg}       {'OK' if neg == 0 else 'FAIL'}")
    print(SEP)
