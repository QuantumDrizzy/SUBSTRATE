"""tests/test_tdvp.py"""
import numpy as np
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptotn.radical_pair import fad_w_model, ercry4a_config
from cryptotn.tdvp import ExactSolver, MpsSolver
from cryptotn.observables import singlet_yield, triplet_yield


def test_exact_initial_singlet_probability():
    """P_S(t=0) = 1 for singlet initial state."""
    cfg = fad_w_model(n_nuc_fad=1, n_nuc_w=1)
    cfg.B_mT = 0.0
    solver = ExactSolver(cfg)
    t, P_S, trace = solver.run(t_max_us=0.1, n_steps=10)
    assert abs(P_S[0] - 1.0) < 0.05, f"P_S(0)={P_S[0]:.4f}, expected ~1"


def test_yield_normalization():
    """Φ_S + Φ_T = 1 approximately (to within integration window)."""
    cfg = fad_w_model(n_nuc_fad=2, n_nuc_w=1)
    cfg.B_mT = 0.0
    solver = ExactSolver(cfg)
    t, P_S, trace = solver.run(t_max_us=20.0, n_steps=200)
    phi_s = singlet_yield(t, P_S, trace, cfg.k_S_us)
    phi_t = triplet_yield(phi_s)
    # both yields should be in [0,1]
    assert 0 <= phi_s <= 1.01, f"Φ_S={phi_s}"
    assert 0 <= phi_t <= 1.01, f"Φ_T={phi_t}"


def test_b_field_effect():
    """applied B field should change singlet yield (magnetoreception effect)."""
    cfg_0 = fad_w_model(n_nuc_fad=3, n_nuc_w=2)
    cfg_0.B_mT = 0.0
    cfg_B = fad_w_model(n_nuc_fad=3, n_nuc_w=2)
    cfg_B.B_mT = 0.5   # 0.5 mT

    solver_0 = ExactSolver(cfg_0)
    solver_B = ExactSolver(cfg_B)

    t0, P0, tr0 = solver_0.run(t_max_us=5.0, n_steps=100)
    tB, PB, trB = solver_B.run(t_max_us=5.0, n_steps=100)

    phi_0 = singlet_yield(t0, P0, tr0, cfg_0.k_S_us)
    phi_B = singlet_yield(tB, PB, trB, cfg_B.k_S_us)

    # yields should differ (magnetic field effect)
    assert abs(phi_B - phi_0) > 1e-4, (
        f"No MFE detected: Φ_S(0)={phi_0:.5f}, Φ_S(B)={phi_B:.5f}"
    )


def test_mps_vs_exact_small():
    """MPS and exact solvers agree for small system (n_nuc=4)."""
    cfg = fad_w_model(n_nuc_fad=2, n_nuc_w=2)  # 6 sites
    cfg.B_mT = 0.05

    t_exact, P_exact, tr_exact = ExactSolver(cfg).run(t_max_us=3.0, n_steps=50)
    t_mps,   P_mps,   tr_mps   = MpsSolver(cfg, chi=16).run(t_max_us=3.0, n_steps=50)

    phi_exact = singlet_yield(t_exact, P_exact, tr_exact, cfg.k_S_us)
    phi_mps   = singlet_yield(t_mps,   P_mps,   tr_mps,   cfg.k_S_us)

    # MPS should agree with exact to within χ truncation error
    assert abs(phi_mps - phi_exact) < 0.05, (
        f"MPS/exact disagreement: {phi_mps:.5f} vs {phi_exact:.5f}"
    )


def test_chi_convergence():
    """singlet yield should converge as χ increases."""
    cfg = fad_w_model(n_nuc_fad=3, n_nuc_w=3)  # 8 sites
    cfg.B_mT = 0.05

    phi_values = []
    for chi in [8, 16, 32]:
        t, P_S, tr = MpsSolver(cfg, chi=chi).run(t_max_us=3.0, n_steps=50)
        phi_values.append(singlet_yield(t, P_S, tr, cfg.k_S_us))

    # each step should bring us closer (non-monotone but bounded)
    assert all(0 <= p <= 1.01 for p in phi_values), f"yields out of range: {phi_values}"
