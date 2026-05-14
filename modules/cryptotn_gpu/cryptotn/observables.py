"""
cryptotn/observables.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
post-processing: singlet/triplet yields, compass sensitivity.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def singlet_yield(
    t_us: np.ndarray,
    P_S: np.ndarray,
    trace: np.ndarray,
    k_S_us: float,
) -> float:
    """
    Φ_S = k_S ∫₀^∞ P_S(t) dt   (integrated singlet yield)

    uses trapezoidal integration over the simulated window.
    for accurate results, integrate until trace ~ 0 (full recombination).
    """
    _trapz = getattr(np, "trapezoid", None) or np.trapz  # numpy ≥2.0 renamed trapz
    return float(k_S_us * _trapz(P_S, t_us))


def triplet_yield(Phi_S: float) -> float:
    """Φ_T = 1 - Φ_S  (probability conservation)."""
    return 1.0 - Phi_S


def compass_sensitivity(
    Phi_S_0: float,
    Phi_S_B: float,
    B_mT: float,
) -> float:
    """
    ΔΦ_S = Φ_S(B) - Φ_S(0)  (magnetic field effect on singlet yield).
    higher |ΔΦ_S| → stronger magnetoreception signal.
    """
    return Phi_S_B - Phi_S_0


def anisotropy(
    phi_s_parallel: float,
    phi_s_perpendicular: float,
) -> float:
    """
    directional anisotropy = Φ_S(B||) - Φ_S(B⊥)
    relevant for compass inclination sensing.
    """
    return phi_s_parallel - phi_s_perpendicular


def fmo_population_transfer(
    t_us: np.ndarray,
    rho_t: np.ndarray,
    n_sites: int = 7,
    source_site: int = 1,    # BChl 1 (0-indexed)
    sink_site: int = 3,      # BChl 4 (reaction center side)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    extract site populations ρ_ii(t) for FMO transfer benchmark.

    rho_t : (n_steps, dim, dim) density matrices
    returns (P_source(t), P_sink(t))
    """
    P_source = rho_t[:, source_site, source_site].real
    P_sink   = rho_t[:, sink_site,   sink_site  ].real
    return P_source, P_sink


def chi_convergence_test(
    chi_values: list,
    phi_s_values: list,
) -> float:
    """
    estimate truncation error as |Φ_S(χ_max) - Φ_S(χ_max/2)|.
    used to verify χ=2500 target is converged vs χ=1500 (Hino 2025).
    """
    if len(chi_values) < 2:
        return float("inf")
    return abs(phi_s_values[-1] - phi_s_values[-2])
