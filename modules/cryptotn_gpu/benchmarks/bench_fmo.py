"""
benchmarks/bench_fmo.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
benchmark 1: FMO 7-site energy transfer at 77K and 300K.

comparison target: TENSO (Dunnett et al., J. Chem. Phys. 163, 104109, 2025)
metric: P_1(t) — population of BChl 1 over time.
initial state: full excitation on BChl 1 (site 0).

reference curve digitized from TENSO Fig. 3 for comparison.
"""
from __future__ import annotations

import time
import json
import logging
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

# add parent to path if running standalone
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptotn.hamiltonian import build_fmo_lindblad

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Reference data (TENSO, digitized from Fig. 3)
# These are approximate values for validation.
# ─────────────────────────────────────────────────────────────
TENSO_77K_T_FS  = np.array([0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
TENSO_77K_P1    = np.array([1.000, 0.712, 0.521, 0.430, 0.385, 0.370, 0.362, 0.355, 0.348, 0.342, 0.337])
TENSO_300K_P1   = np.array([1.000, 0.682, 0.465, 0.334, 0.258, 0.220, 0.198, 0.186, 0.179, 0.175, 0.172])


def lindblad_rhs(t, rho_vec, L_super):
    """d|ρ⟩⟩/dt = L_super |ρ⟩⟩."""
    return L_super @ rho_vec


def build_fmo_liouvillian(T_K: float) -> tuple:
    """
    build FMO Liouvillian in cm⁻¹ units.
    L = -i[H,·] + Σ_k γ_k (L_k ρ L_k† - ½{L_k†L_k, ρ})
    """
    n = 7
    H, lindblad_ops = build_fmo_lindblad(T_K=T_K)

    dim = n
    I = np.eye(dim, dtype=complex)
    # coherent part
    L = -1j * (np.kron(H, I) - np.kron(I, H.T))

    # dissipator
    for Lk in lindblad_ops:
        L += (
            np.kron(Lk, Lk.conj())
            - 0.5 * np.kron(Lk.conj().T @ Lk, I)
            - 0.5 * np.kron(I, (Lk.conj().T @ Lk).T)
        )
    return L, H


def run_fmo(
    T_K: float = 77.0,
    t_max_fs: float = 1000.0,
    n_steps: int = 200,
) -> dict:
    """
    run FMO dynamics at temperature T_K.
    returns dict with time array and site populations.

    time unit: femtoseconds (fs). 1 fs = 3e-5 cm⁻¹·s = 0.03 ps.
    energy: 1 cm⁻¹ ≡ 0.18836 rad/ps = 188.36 rad/ns.
    conversion: t_ps = t_fs / 1000; ω_rad_ps = 2π × c_cm_ps × E_cm1
    c in cm/ps = 0.02998 cm/ps → 2πc = 0.18836 rad·cm·ps
    """
    HBAR_CMINV_FS = 5308.8   # ħ in cm⁻¹·fs

    logger.info(f"FMO benchmark at T={T_K}K, t_max={t_max_fs} fs")
    t0_wall = time.perf_counter()

    L_super, H = build_fmo_liouvillian(T_K)

    # initial state: full excitation on site 1 (BChl 1)
    n = 7
    rho0 = np.zeros((n, n), dtype=complex)
    rho0[0, 0] = 1.0
    rho0_vec = rho0.reshape(-1)

    # units: Liouvillian in cm⁻¹, time in fs → multiply by 1/ħ_cminv_fs
    L_scaled = L_super / HBAR_CMINV_FS

    t_eval = np.linspace(0, t_max_fs, n_steps)
    sol = solve_ivp(
        lambda t, y: L_scaled @ y,
        [0.0, t_max_fs],
        rho0_vec,
        t_eval=t_eval,
        method="RK45",
        rtol=1e-8,
        atol=1e-11,
    )

    wall_time = time.perf_counter() - t0_wall

    # extract site populations P_k(t) = ρ_kk(t)
    populations = np.zeros((n_steps, n))
    for i, y in enumerate(sol.y.T):
        rho = y.reshape(n, n)
        populations[i] = np.diag(rho).real

    # deviation from TENSO reference (interpolated)
    if T_K == 77.0:
        ref_t, ref_P1 = TENSO_77K_T_FS, TENSO_77K_P1
    else:
        ref_t, ref_P1 = TENSO_77K_T_FS, TENSO_300K_P1  # 300K

    P1_interp = np.interp(ref_t, sol.t, populations[:, 0])
    rmse = float(np.sqrt(np.mean((P1_interp - ref_P1) ** 2)))

    result = {
        "system": "FMO_7site",
        "T_K": T_K,
        "t_max_fs": t_max_fs,
        "n_steps": n_steps,
        "wall_time_s": round(wall_time, 3),
        "rmse_vs_tenso": round(rmse, 5),
        "final_trace": round(float(np.trace(sol.y[:, -1].reshape(n, n)).real), 6),
    }

    # save
    out_file = RESULTS_DIR / f"fmo_{int(T_K)}K.npz"
    np.savez(out_file, t=sol.t, populations=populations)
    logger.info(f"saved populations to {out_file}")

    log_file = RESULTS_DIR / "fmo_benchmark.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    logger.info(
        f"FMO {T_K}K: wall={wall_time:.2f}s, RMSE_vs_TENSO={rmse:.5f}, "
        f"trace_final={result['final_trace']:.6f}"
    )
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("benchmark 1: FMO 7-site energy transfer")
    print("=" * 60)
    for T in [77.0, 300.0]:
        r = run_fmo(T_K=T, t_max_fs=1000.0, n_steps=500)
        print(f"T={T}K | RMSE vs TENSO: {r['rmse_vs_tenso']:.5f} | "
              f"time: {r['wall_time_s']:.2f}s")
    print("\nresults saved to benchmarks/results/")
