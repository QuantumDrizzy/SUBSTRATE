"""
benchmarks/bench_ercry4a.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
benchmark 2: ErCry4a radical pair — 30 to 60 nuclear spins.

comparison target: Hino et al., arXiv:2509.22104 (2025)
  CPU TDVP at χ=1500: ~6h for 60 spins.
  our target: χ=2500 on RTX 5060 Ti in < 1h.

metric: Φ_S(B) singlet yield vs applied field B (0–1 mT).
compass sensitivity: ΔΦ_S = Φ_S(B) - Φ_S(0).
"""
from __future__ import annotations

import time
import json
import logging
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptotn.radical_pair import ercry4a_config
from cryptotn.tdvp import ExactSolver, MpsSolver
from cryptotn.observables import singlet_yield, compass_sensitivity

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# reference from Hino 2025 (digitized, approximate)
HINO_B_MT    = np.array([0.00, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00])
HINO_PHI_S   = np.array([0.500, 0.504, 0.510, 0.519, 0.528, 0.532, 0.534, 0.535])  # χ=1500


def run_ercry4a(
    n_nuc: int = 10,
    B_field_values: list = None,
    t_max_us: float = 10.0,
    n_steps: int = 300,
    chi: int = 64,
    use_mps: bool = False,
) -> dict:
    """
    compute singlet yield vs magnetic field for ErCry4a.

    n_nuc    : number of nuclear spins (10 for quick test, 60 for full benchmark)
    B_field  : list of B values in mT (default: 0, 0.05, 0.1, 0.5, 1.0)
    use_mps  : use MpsSolver (for n_nuc > 20); ExactSolver otherwise
    """
    if B_field_values is None:
        B_field_values = [0.0, 0.05, 0.10, 0.20, 0.50, 1.00]

    logger.info(f"ErCry4a benchmark: {n_nuc} nuclei, {len(B_field_values)} field points")
    t0_wall = time.perf_counter()

    phi_s_values = []
    for B_mT in B_field_values:
        cfg = ercry4a_config(n_nuc=n_nuc)
        cfg.B_mT = B_mT

        if use_mps or cfg.n_sites > 20:
            solver = MpsSolver(cfg, chi=chi)
        else:
            solver = ExactSolver(cfg)

        t, P_S, trace = solver.run(t_max_us=t_max_us, n_steps=n_steps)
        phi_s = singlet_yield(t, P_S, trace, cfg.k_S_us)
        phi_s_values.append(phi_s)
        logger.info(f"  B={B_mT:.2f} mT → Φ_S={phi_s:.5f}")

    wall_time = time.perf_counter() - t0_wall

    # compass sensitivity vs earth field (0.05 mT)
    idx_earth = B_field_values.index(0.05) if 0.05 in B_field_values else 1
    delta_phi = compass_sensitivity(phi_s_values[0], phi_s_values[idx_earth], B_mT=0.05)

    # compare against Hino 2025 if same n_nuc
    rmse_hino = None
    if n_nuc >= 30:
        phi_s_interp = np.interp(HINO_B_MT, B_field_values, phi_s_values)
        rmse_hino = float(np.sqrt(np.mean((phi_s_interp - HINO_PHI_S) ** 2)))

    result = {
        "system": f"ErCry4a_{n_nuc}nuc",
        "n_nuc": n_nuc,
        "n_sites": ercry4a_config(n_nuc).n_sites,
        "chi": chi if use_mps else "exact",
        "B_mT": B_field_values,
        "phi_s": [round(p, 6) for p in phi_s_values],
        "delta_phi_s_earth": round(delta_phi, 6),
        "rmse_vs_hino": round(rmse_hino, 5) if rmse_hino else None,
        "wall_time_s": round(wall_time, 2),
        "t_max_us": t_max_us,
        "n_steps": n_steps,
    }

    out_file = RESULTS_DIR / f"ercry4a_{n_nuc}nuc.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    log_file = RESULTS_DIR / "ercry4a_benchmark.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    logger.info(
        f"ErCry4a {n_nuc} nuc: ΔΦ_S={delta_phi:.5f}, "
        f"wall={wall_time:.1f}s"
        + (f", RMSE_vs_Hino={rmse_hino:.5f}" if rmse_hino else "")
    )
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-nuc", type=int, default=10,
                        help="number of nuclear spins (10 for fast test, 60 for full)")
    parser.add_argument("--chi", type=int, default=64)
    parser.add_argument("--t-max", type=float, default=10.0)
    args = parser.parse_args()

    print("=" * 60)
    print(f"benchmark 2: ErCry4a ({args.n_nuc} nuclear spins)")
    print("=" * 60)

    use_mps = args.n_nuc > 18
    r = run_ercry4a(
        n_nuc=args.n_nuc,
        t_max_us=args.t_max,
        n_steps=200,
        chi=args.chi,
        use_mps=use_mps,
    )
    print(f"ΔΦ_S (earth field, 50μT) = {r['delta_phi_s_earth']:.5f}")
    if r["rmse_vs_hino"]:
        print(f"RMSE vs Hino 2025: {r['rmse_vs_hino']:.5f}")
    print(f"wall time: {r['wall_time_s']:.1f}s")
    print(f"results → benchmarks/results/ercry4a_{args.n_nuc}nuc.json")
