"""
benchmarks/bench_chi.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
χ-convergence benchmark for the paper's key figure.

Sweeps χ = 16, 32, 64, 128, 256, 512 on ErCry4a (10 nuclei, 12 sites).
For each χ computes:
  - Φ_S(χ)          singlet yield
  - |Φ_S(χ) − Φ_S(ref)|  absolute error vs reference solver
  - wall time        seconds

Reference solver:
  - GPU available  → CupyKrylovSolver (dense GPU expm, RMSE < 1e-5 vs exact)
  - CPU / --numpy  → ExactSolver from cryptotn.tdvp (scipy dense, exact)

χ sweep solver:
  - GPU available  → CuTDVPSolver.run_2site()  (LR-only sweep; first-order in dt)
  - CPU / --numpy  → CuTDVPSolver(_numpy_mode=True).run_2site()

Produces:
  benchmarks/results/chi_convergence.json   — machine-readable
  benchmarks/results/chi_convergence.jsonl  — append log

This is the χ-convergence curve referenced in §4.2 of the paper.
Reference: Hino et al., arXiv:2509.22104 (2025) Table 2.

Implementation notes
────────────────────
Phase B fix: MpsSolver (quimb, Phase A) and its GPU replacement use different
sweep strategies.  CuTDVPSolver uses a pure LR-only 2-site sweep because the
FSM MPO accumulates Liouvillian contributions left-to-right; adding an RL pass
causes N-fold trace double-counting.  See engine.py §7 for the full analysis.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptotn.radical_pair import ercry4a_config
from cryptotn.observables   import singlet_yield

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── default sweep parameters ──────────────────────────────────────────────────
DEFAULT_CHI_LIST  = [16, 32, 64, 128, 256, 512]
DEFAULT_N_NUC     = 10     # nuclei per electron (12 sites total for ErCry4a)
DEFAULT_T_MAX_US  = 10.0
DEFAULT_N_STEPS   = 300


# ─────────────────────────────────────────────────────────────────────────────
# Reference solvers
# ─────────────────────────────────────────────────────────────────────────────

def run_exact_cpu(cfg, t_max_us: float, n_steps: int) -> tuple:
    """
    CPU exact reference via ExactSolver (scipy dense expm).
    Only feasible for n_sites ≤ ~14.
    Returns (phi_s, t, P_S, trace).
    """
    from cryptotn.tdvp import ExactSolver
    solver = ExactSolver(cfg)
    t, P_S, trace = solver.run(t_max_us=t_max_us, n_steps=n_steps)
    phi_s = singlet_yield(t, P_S, trace, cfg.k_S_us)
    return float(phi_s), t, P_S, trace


def run_exact_gpu(cfg, t_max_us: float, n_steps: int) -> tuple:
    """
    GPU dense reference via CupyKrylovSolver (RMSE < 1e-5 vs exact).
    Returns (phi_s, t, P_S, trace).
    """
    from cryptotn.cuda.engine import CupyKrylovSolver
    solver = CupyKrylovSolver(cfg, krylov_dim=50)
    t, P_S, trace = solver.run(t_max_us=t_max_us, n_steps=n_steps)
    phi_s = singlet_yield(t, P_S, trace, cfg.k_S_us)
    return float(phi_s), t, P_S, trace


# ─────────────────────────────────────────────────────────────────────────────
# χ sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_tdvp_chi(cfg, chi: int, numpy_mode: bool,
                 t_max_us: float, n_steps: int) -> tuple:
    """
    Run CuTDVPSolver.run_2site() at a given χ.
    Returns (phi_s, wall_time_s, t, P_S, trace).
    """
    from cryptotn.cuda.engine import CuTDVPSolver
    solver = CuTDVPSolver(cfg, chi=chi, _numpy_mode=numpy_mode)
    t0 = time.perf_counter()
    t, P_S, trace = solver.run_2site(t_max_us=t_max_us, n_steps=n_steps)
    wall = time.perf_counter() - t0
    phi_s = singlet_yield(t, P_S, trace, cfg.k_S_us)
    return float(phi_s), wall, t, P_S, trace


# ─────────────────────────────────────────────────────────────────────────────
# Main convergence sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_convergence(
    n_nuc:      int   = DEFAULT_N_NUC,
    chi_list:   list  = None,
    numpy_mode: bool  = False,
    t_max_us:   float = DEFAULT_T_MAX_US,
    n_steps:    int   = DEFAULT_N_STEPS,
    last_as_ref: bool = False,
) -> list:
    """
    Main convergence sweep.

    Reference strategy (in priority order):
      1. --last-as-ref flag         → use χ=max(chi_list) TDVP run as reference
      2. n_sites > 10 on GPU        → automatic fallback to last-as-ref
         (CupyKrylovSolver needs Liouville dim = 4^n, OOM for n > 10 on 16 GB)
      3. numpy_mode                 → ExactSolver CPU (scipy sparse, up to n≈12)
      4. GPU, n_sites ≤ 10          → CupyKrylovSolver GPU

    last-as-ref is the standard approach in DMRG/TDVP papers: compare
    smaller-χ results against the largest-χ run to show χ-convergence.
    """
    if chi_list is None:
        chi_list = DEFAULT_CHI_LIST

    cfg = ercry4a_config(n_nuc=n_nuc)
    logger.info(f"ErCry4a: {cfg.n_sites} sites, {n_nuc} nuc/rad, "
                f"B={cfg.B_mT} mT")

    # ── decide reference strategy ─────────────────────────────────────────────
    _gpu_krylov_max_sites = 10   # CupyKrylovSolver OOMs for n_sites > 10 on 16 GB

    if last_as_ref or (not numpy_mode and cfg.n_sites > _gpu_krylov_max_sites):
        if cfg.n_sites > _gpu_krylov_max_sites and not last_as_ref:
            logger.warning(
                f"N={cfg.n_sites} > {_gpu_krylov_max_sites}: "
                "CupyKrylovSolver would OOM on 16 GB GPU. "
                f"Using χ={max(chi_list)} (last in sweep) as reference instead."
            )
        else:
            logger.info(f"Using χ={max(chi_list)} (last in sweep) as reference.")
        phi_ref    = None
        ref_solver = f"TDVP_chi{max(chi_list)}"
    elif numpy_mode:
        logger.info("Running ExactSolver (CPU) as reference…")
        phi_ref, *_ = run_exact_cpu(cfg, t_max_us, n_steps)
        ref_solver  = "ExactSolver_cpu"
        logger.info(f"Reference Φ_S = {phi_ref:.6f}")
    else:
        logger.info("Running CupyKrylovSolver (GPU) as reference…")
        phi_ref, *_ = run_exact_gpu(cfg, t_max_us, n_steps)
        ref_solver  = "CupyKrylovSolver_gpu"
        logger.info(f"Reference Φ_S = {phi_ref:.6f}")

    # ── χ sweep ───────────────────────────────────────────────────────────────
    results      = []
    last_phi     = None   # tracks φ_S of the last completed χ run
    phi_ref_used = phi_ref

    for chi in chi_list:
        logger.info(f"χ = {chi:4d}  numpy_mode={numpy_mode}…")
        try:
            phi_s, wall, t, P_S, trace = run_tdvp_chi(
                cfg, chi, numpy_mode, t_max_us, n_steps
            )
            last_phi = phi_s

            # For last-as-ref: error is computed in post-processing below
            ref = phi_ref_used
            err = abs(phi_s - ref) if ref is not None else None

            rec = {
                "chi":              chi,
                "phi_s":            round(phi_s, 6),
                "phi_s_ref":        round(ref, 6) if ref is not None else None,
                "abs_error":        round(err, 6) if err is not None else None,
                "wall_time_s":      round(wall, 3),
                "numpy_mode":       numpy_mode,
                "ref_solver":       ref_solver,
                "n_nuc":            n_nuc,
                "n_sites":          cfg.n_sites,
                "t_max_us":         t_max_us,
                "n_steps":          n_steps,
            }
            results.append(rec)

            err_str = f"  |ΔΦ_S|={err:.2e}" if err is not None else ""
            logger.info(f"  χ={chi:4d}: Φ_S={phi_s:.5f}{err_str}  wall={wall:.2f}s")

        except Exception as e:
            logger.error(f"  χ={chi}: FAILED — {e}")
            results.append({"chi": chi, "error": str(e),
                            "n_sites": cfg.n_sites, "n_nuc": n_nuc})

    # ── backfill last-as-ref errors ───────────────────────────────────────────
    # When ref_solver starts with "TDVP_chi", the reference is the last
    # successful chi run.  Go back and fill in abs_error for all earlier runs.
    if phi_ref is None and last_phi is not None:
        logger.info(f"Reference (χ={max(chi_list)}): Φ_S = {last_phi:.6f}")
        for rec in results:
            if "error" in rec:
                continue
            rec["phi_s_ref"] = round(last_phi, 6)
            rec["abs_error"] = round(abs(rec["phi_s"] - last_phi), 6)
        # The last entry has abs_error=0 by definition
        for rec in reversed(results):
            if "error" not in rec:
                rec["abs_error"] = 0.0
                break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_results(results: list) -> None:
    out_file = RESULTS_DIR / "chi_convergence.json"
    log_file = RESULTS_DIR / "chi_convergence.jsonl"

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    with open(log_file, "a") as f:
        for rec in results:
            f.write(json.dumps(rec) + "\n")

    logger.info(f"Results → {out_file}")


def print_table(results: list) -> None:
    if not results:
        print("(no results)")
        return
    first = next((r for r in results if "error" not in r), {})
    n_sites  = first.get("n_sites", "?")
    ref_name = first.get("ref_solver", "?")
    print("\n" + "=" * 66)
    print(f"  chi-convergence  (ErCry4a, {n_sites} sites, ref={ref_name})")
    print("=" * 66)
    print(f"{'chi':>6}  {'Phi_S':>10}  {'Phi_S_ref':>10}  {'|dPhi_S|':>10}  {'wall (s)':>10}")
    print("-" * 54)
    for r in results:
        if "error" in r:
            print(f"{r['chi']:6d}  FAILED: {r['error']}")
            continue
        ref_str = f"{r['phi_s_ref']:.6f}" if r.get("phi_s_ref") is not None else "  N/A   "
        err_str = f"{r['abs_error']:.2e}" if r.get("abs_error") is not None else "  N/A   "
        print(f"{r['chi']:6d}  {r['phi_s']:10.6f}  {ref_str:>10}  {err_str:>10}  "
              f"{r['wall_time_s']:10.2f}")
    print("=" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="χ-convergence benchmark")
    parser.add_argument("--n-nuc",   type=int,   default=DEFAULT_N_NUC,
                        help="nuclei per electron (default: 10 → 12 sites)")
    parser.add_argument("--t-max",   type=float, default=DEFAULT_T_MAX_US,
                        help="integration time in μs (default: 10.0)")
    parser.add_argument("--n-steps", type=int,   default=DEFAULT_N_STEPS,
                        help="number of time steps (default: 300)")
    parser.add_argument("--chi",     type=int,   nargs="+",
                        default=DEFAULT_CHI_LIST,
                        help="χ values to sweep (default: 16 32 64 128 256 512)")
    parser.add_argument("--backend", choices=["numpy", "cupy", "cutn"],
                        default=None,
                        help="compute backend: numpy=CPU, cupy=GPU sparse Krylov, "
                             "cutn=GPU MPO-MPS. "
                             "Default: use GPU if cupy is available, else exit with "
                             "an error — no silent CPU fallback.")
    parser.add_argument("--numpy",        action="store_true",
                        help="shorthand for --backend numpy (CPU, backward compat)")
    parser.add_argument("--last-as-ref",  action="store_true",
                        help="use χ=max(chi_list) TDVP run as reference "
                             "(automatic for n_sites > 10 on GPU)")
    parser.add_argument("--fast",         action="store_true",
                        help="short smoke-test run (χ≤64, 50 steps, 2 μs)")
    args = parser.parse_args()

    # ── resolve backend ───────────────────────────────────────────────────────
    if args.numpy and args.backend is not None and args.backend != "numpy":
        parser.error("--numpy and --backend are mutually exclusive")

    if args.numpy:
        backend = "numpy"
    elif args.backend is not None:
        backend = args.backend
    else:
        # auto-detect: use GPU if available, never silently fall to CPU
        from cryptotn.cuda.engine import HAS_CUPY
        if HAS_CUPY:
            backend = "cupy"
            print("[INFO] --backend not specified, cupy detected → using GPU (cupy)")
        else:
            print("[ERROR] --backend not specified and cupy is not installed.")
            print("        Run with --backend numpy for CPU-only mode, or")
            print("        install cupy:  pip install cupy-cuda12x")
            sys.exit(1)

    numpy_mode = (backend == "numpy")

    if args.fast:
        chi_list = [16, 32, 64]
        n_steps  = 50
        t_max_us = 2.0
    else:
        chi_list = args.chi
        n_steps  = args.n_steps
        t_max_us = args.t_max

    mode_str = f"numpy/CPU" if numpy_mode else f"GPU ({backend})"
    print("=" * 66)
    print(f"chi-convergence benchmark  (backend={backend}, mode={mode_str})")
    print(f"ErCry4a  n_nuc={args.n_nuc}  t_max={t_max_us}us  n_steps={n_steps}")
    print("=" * 66)

    results = run_convergence(
        n_nuc        = args.n_nuc,
        chi_list     = chi_list,
        numpy_mode   = numpy_mode,
        t_max_us     = t_max_us,
        n_steps      = n_steps,
        last_as_ref  = args.last_as_ref,
    )

    print_table(results)
    save_results(results)
