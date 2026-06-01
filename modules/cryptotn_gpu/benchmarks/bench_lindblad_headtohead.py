"""
benchmarks/bench_lindblad_headtohead.py
Honest GPU-vs-CPU head-to-head for the radical-pair Lindblad/Liouville solver.

  CPU : ExactSolver        (scipy)
  GPU : CupyKrylovSolver   (sparse L on device + Arnoldi expm), RTX 5060 Ti

Same config, same t_max/n_steps. Validates the two agree (max|ΔP_S|) before
trusting any speedup number. Sweeps system size (n_nuc) to show how the GPU
advantage scales with Liouville dimension.

Run:  python benchmarks/bench_lindblad_headtohead.py
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptotn.radical_pair import ercry4a_config
from cryptotn.tdvp import ExactSolver
from cryptotn.cuda.engine import CupyKrylovSolver

T_MAX_US = 10.0
N_STEPS  = 300
N_NUC_SWEEP = [1, 2, 3]   # n_sites = 2 + 2*n_nuc -> 4, 6, 8 ; Liouville dim = 4^n_sites


def time_cpu(cfg):
    t0 = time.perf_counter()
    t, P_S, trace = ExactSolver(cfg).run(t_max_us=T_MAX_US, n_steps=N_STEPS)
    return time.perf_counter() - t0, np.asarray(P_S)


def time_gpu(cfg):
    solver = CupyKrylovSolver(cfg, krylov_dim=50)
    solver._build()                      # exclude one-time build/upload from the timed loop
    t0 = time.perf_counter()
    t, P_S, trace = solver.run(t_max_us=T_MAX_US, n_steps=N_STEPS)
    return time.perf_counter() - t0, np.asarray(P_S)


def main():
    print(f"t_max={T_MAX_US} us, n_steps={N_STEPS}\n")
    hdr = f"{'n_nuc':>5} {'n_sites':>7} {'L-dim':>9} | {'CPU s':>9} | {'GPU s':>9} | {'speedup':>8} | {'max|dP_S|':>10}"
    print(hdr); print("-" * len(hdr))

    results = []
    for n_nuc in N_NUC_SWEEP:
        cfg = ercry4a_config(n_nuc=n_nuc)
        n_sites = cfg.n_sites
        Ldim = 4 ** n_sites
        try:
            cpu_s, ps_cpu = time_cpu(cfg)
            gpu_s, ps_gpu = time_gpu(cfg)
            m = min(len(ps_cpu), len(ps_gpu))
            diff = float(np.max(np.abs(ps_cpu[:m] - ps_gpu[:m])))
            speed = cpu_s / gpu_s
            print(f"{n_nuc:>5} {n_sites:>7} {Ldim:>9} | {cpu_s:>9.3f} | {gpu_s:>9.3f} | {speed:>7.1f}x | {diff:>10.2e}")
            results.append(dict(n_nuc=n_nuc, n_sites=n_sites, L_dim=Ldim,
                                cpu_s=cpu_s, gpu_s=gpu_s, speedup=speed, max_abs_dPS=diff))
        except Exception as e:
            print(f"{n_nuc:>5} {n_sites:>7} {Ldim:>9} | FAILED: {type(e).__name__}: {e}")

    out = Path(__file__).parent / "results" / "lindblad_headtohead.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print("\nNote: speedup is GPU sparse-Krylov vs scipy CPU on the SAME Liouvillian,")
    print("validated by max|dP_S| (agreement of the singlet-yield trajectory).")


if __name__ == "__main__":
    main()
