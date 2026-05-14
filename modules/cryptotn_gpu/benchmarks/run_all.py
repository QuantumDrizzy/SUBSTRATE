"""
benchmarks/run_all.py — run all three benchmarks sequentially.
usage: python benchmarks/run_all.py [--fast]
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bench_fmo import run_fmo
from bench_ercry4a import run_ercry4a
from bench_tetrad_trp import run_tetrad

parser = argparse.ArgumentParser()
parser.add_argument("--fast", action="store_true",
                    help="quick smoke test (reduced steps/spins)")
args = parser.parse_args()

print("\n" + "=" * 60)
print(" cryptotn-gpu benchmark suite")
print("=" * 60 + "\n")

if args.fast:
    r1a = run_fmo(T_K=77.0,  t_max_fs=500.0, n_steps=100)
    r1b = run_fmo(T_K=300.0, t_max_fs=500.0, n_steps=100)
    r2  = run_ercry4a(n_nuc=6, t_max_us=3.0, n_steps=100)
    r3  = run_tetrad(t_max_us=5.0, n_steps=200)
else:
    r1a = run_fmo(T_K=77.0,  t_max_fs=1000.0, n_steps=500)
    r1b = run_fmo(T_K=300.0, t_max_fs=1000.0, n_steps=500)
    r2  = run_ercry4a(n_nuc=10, t_max_us=10.0, n_steps=300)
    r3  = run_tetrad(t_max_us=20.0, n_steps=800)

print("\n" + "=" * 60)
print(" summary")
print("=" * 60)
print(f"FMO  77K | RMSE vs TENSO: {r1a['rmse_vs_tenso']:.5f} | {r1a['wall_time_s']:.2f}s")
print(f"FMO 300K | RMSE vs TENSO: {r1b['rmse_vs_tenso']:.5f} | {r1b['wall_time_s']:.2f}s")
print(f"ErCry4a  | ΔΦ_S: {r2['delta_phi_s_earth']:.5f}             | {r2['wall_time_s']:.2f}s")
print(f"Tetrad   | Φ_S: {r3['phi_s']:.5f}                | {r3['wall_time_s']:.2f}s")
print("\nall results saved to benchmarks/results/")
