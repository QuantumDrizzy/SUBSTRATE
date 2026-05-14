"""
SUBSTRATE — Unified Research Pipeline
======================================
Multi-scale electromagnetic field analysis:
  quantum substrate → bio sensing → geomagnetic → heliospheric → cosmological

Usage:
  python pipeline.py --module all
  python pipeline.py --module cycle_project
  python pipeline.py --module cycle_project,cosmological
"""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent
MODULES = ROOT / "modules"

MODULE_RUNNERS = {
    "cycle_project": MODULES / "cycle_project" / "src" / "pipeline.py",
    # Add more as they are ready:
    # "quantum_lab":   MODULES / "quantum_lab" / "run.py",
    # "cryptotn_gpu":  MODULES / "cryptotn_gpu" / "run.py",
    # "magnon":        MODULES / "magnon" / "run.py",
}

def run_module(name: str):
    runner = MODULE_RUNNERS.get(name)
    if runner is None:
        print(f"[SUBSTRATE] Module '{name}' not yet wired into pipeline.")
        return
    if not runner.exists():
        print(f"[SUBSTRATE] Runner not found: {runner}")
        return
    print(f"\n{'='*60}\n  SUBSTRATE: running {name}\n{'='*60}")
    result = subprocess.run([sys.executable, str(runner)], cwd=runner.parent.parent.parent)
    if result.returncode != 0:
        print(f"[SUBSTRATE] {name} exited with code {result.returncode}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SUBSTRATE pipeline")
    parser.add_argument("--module", default="all", help="Module(s) to run, comma-separated, or 'all'")
    args = parser.parse_args()

    targets = list(MODULE_RUNNERS.keys()) if args.module == "all" else args.module.split(",")
    for mod in targets:
        run_module(mod.strip())
    print("\n[SUBSTRATE] Done.")
