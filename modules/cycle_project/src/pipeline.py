"""
pipeline.py — Full cycle_project pipeline runner.
Runs all 5 modules in dependency order.
"""
import subprocess, sys, json
from pathlib import Path

ROOT = Path(__file__).parent.parent

steps = [
    ("CYCLE_DETECT",     [sys.executable, str(ROOT/"src/cycle_detect/gnn_prototype.py")]),
    ("FORWARD_PROBE",    [sys.executable, str(ROOT/"src/forward_probe/run_forward_probe.py")]),
    # myth_rag and pole_shift_sim are run separately (RAG needs setup, LBM needs --args)
    ("UNIFIED_FIGURE",   [sys.executable, str(ROOT/"src/unified_figure.py")]),
]

for name, cmd in steps:
    print(f"\n{'='*60}\n  PIPELINE: {name}\n{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        print(f"[PIPELINE] {name} failed with code {result.returncode}")
        sys.exit(1)

print("\n[PIPELINE] All steps complete.")
print("[PIPELINE] data/processed/ now contains:")
for f in sorted((ROOT/"data/processed").glob("*.json")):
    size = f.stat().st_size
    print(f"  {f.name} ({size} bytes)")
