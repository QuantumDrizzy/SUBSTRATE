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
    # "magnon":        MODULES / "magnon" / "run.py",
}

# Modules with custom Python entry points (not subprocess runners)
INLINE_RUNNERS = {
    "cryptotn_gpu": "nexus.substrate_material_bridge:run_material_bridge",
}

def run_module(name: str):
    # Check inline Python runners first
    inline = INLINE_RUNNERS.get(name)
    if inline:
        print(f"\n{'='*60}\n  SUBSTRATE: running {name} (inline)\n{'='*60}")
        mod_path, func_name = inline.rsplit(":", 1)
        # ensure SUBSTRATE root is in path
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        try:
            import importlib
            mod = importlib.import_module(mod_path)
            fn  = getattr(mod, func_name)
            result = fn()
            if result:
                import json
                print(json.dumps(result, indent=2))
        except Exception as exc:
            print(f"[SUBSTRATE] {name} failed: {exc}")
        return

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

    all_modules = list(MODULE_RUNNERS.keys()) + list(INLINE_RUNNERS.keys())
    targets = all_modules if args.module == "all" else args.module.split(",")
    for mod in targets:
        run_module(mod.strip())
    print("\n[SUBSTRATE] Done.")
