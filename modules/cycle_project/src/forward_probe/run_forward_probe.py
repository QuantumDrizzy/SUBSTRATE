"""
run_forward_probe.py — FORWARD_PROBE: Module 5 of cycle_project.

Runs all 4 submodules in sequence:
  1. spectral     — dominant VADM cycles (FFT + CWT)
  2. decay_model  — threshold crossing forecast (3 models)
  3. fingerprint  — pre-excursion pattern classifier
  4. lstm_ensemble — Monte Carlo forward projection

Usage:
  cd cycle_project
  python src/forward_probe/run_forward_probe.py
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forward_probe.spectral       import run_spectral
from forward_probe.decay_model    import run_decay_model
from forward_probe.fingerprint    import run_fingerprint
from forward_probe.lstm_ensemble  import run_lstm_ensemble


BANNER = """
╔══════════════════════════════════════════════════════════════╗
║           FORWARD_PROBE — CYCLE_PROJECT MODULE 5             ║
║     Geomagnetic Field Forward Projection & Risk Assessment   ║
╚══════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)

    # ── 1. Spectral analysis ──────────────────────────────────────────────────
    print("\n── [1/4] SPECTRAL ANALYSIS ──────────────────────────────────────\n")
    spec = run_spectral()
    top_periods = spec.get("dominant_periods", [])

    # ── 2. Decay model ────────────────────────────────────────────────────────
    print("\n── [2/4] DECAY MODEL ────────────────────────────────────────────\n")
    decay = run_decay_model()
    thresholds = decay.get("thresholds", {})

    # ── 3. Pre-excursion fingerprint ──────────────────────────────────────────
    print("\n── [3/4] PRE-EXCURSION FINGERPRINT ─────────────────────────────\n")
    fp = run_fingerprint()
    prob   = fp.get("probability", float("nan"))
    status = fp.get("status", "UNKNOWN")

    # ── 4. LSTM ensemble ──────────────────────────────────────────────────────
    print("\n── [4/4] LSTM ENSEMBLE FORECAST ─────────────────────────────────\n")
    lstm   = run_lstm_ensemble()
    fc_mean = lstm.get("forecast_mean", [])
    fc_std  = lstm.get("forecast_std",  [])
    future  = lstm.get("future_yr",     [])

    def _yr_val(target_yr):
        """Return (mean, std) at a target year from now."""
        import numpy as np
        future_arr = list(future)
        fc_arr     = list(fc_mean)
        fc_std_arr = list(fc_std)
        diffs = [abs(y - target_yr) for y in future_arr]
        idx   = diffs.index(min(diffs))
        return fc_arr[idx], fc_std_arr[idx]

    m1k, s1k = _yr_val(1_000)
    m5k, s5k = _yr_val(5_000)

    # ── Summary report ────────────────────────────────────────────────────────
    print("""
═══════════════════════════════════════════════════════════════
  FORWARD PROBE — SUMMARY REPORT
═══════════════════════════════════════════════════════════════""")

    if top_periods:
        periods_str = ", ".join(f"{p:,.0f} yr" for p in top_periods[:3])
        print(f"  [SPECTRUM]    Dominant VADM cycles : {periods_str}")
    else:
        print("  [SPECTRUM]    No dominant periods detected")

    decay_results = decay.get("results", {})
    for model_name, yr in thresholds.items():
        res = decay_results.get(model_name, {})
        lo, hi = res.get("ci", (float("nan"), float("nan")))
        import math
        if yr and math.isfinite(yr) and yr > 0:
            ci_str = (f"(95% CI {lo:,.0f}–{hi:,.0f})"
                      if math.isfinite(lo) and math.isfinite(hi) else "(CI n/a)")
            print(f"  [DECAY/{model_name:<9}] Threshold crossing : ~{yr:,.0f} yr  {ci_str}")
        else:
            print(f"  [DECAY/{model_name:<9}] Threshold crossing : not reached within forecast window")

    print(f"  [FINGERPRINT] Pre-excursion prob    : {prob:.3f}  →  {status}")
    print(f"  [LSTM]        VADM @ +1,000 yr      : {m1k:.3f} ± {s1k:.3f}")
    print(f"  [LSTM]        VADM @ +5,000 yr      : {m5k:.3f} ± {s5k:.3f}")

    print("""═══════════════════════════════════════════════════════════════
  Outputs saved to data/processed/
    vadm_spectrum.png   — FFT + CWT power spectrum
    vadm_forecast.png   — 3-model decay projection
    vadm_lstm_forecast.png — LSTM ensemble with uncertainty band
═══════════════════════════════════════════════════════════════
""")

    # Save machine-readable state for field_coherence_monitor
    import math
    threshold_values = list(thresholds.values())
    best_threshold = next(
        (int(v) for v in threshold_values if v and math.isfinite(v) and v > 0),
        0,
    )
    probe_state = {
        "pre_excursion_prob": round(prob, 6) if math.isfinite(prob) else 0.0,
        "lstm_vadm_1kyr": round(m1k, 6) if math.isfinite(m1k) else 0.0,
        "lstm_vadm_5kyr": round(m5k, 6) if math.isfinite(m5k) else 0.0,
        "instrumental_threshold_yr": best_threshold,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    probe_path = PROJECT_ROOT / "data" / "processed" / "probe_state.json"
    with open(probe_path, "w") as fh:
        json.dump(probe_state, fh, indent=2)
    print(f"[FORWARD_PROBE] probe_state.json → {probe_path}")


if __name__ == "__main__":
    main()
