#!/usr/bin/env python3
"""Exp 0 — Pearson correlation analysis: Dst(nT) vs singlet_yield (Y_s).

Usage:
    python tools/pearson_analysis.py
"""
from __future__ import annotations

import json
from pathlib import Path

LOG_FILE = Path.home() / ".cache" / "substrate" / "experiment_0" / "log.jsonl"

def main():
    rows = []
    errors = 0
    with LOG_FILE.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("singlet_yield") is not None and r.get("dst_nT") is not None:
                    rows.append((float(r["dst_nT"]), float(r["singlet_yield"]), r["utc"]))
                else:
                    errors += 1
            except Exception:
                errors += 1

    print(f"\nExp 0 — Pearson Analysis")
    print(f"{'='*50}")
    print(f"Total lines on disk : {errors + len(rows)}")
    print(f"Valid samples        : {len(rows)}")
    print(f"Dropped (errors/null): {errors}")

    if len(rows) < 10:
        print("Not enough data.")
        return

    dst_vals = [r[0] for r in rows]
    ys_vals  = [r[1] for r in rows]
    times    = [r[2] for r in rows]

    # Basic stats
    dst_min, dst_max = min(dst_vals), max(dst_vals)
    ys_min,  ys_max  = min(ys_vals),  max(ys_vals)
    dst_mean = sum(dst_vals) / len(dst_vals)
    ys_mean  = sum(ys_vals)  / len(ys_vals)
    dst_range = dst_max - dst_min
    ys_range  = ys_max  - ys_min

    print(f"\nTime range  : {times[0]}  →  {times[-1]}")
    print(f"Dst  range  : [{dst_min:+.1f}, {dst_max:+.1f}] nT  (Δ={dst_range:.1f} nT, mean={dst_mean:+.2f})")
    print(f"Y_s  range  : [{ys_min:.8f}, {ys_max:.8f}]  (Δ={ys_range:.2e})")

    # Pearson r
    n = len(rows)
    cov  = sum((d - dst_mean) * (y - ys_mean) for d, y, _ in rows) / n
    std_d = (sum((d - dst_mean)**2 for d, y, _ in rows) / n) ** 0.5
    std_y = (sum((y - ys_mean)**2 for d, y, _ in rows) / n) ** 0.5

    if std_d < 1e-12 or std_y < 1e-12:
        print("\nPearson r  : N/A — one variable is constant (no Dst variation during window)")
        print("\nDiagnosis  : Dst has been stable throughout the experiment window.")
        print("             A geomagnetic storm or disturbance is needed to test the cable.")
        print("             Keep logging — the next Kp≥4 event will provide the signal.")
        return

    r_pearson = cov / (std_d * std_y)

    print(f"\nPearson r   : {r_pearson:+.6f}")

    # Interpretation
    abs_r = abs(r_pearson)
    if abs_r > 0.7:
        strength = "STRONG"
    elif abs_r > 0.4:
        strength = "MODERATE"
    elif abs_r > 0.2:
        strength = "WEAK"
    else:
        strength = "NEGLIGIBLE"

    direction = "negative" if r_pearson < 0 else "positive"
    print(f"Strength    : {strength} {direction} correlation")

    # Physical interpretation
    print(f"\nPhysical interpretation:")
    if abs_r > 0.4 and r_pearson < 0:
        print("  ✓ Dst↓ (storm) → Bh↑ → dephasing↑ → Y_s↓  [expected from anisotropic T2 model]")
    elif abs_r > 0.4 and r_pearson > 0:
        print("  ✗ Dst↑ → Y_s↑  [unexpected — check sign convention in _dst_to_b_earth()]")
    else:
        print("  ~ Insufficient Dst variation to confirm anisotropic T2 sensitivity in this window.")
        print("    Expected: r < -0.4 during a storm event (Kp ≥ 4).")

    # Dst variation breakdown
    print(f"\nDst variation breakdown:")
    from collections import Counter
    dst_counts = Counter(round(d) for d in dst_vals)
    for val in sorted(dst_counts):
        pct = dst_counts[val] / n * 100
        bar = "█" * int(pct / 2)
        print(f"  {val:+4.0f} nT : {dst_counts[val]:5d} samples ({pct:5.1f}%) {bar}")

if __name__ == "__main__":
    main()
