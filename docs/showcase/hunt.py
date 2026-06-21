#!/usr/bin/env python3
"""
General exoplanet transit hunter — point it at any target with TESS data.

    python hunt.py "WASP-39"     # a hot Saturn (JWST found CO2 in its atmosphere)
    python hunt.py "WASP-18"     # a hot Jupiter
    python hunt.py "TIC 28159019"

The standard, honest pipeline (no inflation): lightkurve -> NASA/MAST -> detrend
-> BLS periodogram -> phase-fold. Real spacecraft data; the figure is reproducible.
This is research-engineering: a tool that *recovers and validates* a discovery from
real public data — which is the actual day-to-day of the work.

ALWAYS VALIDATE the recovered period against the literature. With sparse coverage the
BLS can alias (e.g. WASP-18 b recovers cleanly: 0.9416 d vs true 0.9415 d ✓; WASP-39 b
with only 2 sectors aliased to 7.52 d vs the true 4.055 d ✗). The recovered number is a
candidate, not gospel — that validation step IS the science.
"""
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import warnings
warnings.filterwarnings("ignore")

TARGET = sys.argv[1] if len(sys.argv) > 1 else "WASP-18"
slug = re.sub(r"[^A-Za-z0-9]+", "-", TARGET).strip("-")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"exoplanet-transit-{slug}.png")

print(f"[hunt] MAST query: {TARGET}")
sr = lk.search_lightcurve(TARGET, author="SPOC", exptime=120)
if len(sr) == 0:
    sr = lk.search_lightcurve(TARGET)
if len(sr) == 0:
    raise SystemExit(f"no TESS data for {TARGET}")
print(f"[hunt] {len(sr)} results; downloading 2 sectors of real TESS photons...")
lc = sr[:2].download_all().stitch().remove_nans().remove_outliers(sigma=5).flatten(window_length=901)

pg = lc.to_periodogram(method="bls", period=np.arange(0.5, 8.0, 0.001))
P = float(pg.period_at_max_power.value)
t0 = pg.transit_time_at_max_power
print(f"[hunt] BLS recovered period = {P:.5f} d")

folded = lc.fold(period=P, epoch_time=t0)
ph = np.asarray(folded.time.value); fl = np.asarray(folded.flux.value)
m = np.isfinite(ph) & np.isfinite(fl); ph, fl = ph[m], fl[m]
nb = 140
bins = np.linspace(ph.min(), ph.max(), nb + 1)
idx = np.digitize(ph, bins)
bx = 0.5 * (bins[:-1] + bins[1:])
by = np.array([np.nanmedian(fl[idx == i + 1]) if np.any(idx == i + 1) else np.nan for i in range(nb)])

plt.style.use("default")  # light, clean
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"{TARGET} — exoplanet transit recovered from real TESS data  (BLS period {P:.4f} d)",
             fontweight="bold", fontsize=13)
ax1.scatter(lc.time.value, lc.flux.value, s=1, c="#2563eb", alpha=0.3)
ax1.set_title("Detrended light curve (NASA / MAST)", fontsize=11)
ax1.set_xlabel("Time (BTJD)"); ax1.set_ylabel("Normalized flux"); ax1.grid(alpha=0.25)
ax2.scatter(ph, fl, s=2, c="#94a3b8", alpha=0.22)
ax2.plot(bx, by, c="#dc2626", lw=2.3, label="binned — the transit")
ax2.set_title("Phase-folded — the transit 'shadow'", fontsize=11)
ax2.set_xlabel("Phase (days)"); ax2.set_ylabel("Normalized flux"); ax2.grid(alpha=0.25); ax2.legend(loc="lower left")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT, dpi=150, facecolor="white")
print(f"[hunt] SAVED: {OUT}  | transit depth recovered: {(1 - np.nanmin(by)) * 100:.2f}%")
