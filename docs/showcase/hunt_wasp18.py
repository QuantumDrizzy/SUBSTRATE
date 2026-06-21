#!/usr/bin/env python3
"""
Recover a real exoplanet transit from real NASA/TESS data — reproducible.

The standard pipeline, no inflation: lightkurve -> MAST -> detrend -> BLS
periodogram -> phase-fold. WASP-18 b is a hot Jupiter, so its ~1% transit is
deep and clearly visible once folded.

    pip install lightkurve numpy matplotlib
    python hunt_wasp18.py     # -> exoplanet-transit-WASP18.png

Expected: BLS recovers period ~0.9416 d (true 0.9415 d) and a ~1.1% transit.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
import warnings
warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exoplanet-transit-WASP18.png")

print("[hunt] MAST query: WASP-18 (a hot Jupiter)")
sr = lk.search_lightcurve("WASP-18", author="SPOC", exptime=120)
if len(sr) == 0:
    sr = lk.search_lightcurve("WASP-18")
print(f"[hunt] {len(sr)} results; downloading 2 sectors of real TESS photons...")
lc = sr[:2].download_all().stitch().remove_nans().remove_outliers(sigma=5).flatten(window_length=901)

pg = lc.to_periodogram(method="bls", period=np.arange(0.5, 5.0, 0.0008))
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

plt.style.use("default")  # light, clean — not everything has to be dark
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"WASP-18 b — a hot Jupiter recovered from real TESS data  (BLS period {P:.4f} d)",
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
