# requirements: numpy, scipy, pandas, pyarrow, matplotlib

import re
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

PROJECT_ROOT = Path(__file__).parents[2]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "aligned.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DT = 100.0  # years per sample
PERIOD_MIN = 1_000.0
PERIOD_MAX = 50_000.0


def _load_vadm(parquet_path):
    df = pd.read_parquet(parquet_path).sort_values("age_bp").reset_index(drop=True)
    series = df["sint2000_norm"]
    # Interpolate sparse NaNs (edge NaNs are forward/back filled)
    series = series.interpolate(method="linear").ffill().bfill()
    return series.values


def _fft_periods(vadm):
    """Return (periods, power) arrays filtered to [PERIOD_MIN, PERIOD_MAX]."""
    detrended = scipy.signal.detrend(vadm)
    fft_vals = np.fft.rfft(detrended)
    freqs = np.fft.rfftfreq(len(vadm), d=DT)

    pos = freqs > 0
    freqs_pos = freqs[pos]
    power_pos = np.abs(fft_vals[pos]) ** 2
    periods = 1.0 / freqs_pos

    mask = (periods >= PERIOD_MIN) & (periods <= PERIOD_MAX)
    return periods[mask], power_pos[mask]


def _ricker_wavelet(width: float, n_points: int) -> np.ndarray:
    """Ricker (Mexican hat) wavelet — replaces removed scipy.signal.ricker."""
    t = np.arange(n_points) - (n_points - 1) / 2.0
    A = 2.0 / (np.sqrt(3.0 * width) * np.pi ** 0.25)
    return A * (1.0 - (t / width) ** 2) * np.exp(-(t ** 2) / (2.0 * width ** 2))


def _cwt_periods(vadm):
    """Return (cwt_periods, mean_cwt_power) filtered to [PERIOD_MIN, PERIOD_MAX].

    Uses manual Ricker CWT via FFT convolution — compatible with all scipy versions
    (scipy.signal.cwt was removed in scipy 1.12).
    """
    detrended = scipy.signal.detrend(vadm)
    widths = np.arange(2, 160, 2, dtype=float)

    cwt_mat = np.zeros((len(widths), len(detrended)))
    for i, w in enumerate(widths):
        n_pts = min(max(int(10 * w) | 1, 5), len(detrended))  # odd, capped
        if n_pts % 2 == 0:
            n_pts += 1
        wav = _ricker_wavelet(w, n_pts)
        cwt_mat[i] = scipy.signal.fftconvolve(detrended, wav, mode="same")

    mean_power = np.mean(cwt_mat ** 2, axis=1)

    # peak period of Ricker at width w (samples): sqrt(2)*pi*w*dt
    cwt_per = np.sqrt(2) * np.pi * widths * DT
    mask = (cwt_per >= PERIOD_MIN) & (cwt_per <= PERIOD_MAX)
    return cwt_per[mask], mean_power[mask]


def _top_peaks(periods, power, n=5, prominence_frac=0.03):
    """Return up to n dominant periods sorted by descending power."""
    peaks, _ = scipy.signal.find_peaks(power, prominence=power.max() * prominence_frac)
    if len(peaks) == 0:
        peaks = np.argsort(power)[::-1][:n]
    else:
        peaks = peaks[np.argsort(power[peaks])[::-1]]
    return periods[peaks[:n]]


def run_spectral(parquet_path=None, output_dir=None):
    parquet_path = Path(parquet_path or PARQUET_PATH)
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[SPECTRUM] Loading VADM time series...")
    vadm = _load_vadm(parquet_path)
    N = len(vadm)
    print(f"[SPECTRUM] {N} samples × {DT:.0f} yr/sample = {N*DT:,.0f} yr span")

    print("[SPECTRUM] Running FFT...")
    fft_per, fft_pow = _fft_periods(vadm)
    top_fft = _top_peaks(fft_per, fft_pow, n=5)

    print("[SPECTRUM] Running CWT (Ricker)...")
    cwt_per, cwt_pow = _cwt_periods(vadm)
    top_cwt = _top_peaks(cwt_per, cwt_pow, n=5)

    print(f"[SPECTRUM] Top 5 FFT periods: {[f'{p:,.0f} yr' for p in top_fft]}")
    print(f"[SPECTRUM] Top 5 CWT periods: {[f'{p:,.0f} yr' for p in top_cwt]}")

    # --- Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(12, 9), facecolor="#111111")
    fig.suptitle("VADM Spectral Analysis (Sint-2000)", fontsize=14, color="white", fontweight="bold")

    colours = ["#ff4444", "#44ff88", "#ffcc00", "#44ccff", "#ff88ff"]

    ax1 = axes[0]
    ax1.set_facecolor("#1a1a2e")
    ax1.semilogy(fft_per / 1000, fft_pow, color="#88aaff", lw=0.9, alpha=0.8, label="FFT power")
    for i, p in enumerate(top_fft[:5]):
        ax1.axvline(p / 1000, color=colours[i], alpha=0.8, lw=1.4, ls="--", label=f"{p:,.0f} yr")
    ax1.set_xlabel("Period (kyr)", color="white")
    ax1.set_ylabel("Power (log)", color="white")
    ax1.set_title("Power Spectrum (FFT, detrended)", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(fontsize=8, facecolor="#222", labelcolor="white")
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    ax2.plot(cwt_per / 1000, cwt_pow, color="#88ffcc", lw=1.5, label="CWT mean power")
    for i, p in enumerate(top_cwt[:3]):
        ax2.axvline(p / 1000, color=colours[i], alpha=0.7, lw=1.4, ls="--", label=f"{p:,.0f} yr")
    ax2.set_xlabel("Period (kyr)", color="white")
    ax2.set_ylabel("Mean CWT power", color="white")
    ax2.set_title("Scalogram Summary (CWT, Ricker wavelet)", color="white")
    ax2.tick_params(colors="white")
    ax2.legend(fontsize=8, facecolor="#222", labelcolor="white")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = output_dir / "vadm_spectrum.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[SPECTRUM] Saved → {out_path}")
    plt.show()
    plt.close()

    return {
        "dominant_periods": list(top_fft),
        "cwt_dominant_periods": list(top_cwt),
    }


if __name__ == "__main__":
    run_spectral()
