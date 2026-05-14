"""
eeg_layer.py — SUBSTRATE EEG Layer v1.0

Dual-mode EEG layer for Muse 2 (or any LSL EEG source).

MODE SELECTION (automatic):
  - If a pylsl stream of type 'EEG' is found  → REAL mode (Muse 2 or any LSL source)
  - Otherwise                                  → SIMULATED mode (1/f noise + synthetic
                                                 band power, physiologically plausible)

Simulated mode produces data statistically indistinguishable from resting-state EEG
so the pipeline, GUI, and correlation engine can be built and tested before hardware
arrives. When the Muse 2 connects, set 'mode: auto' in substrate.toml and it activates
transparently.

Band definitions (Hz):
  delta 0.5–4 | theta 4–8 | alpha 8–13 | beta 13–30 | gamma 30–50

Score [0,1]: alpha relative power (higher = more relaxed / coherent state).

Target: Arch Linux · Python 3.10+ · optional pylsl / muselsl
"""
from __future__ import annotations

import time
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

_FS          = 256          # Muse 2 sample rate (Hz)
_N_CH        = 4            # EEG channels (TP9, AF7, AF8, TP10)
_WINDOW_S    = 4.0          # analysis window (s)
_RESOLVE_S   = 1.0          # LSL stream resolve timeout (s)
_BANDS = {
    "delta": (0.5,  4.0),
    "theta": (4.0,  8.0),
    "alpha": (8.0,  13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 50.0),
}

# Cache path for last real-mode result (allows stale fallback)
_CACHE_DIR  = Path.home() / ".cache" / "substrate" / "eeg"
_CACHE_FILE = _CACHE_DIR / "last_result.json"


# ─────────────────────────────────────────────────────────────
# HELPERS — band power from PSD
# ─────────────────────────────────────────────────────────────

def _band_power(psd: np.ndarray, freqs: np.ndarray, fmin: float, fmax: float) -> float:
    """Absolute band power via trapezoidal integration over PSD."""
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not mask.any():
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def _compute_bands(data: np.ndarray, fs: int) -> dict[str, float]:
    """
    Compute mean band power across channels using Welch PSD.
    data: shape (n_samples, n_channels)
    Returns dict band_name → relative power [0,1].
    """
    from scipy.signal import welch  # type: ignore[import]

    n_samples = data.shape[0]
    nperseg   = min(n_samples, fs * 2)       # 2-second segments
    total_abs  = {}
    freqs_ref  = None

    for ch in range(data.shape[1]):
        freqs, psd = welch(data[:, ch], fs=fs, nperseg=nperseg)
        if freqs_ref is None:
            freqs_ref = freqs
        for name, (flo, fhi) in _BANDS.items():
            total_abs[name] = total_abs.get(name, 0.0) + _band_power(psd, freqs, flo, fhi)

    # Average across channels
    n_ch = data.shape[1]
    abs_powers = {k: v / n_ch for k, v in total_abs.items()}

    # Relative power
    total = sum(abs_powers.values()) or 1.0
    return {k: v / total for k, v in abs_powers.items()}


# ─────────────────────────────────────────────────────────────
# SIMULATED MODE — 1/f noise + plausible alpha peak
# ─────────────────────────────────────────────────────────────

def _simulate_eeg(n_samples: int, fs: int, n_channels: int, rng: np.random.Generator) -> np.ndarray:
    """
    Generate synthetic EEG: 1/f background + alpha peak at 10 Hz + gaussian noise.
    Physiologically plausible for resting-state EEG.
    """
    t   = np.arange(n_samples) / fs
    out = np.zeros((n_samples, n_channels))

    for ch in range(n_channels):
        # 1/f colored noise (pink noise via inverse FFT)
        f_ax  = np.fft.rfftfreq(n_samples, d=1.0 / fs)
        with np.errstate(divide="ignore", invalid="ignore"):
            power = np.where(f_ax > 0, 1.0 / np.sqrt(f_ax), 0.0)
        phase = rng.uniform(0, 2 * np.pi, size=len(power))
        pink  = np.fft.irfft(power * np.exp(1j * phase), n=n_samples)
        pink  = pink * 2.0 / pink.std()                      # scale to ~2 µV RMS

        # Alpha oscillation (~10 Hz, 15-35 µV amplitude, slight per-channel jitter)
        alpha_freq = 10.0 + rng.uniform(-0.5, 0.5)
        alpha_amp  = rng.uniform(15.0, 35.0)
        alpha      = alpha_amp * np.sin(2 * np.pi * alpha_freq * t + rng.uniform(0, 2 * np.pi))

        # White noise floor (~5 µV RMS)
        noise = rng.normal(0, 5.0, n_samples)

        out[:, ch] = pink + alpha + noise

    return out


# ─────────────────────────────────────────────────────────────
# REAL MODE — Muse 2 via LSL
# ─────────────────────────────────────────────────────────────

def _find_lsl_stream() -> Any | None:
    """Return first LSL EEG inlet found within _RESOLVE_S, or None."""
    try:
        import pylsl  # type: ignore[import]
        streams = pylsl.resolve_byprop("type", "EEG", timeout=_RESOLVE_S)
        if not streams:
            return None
        inlet = pylsl.StreamInlet(streams[0])
        logger.info(f"EEG: found LSL stream '{streams[0].name()}'")
        return inlet
    except Exception as exc:
        logger.debug(f"EEG: LSL not available — {exc}")
        return None


def _pull_lsl_window(inlet: Any, n_samples: int) -> np.ndarray | None:
    """Pull `n_samples` EEG samples from an LSL inlet. Returns (n_samples, n_ch) or None."""
    try:
        samples, _ = inlet.pull_chunk(timeout=_WINDOW_S * 1.5, max_samples=n_samples * 2)
        if not samples or len(samples) < n_samples // 2:
            return None
        arr = np.array(samples, dtype=np.float32)
        # Trim / pad to exactly n_samples
        if arr.shape[0] >= n_samples:
            return arr[-n_samples:, :]
        return None
    except Exception as exc:
        logger.warning(f"EEG: LSL pull failed — {exc}")
        return None


# ─────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────

def _save_cache(result: dict) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w") as f:
        json.dump(result, f)


def _load_cache() -> dict | None:
    if not _CACHE_FILE.exists():
        return None
    try:
        with open(_CACHE_FILE) as f:
            return json.load(f)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run(params: dict = None) -> dict:
    """
    SUBSTRATE layer entrypoint.

    Returns dict with keys:
      layer, score, data {mode, band_powers, alpha_rel, latency_ms, ...}

    Score = alpha relative power [0, 1].
    Higher score → stronger alpha → more relaxed / coherent cortical state.
    """
    if params is None:
        params = {}

    t0       = time.perf_counter()
    n_samples = int(_FS * _WINDOW_S)
    mode_req  = params.get("mode", "auto")   # "auto" | "simulated" | "real"
    rng       = np.random.default_rng(int(time.time() * 1000) & 0xFFFFFFFF)

    data_array: np.ndarray | None = None
    mode_used  = "simulated"
    stream_name = ""

    # ── Try real mode ─────────────────────────────────────────
    inlet = None
    if mode_req in ("auto", "real"):
        inlet = _find_lsl_stream()
        if inlet is not None:
            data_array = _pull_lsl_window(inlet, n_samples)
            if data_array is not None:
                mode_used   = "real"
                stream_name = "lsl_eeg"
            else:
                logger.warning("EEG: LSL stream found but no data pulled; falling back to simulated")
                inlet = None

    # ── Simulated fallback ────────────────────────────────────
    if data_array is None:
        data_array = _simulate_eeg(n_samples, _FS, _N_CH, rng)
        mode_used  = "simulated"

    # ── Band analysis ─────────────────────────────────────────
    try:
        band_powers = _compute_bands(data_array, _FS)
        has_scipy   = True
    except ImportError:
        # scipy not available — rough FFT fallback
        has_scipy = False
        fft_data  = np.abs(np.fft.rfft(data_array[:, 0]))
        freqs_fft = np.fft.rfftfreq(n_samples, d=1.0 / _FS)
        band_powers = {}
        total = 0.0
        for name, (flo, fhi) in _BANDS.items():
            mask = (freqs_fft >= flo) & (freqs_fft <= fhi)
            p = float(fft_data[mask].mean()) if mask.any() else 0.0
            band_powers[name] = p
            total += p
        if total > 0:
            band_powers = {k: v / total for k, v in band_powers.items()}

    alpha_rel = float(band_powers.get("alpha", 0.0))
    score     = float(np.clip(alpha_rel, 0.0, 1.0))

    latency_ms = (time.perf_counter() - t0) * 1000.0

    result = {
        "layer": "eeg",
        "score": round(score, 4),
        "data": {
            "mode":          mode_used,
            "stream":        stream_name,
            "fs_hz":         _FS,
            "n_channels":    data_array.shape[1],
            "window_s":      _WINDOW_S,
            "band_powers":   {k: round(v, 4) for k, v in band_powers.items()},
            "alpha_rel":     round(alpha_rel, 4),
            "has_scipy":     has_scipy,
            "latency_ms":    round(latency_ms, 2),
            "synthetic":     mode_used == "simulated",
        },
    }

    if mode_used == "real":
        _save_cache(result)

    return result


if __name__ == "__main__":
    import pprint
    print("=== SUBSTRATE EEG Layer v1.0 ===")
    out = run()
    pprint.pprint(out)
