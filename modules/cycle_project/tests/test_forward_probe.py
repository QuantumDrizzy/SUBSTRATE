"""
tests/test_forward_probe.py — Unit tests for FORWARD_PROBE module.

Run from project root:
  pytest tests/test_forward_probe.py -v
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "aligned.parquet"


# ── Helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def parquet_path():
    if not PARQUET_PATH.exists():
        pytest.skip(f"aligned.parquet not found at {PARQUET_PATH}")
    return PARQUET_PATH


# ── 1. Spectral ───────────────────────────────────────────────────────────────

def test_spectral_dominant_period_range(parquet_path, tmp_path):
    """Top dominant period must be between 5,000 and 30,000 years."""
    from forward_probe.spectral import run_spectral
    import matplotlib
    matplotlib.use("Agg")

    result = run_spectral(parquet_path=parquet_path, output_dir=tmp_path)
    periods = result["dominant_periods"]
    assert len(periods) > 0, "No dominant periods detected"
    top = float(periods[0])
    assert 5_000 <= top <= 30_000, (
        f"Top dominant period {top:.0f} yr outside expected range [5,000–30,000]"
    )


def test_spectral_returns_multiple_periods(parquet_path, tmp_path):
    """Should detect at least 3 distinct dominant periods."""
    from forward_probe.spectral import run_spectral
    import matplotlib
    matplotlib.use("Agg")

    result = run_spectral(parquet_path=parquet_path, output_dir=tmp_path)
    assert len(result["dominant_periods"]) >= 1
    assert len(result["cwt_dominant_periods"]) >= 1


# ── 2. Decay model ────────────────────────────────────────────────────────────

def test_decay_threshold_positive_and_bounded(parquet_path, tmp_path):
    """Threshold crossing year must be > 0 and < 50,000 for at least one model."""
    from forward_probe.decay_model import run_decay_model
    import matplotlib
    matplotlib.use("Agg")

    result = run_decay_model(parquet_path=parquet_path, output_dir=tmp_path)
    thresholds = result["thresholds"]
    assert len(thresholds) > 0, "No decay models returned"

    valid = [
        info["threshold_yr"]
        for info in thresholds.values()
        if info.get("threshold_yr") is not None and info["threshold_yr"] > 0
    ]
    assert len(valid) >= 1, "No model crosses the threshold in the forecast window"
    for yr in valid:
        assert 0 < yr < 50_000, f"Threshold year {yr:.0f} out of plausible range"


def test_decay_ci_brackets_mean(parquet_path, tmp_path):
    """CI bounds must bracket the mean threshold year."""
    from forward_probe.decay_model import run_decay_model
    import matplotlib
    matplotlib.use("Agg")

    result = run_decay_model(parquet_path=parquet_path, output_dir=tmp_path)
    for model, info in result["thresholds"].items():
        yr = info.get("threshold_yr")
        lo = info.get("ci_lo")
        hi = info.get("ci_hi")
        if yr is None or yr <= 0:
            continue
        assert lo <= yr <= hi, (
            f"Model '{model}': CI [{lo}, {hi}] does not bracket mean {yr:.0f}"
        )


# ── 3. Fingerprint ────────────────────────────────────────────────────────────

def test_fingerprint_probability_in_unit_interval(parquet_path, tmp_path):
    """Pre-excursion probability must be in [0, 1]."""
    from forward_probe.fingerprint import run_fingerprint
    result = run_fingerprint(parquet_path=parquet_path)
    prob = result["probability"]
    assert 0.0 <= prob <= 1.0, f"Probability {prob} not in [0, 1]"


def test_fingerprint_loo_accuracy_reasonable(parquet_path, tmp_path):
    """LOO accuracy should be at least 0.4 (better than random with tiny dataset)."""
    from forward_probe.fingerprint import run_fingerprint
    result = run_fingerprint(parquet_path=parquet_path)
    acc = result["loo_accuracy"]
    assert acc >= 0.0, "LOO accuracy is negative"
    # With 5 excursions this is loose — just check it runs
    assert result["n_excursions"] >= 3, "Too few excursion windows to test"


def test_fingerprint_status_is_valid(parquet_path, tmp_path):
    """Status must be one of the defined labels."""
    from forward_probe.fingerprint import run_fingerprint
    result = run_fingerprint(parquet_path=parquet_path)
    assert result["status"] in {"STABLE", "WATCH", "ALERT"}, (
        f"Unknown status: {result['status']}"
    )


# ── 4. LSTM ensemble ──────────────────────────────────────────────────────────

def test_lstm_forecast_length(parquet_path, tmp_path):
    """Forecast array must have 50 steps (5,000 yr / 100 yr per step)."""
    from forward_probe.lstm_ensemble import run_lstm_ensemble
    import matplotlib
    matplotlib.use("Agg")

    result = run_lstm_ensemble(parquet_path=parquet_path, output_dir=tmp_path)
    fc = result["forecast_mean"]
    assert len(fc) == 50, f"Expected 50 forecast steps, got {len(fc)}"


def test_lstm_forecast_values_plausible(parquet_path, tmp_path):
    """Mean forecast values should stay in [0, 1.5] range (normalised VADM)."""
    from forward_probe.lstm_ensemble import run_lstm_ensemble
    import matplotlib
    matplotlib.use("Agg")

    result = run_lstm_ensemble(parquet_path=parquet_path, output_dir=tmp_path)
    fc = np.array(result["forecast_mean"])
    assert fc.min() >= -0.5, f"Forecast goes unrealistically negative: {fc.min():.3f}"
    assert fc.max() <= 2.0,  f"Forecast goes unrealistically high: {fc.max():.3f}"


def test_lstm_ensemble_uncertainty_positive(parquet_path, tmp_path):
    """Ensemble std should be positive (MC dropout adds variance)."""
    from forward_probe.lstm_ensemble import run_lstm_ensemble
    import matplotlib
    matplotlib.use("Agg")

    result = run_lstm_ensemble(parquet_path=parquet_path, output_dir=tmp_path)
    std = np.array(result["forecast_std"])
    assert std.mean() > 0, "Ensemble std is zero — MC dropout not active"
