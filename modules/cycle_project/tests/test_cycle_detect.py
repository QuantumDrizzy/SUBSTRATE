"""
Tests for CYCLE_DETECT module.
Uses actual API from gnn_numpy.py: make_windows, correlation_adjacency,
graph_diffuse, reconstruction_error, fit_and_score.
CPU-only, no GPU or NOAA network required.
"""
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cycle_detect.gnn_numpy import (
    make_windows,
    correlation_adjacency,
    graph_diffuse,
    reconstruction_error,
    fit_and_score,
)

PROXIES = ["gisp2_d18o", "vostok_dd", "grip_be10", "sint2000_vadm"]


@pytest.fixture(scope="module")
def clean_df():
    """Aligned 4-proxy DataFrame with Laschamp and YD signatures embedded."""
    rng = np.random.default_rng(42)
    ages = np.arange(0, 80_001, 100, dtype=float)
    n = len(ages)
    df = pd.DataFrame({"age_bp": ages})
    for col in PROXIES:
        df[col] = rng.normal(0, 1, n)

    # Laschamp ~41 ka: VADM collapse + Be-10 spike (extended for window coverage)
    la = np.argmin(np.abs(ages - 41_000))
    df.loc[la-10:la+10, "sint2000_vadm"] -= 6.0
    df.loc[la-10:la+10, "grip_be10"]     += 7.0

    # Younger Dryas ~12.9 ka: d18O drop + Be-10 spike (extended)
    yd = np.argmin(np.abs(ages - 12_900))
    df.loc[yd-8:yd+8, "gisp2_d18o"]   -= 5.0
    df.loc[yd-8:yd+8, "grip_be10"]    += 5.0

    # Z-score
    for col in PROXIES:
        df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-9)
    return df


# ── make_windows ──────────────────────────────────────────────────────────────

class TestMakeWindows:
    def test_returns_tuple(self, clean_df):
        result = make_windows(clean_df, PROXIES, window_size=50, stride=5)
        assert isinstance(result, tuple) and len(result) == 2

    def test_window_shape(self, clean_df):
        windows, centres = make_windows(clean_df, PROXIES, window_size=50, stride=5)
        assert len(windows) > 0
        # window shape = (n_proxies, window_size)
        assert windows[0].shape == (len(PROXIES), 50)

    def test_centres_length_matches_windows(self, clean_df):
        windows, centres = make_windows(clean_df, PROXIES, window_size=50, stride=5)
        assert len(windows) == len(centres)

    def test_larger_stride_fewer_windows(self, clean_df):
        w1, _ = make_windows(clean_df, PROXIES, window_size=50, stride=5)
        w2, _ = make_windows(clean_df, PROXIES, window_size=50, stride=25)
        assert len(w1) > len(w2)

    def test_centres_within_data_range(self, clean_df):
        _, centres = make_windows(clean_df, PROXIES, window_size=50, stride=5)
        ages = clean_df["age_bp"].values
        assert np.all(centres >= ages.min())
        assert np.all(centres <= ages.max())


# ── correlation_adjacency ─────────────────────────────────────────────────────

class TestCorrelationAdjacency:
    """window shape = (n_nodes, T) as used internally by make_windows."""

    def test_output_shape(self):
        rng = np.random.default_rng(1)
        window = rng.normal(size=(4, 50))  # 4 proxies, 50 timesteps
        A = correlation_adjacency(window, threshold=0.3)
        assert A.shape == (4, 4)

    def test_no_self_loops(self):
        rng = np.random.default_rng(2)
        window = rng.normal(size=(4, 50))
        A = correlation_adjacency(window, threshold=0.3)
        assert np.all(np.diag(A) == 0)

    def test_symmetric(self):
        rng = np.random.default_rng(3)
        window = rng.normal(size=(4, 50))
        A = correlation_adjacency(window, threshold=0.3)
        np.testing.assert_array_equal(A, A.T)

    def test_higher_threshold_fewer_edges(self):
        rng = np.random.default_rng(4)
        window = rng.normal(size=(4, 100))
        A_low  = correlation_adjacency(window, threshold=0.1)
        A_high = correlation_adjacency(window, threshold=0.8)
        assert A_low.sum() >= A_high.sum()

    def test_correlated_series_has_edges(self):
        """Strongly correlated series should produce edges above threshold."""
        t = np.linspace(0, 4 * np.pi, 100)
        # 4 proxies all correlated (sin + tiny noise)
        rng = np.random.default_rng(99)
        window = np.stack([np.sin(t) + rng.normal(0, 0.01, len(t)) for _ in range(4)])
        A = correlation_adjacency(window, threshold=0.5)
        # Diagonal is zero, off-diagonal should have at least some edges
        off_diag = A[~np.eye(4, dtype=bool)]
        assert off_diag.sum() > 0, "Expected correlated proxies to form edges"


# ── graph_diffuse ─────────────────────────────────────────────────────────────

class TestGraphDiffuse:
    def test_shape_preserved(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(4, 50))
        A = correlation_adjacency(X, threshold=0.3)
        X_diff = graph_diffuse(X, A)
        assert X_diff.shape == X.shape

    def test_zero_adjacency_returns_X(self):
        """With no edges, (I + 0) @ X = X (or similar identity behaviour)."""
        rng = np.random.default_rng(6)
        X = rng.normal(size=(4, 50))
        A = np.zeros((4, 4))
        X_diff = graph_diffuse(X, A)
        # Should equal X (identity diffusion)
        np.testing.assert_allclose(X_diff, X, atol=1e-10)

    def test_connected_graph_changes_values(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(4, 50))
        A = np.ones((4, 4)) - np.eye(4)  # fully connected, no self-loops
        X_diff = graph_diffuse(X, A)
        assert not np.allclose(X_diff, X)


# ── reconstruction_error ──────────────────────────────────────────────────────

class TestReconstructionError:
    def test_full_components_near_zero(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(4, 50))
        err = reconstruction_error(X, n_components=4)
        assert err < 1e-6

    def test_fewer_components_higher_error(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(4, 50))
        e1 = reconstruction_error(X, n_components=1)
        e3 = reconstruction_error(X, n_components=3)
        assert e1 >= e3

    def test_non_negative(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(4, 50))
        err = reconstruction_error(X, n_components=2)
        assert err >= 0.0


# ── fit_and_score (full pipeline) ─────────────────────────────────────────────

class TestFitAndScore:
    def test_returns_dataframe(self, clean_df):
        windows, centres = make_windows(clean_df, PROXIES, window_size=50, stride=10)
        result = fit_and_score(windows, centres, corr_threshold=0.3, n_components=2)
        assert hasattr(result, "columns")
        assert "age_bp" in result.columns
        assert "anomaly_score" in result.columns

    def test_score_count_equals_window_count(self, clean_df):
        windows, centres = make_windows(clean_df, PROXIES, window_size=50, stride=10)
        result = fit_and_score(windows, centres, corr_threshold=0.3, n_components=2)
        assert len(result) == len(windows)

    def test_all_scores_non_negative(self, clean_df):
        windows, centres = make_windows(clean_df, PROXIES, window_size=50, stride=10)
        result = fit_and_score(windows, centres, corr_threshold=0.3, n_components=2)
        assert (result["anomaly_score"] >= 0).all()

    def test_laschamp_in_top_anomalies(self, clean_df):
        """Embedded Laschamp signal must rank in top 5% of windows."""
        windows, centres = make_windows(clean_df, PROXIES, window_size=50, stride=5)
        result = fit_and_score(windows, centres, corr_threshold=0.3, n_components=2)
        p95  = result["anomaly_score"].quantile(0.95)
        top  = result[result["anomaly_score"] >= p95]
        near = np.any(np.abs(top["age_bp"].values - 41_000) < 10_000)  # Laschamp excursion spans ~4 ka; ±10 ka tolerance
        assert near, (
            f"Laschamp not detected in top 5%.\n"
            f"Top ages: {sorted(top['age_bp'].values)}"
        )
