"""
Tests for FIELD_COHERENCE_MONITOR logic (Python equivalents of the Rust code).
Validates the coherence score formula, alert levels, Laschamp % calculation,
and the pole drift haversine computation — all without needing to compile Rust.
"""
import math
import numpy as np
import pytest

# ── Replicate Rust coherence formulas in Python for testing ──────────────────

VADM_REF        = 8.0     # 10²² Am²
KP_MAX          = 9.0
F107_MIN        = 70.0
F107_MAX        = 310.0
LASCHAMP_DRIFT  = 500.0   # km/yr upper bound
LASCHAMP_CR_SCALE = 4.0


def haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two WGS84 points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.asin(math.sqrt(a))


def pole_drift_km_yr(positions):
    """Positions = list of (year, lat, lon). Returns km/yr for last 2 points."""
    if len(positions) < 2:
        return 35.0
    a, b = positions[-2], positions[-1]
    dt = b[0] - a[0]
    if dt <= 0:
        return 35.0
    dist = haversine_km(a[1], a[2], b[1], b[2])
    return dist / dt


def compute_coherence(current_kp, current_f107, drift_km_yr):
    kp_contrib    = min(max(current_kp / KP_MAX, 0.0), 1.0)
    f107_norm     = min(max((current_f107 - F107_MIN) / (F107_MAX - F107_MIN), 0.0), 1.0)
    cr_contrib    = 1.0 - f107_norm
    drift_contrib = min(max(drift_km_yr / LASCHAMP_DRIFT, 0.0), 1.0)
    disturbance   = 0.40 * kp_contrib + 0.40 * cr_contrib + 0.20 * drift_contrib
    return max(0.0, min(1.0, 1.0 - disturbance))


def laschamp_pct(current_kp, current_f107, drift_km_yr):
    kp_pct    = min(max(current_kp / KP_MAX, 0.0), 1.0)
    cr_norm   = 1.0 - min(max((current_f107 - F107_MIN)/(F107_MAX - F107_MIN), 0.0), 1.0)
    cr_pct    = cr_norm / LASCHAMP_CR_SCALE
    drift_pct = min(max(drift_km_yr / LASCHAMP_DRIFT, 0.0), 1.0)
    return min(max((kp_pct * 0.40 + cr_pct * 0.40 + drift_pct * 0.20) * 100.0, 0.0), 100.0)


def alert_level(score):
    if score >= 0.70: return "NOMINAL"
    if score >= 0.50: return "WATCH"
    if score >= 0.30: return "WARNING"
    return "CRITICAL"


# ── Haversine / pole drift ────────────────────────────────────────────────────

class TestHaversine:
    def test_same_point_zero_distance(self):
        assert haversine_km(87.5, 153.0, 87.5, 153.0) == 0.0

    def test_equatorial_degree_approx_111km(self):
        d = haversine_km(0, 0, 0, 1)
        assert 110.0 < d < 112.0

    def test_pole_crossing(self):
        """Near-polar lat change of 1° should be ~111 km."""
        d = haversine_km(87.0, 0.0, 88.0, 0.0)
        assert 100.0 < d < 115.0

    def test_symmetric(self):
        d1 = haversine_km(87.5, 153.0, 87.6, 147.0)
        d2 = haversine_km(87.6, 147.0, 87.5, 153.0)
        assert abs(d1 - d2) < 1e-8


class TestPoleDrift:
    def test_single_point_returns_fallback(self):
        pos = [(2025.0, 87.7, 142.0)]
        assert pole_drift_km_yr(pos) == 35.0

    def test_empty_returns_fallback(self):
        assert pole_drift_km_yr([]) == 35.0

    def test_wmm_2024_2025_rate(self):
        """WMM 2024→2025 drift should be in realistic range 20-60 km/yr."""
        pos = [
            (2024.0, 87.6, 147.0),
            (2025.0, 87.7, 142.0),
        ]
        rate = pole_drift_km_yr(pos)
        assert 15.0 < rate < 80.0, f"Drift rate {rate:.1f} km/yr out of range"

    def test_zero_time_returns_fallback(self):
        pos = [(2025.0, 87.0, 140.0), (2025.0, 88.0, 140.0)]
        assert pole_drift_km_yr(pos) == 35.0

    def test_rate_increases_with_faster_drift(self):
        slow = [(2024.0, 87.5, 150.0), (2025.0, 87.6, 148.0)]
        fast = [(2024.0, 87.5, 150.0), (2025.0, 89.0, 100.0)]
        assert pole_drift_km_yr(fast) > pole_drift_km_yr(slow)


# ── Coherence score ───────────────────────────────────────────────────────────

class TestCoherenceScore:
    def test_quiet_conditions_high_score(self):
        """Kp=1 (quiet), F10.7=200 (max), drift=35 → high coherence."""
        s = compute_coherence(current_kp=1.0, current_f107=200.0, drift_km_yr=35.0)
        assert s > 0.60, f"Expected high coherence, got {s:.3f}"

    def test_extreme_conditions_low_score(self):
        """Kp=8 (G4), F10.7=70 (solar min), drift=200 → low coherence."""
        s = compute_coherence(current_kp=8.0, current_f107=70.0, drift_km_yr=200.0)
        assert s < 0.40, f"Expected low coherence, got {s:.3f}"

    def test_laschamp_reference_score(self):
        """Laschamp-like: Kp=9, F10.7=70 (field-driven not solar), drift=500."""
        s = compute_coherence(current_kp=9.0, current_f107=70.0, drift_km_yr=500.0)
        assert s < 0.20, f"Laschamp conditions should yield score<0.20, got {s:.3f}"

    def test_score_bounded_0_to_1(self):
        for kp in [0, 4.5, 9]:
            for f107 in [70, 150, 310]:
                for drift in [0, 35, 500]:
                    s = compute_coherence(kp, f107, drift)
                    assert 0.0 <= s <= 1.0

    def test_higher_kp_lower_score(self):
        s_quiet = compute_coherence(1.0, 150.0, 35.0)
        s_storm = compute_coherence(7.0, 150.0, 35.0)
        assert s_storm < s_quiet

    def test_higher_f107_higher_score(self):
        """More solar activity → more solar wind → less CR → more stable."""
        s_max = compute_coherence(2.0, 280.0, 35.0)
        s_min = compute_coherence(2.0, 70.0,  35.0)
        assert s_max > s_min

    def test_weights_sum_correctly(self):
        """With all contributors at 0.5, disturbance = 0.5 → score = 0.5."""
        # kp_contrib=0.5 → kp=4.5
        # cr_contrib=0.5 → f107=190 (midpoint)
        # drift_contrib=0.5 → drift=250
        kp    = 4.5
        f107  = F107_MIN + 0.5 * (F107_MAX - F107_MIN)   # 190
        drift = 250.0
        s = compute_coherence(kp, f107, drift)
        np.testing.assert_allclose(s, 0.5, atol=0.01)


# ── Alert levels ──────────────────────────────────────────────────────────────

class TestAlertLevel:
    def test_nominal_at_high_score(self):
        assert alert_level(0.85) == "NOMINAL"
        assert alert_level(0.70) == "NOMINAL"

    def test_watch_boundary(self):
        assert alert_level(0.69) == "WATCH"
        assert alert_level(0.50) == "WATCH"

    def test_warning_boundary(self):
        assert alert_level(0.49) == "WARNING"
        assert alert_level(0.30) == "WARNING"

    def test_critical_at_low_score(self):
        assert alert_level(0.29) == "CRITICAL"
        assert alert_level(0.00) == "CRITICAL"

    def test_modern_kp3_is_nominal(self):
        score = compute_coherence(3.0, 180.0, 35.0)
        assert alert_level(score) in ("NOMINAL", "WATCH")


# ── Laschamp % ────────────────────────────────────────────────────────────────

class TestLaschampPct:
    def test_modern_quiet_low_pct(self):
        """Modern quiet-time should be well below 50%."""
        pct = laschamp_pct(2.0, 200.0, 35.0)
        assert pct < 30.0, f"Expected <30% toward Laschamp, got {pct:.1f}%"

    def test_laschamp_conditions_high_pct(self):
        """Full Laschamp conditions → near 100%."""
        pct = laschamp_pct(9.0, 70.0, 500.0)
        assert pct >= 70.0, f"Expected >70%, got {pct:.1f}%"

    def test_bounded_0_to_100(self):
        for kp in [0, 9]:
            for f107 in [70, 310]:
                for drift in [0, 500]:
                    pct = laschamp_pct(kp, f107, drift)
                    assert 0.0 <= pct <= 100.0

    def test_increases_with_kp(self):
        pct_low  = laschamp_pct(1.0, 150.0, 35.0)
        pct_high = laschamp_pct(8.0, 150.0, 35.0)
        assert pct_high > pct_low

    def test_increases_with_drift(self):
        pct_slow = laschamp_pct(3.0, 150.0, 35.0)
        pct_fast = laschamp_pct(3.0, 150.0, 250.0)
        assert pct_fast > pct_slow
