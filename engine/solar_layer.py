"""
solar_layer.py — SUBSTRATE Solar Layer v2.0
Real-time solar activity analysis via NOAA SWPC public feeds.
Local-first: caches data, works offline with stale cache.

Target: Arch Linux · Python 3.10+ · No external APIs except NOAA (public)
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

NOAA_CYCLE_URL = (
    "https://services.swpc.noaa.gov/json/solar-cycle/"
    "observed-solar-cycle-indices.json"
)
NOAA_KP_URL = (
    "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
)
CACHE_TTL_SECONDS = 6 * 3600  # 6 hours
CACHE_DIR = Path.home() / ".cache" / "substrate" / "solar"

# Historical maxima for normalization (Solar Cycle 24/25 context)
MAX_SSN = 250.0          # ~max smoothed sunspot number
MAX_F10_7 = 300.0        # ~max F10.7 flux sfu
MAX_AP = 100.0           # ~max Ap index

# ─────────────────────────────────────────────────────────────
# CACHE HELPERS
# ─────────────────────────────────────────────────────────────


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_path(name: str) -> Path:
    _ensure_cache_dir()
    return CACHE_DIR / f"{name}.json"


def _is_fresh(path: Path, ttl: int = CACHE_TTL_SECONDS) -> bool:
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age < ttl


def _load_cache(name: str) -> Optional[List[Dict[str, Any]]]:
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(name: str, data: List[Dict[str, Any]]) -> None:
    path = _cache_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────
# FETCH (with fallback to cache)
# ─────────────────────────────────────────────────────────────


def _fetch_noaa_cycle() -> List[Dict[str, Any]]:
    """Fetch observed solar cycle indices (SSN + F10.7)."""
    # Check cache freshness first — avoid unnecessary network calls
    cached = _load_cache("cycle")
    if cached is not None and _is_fresh(_cache_path("cycle")):
        return cached

    # Try network
    try:
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(NOAA_CYCLE_URL, context=ctx, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        _save_cache("cycle", raw)
        return raw
    except Exception:
        # Fallback to cache (even if stale — offline mode)
        if cached is not None:
            return cached
        return []


def _fetch_noaa_kp() -> List[Dict[str, Any]]:
    """Fetch planetary K/Ap index (last ~30 days)."""
    cached = _load_cache("kp")
    if cached is not None and _is_fresh(_cache_path("kp")):
        return cached

    try:
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(NOAA_KP_URL, context=ctx, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        _save_cache("kp", raw)
        return raw
    except Exception:
        if cached is not None:
            return cached
        return []


# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────


def _parse_cycle_data(
    records: List[Dict[str, Any]],
) -> Tuple[float, float, List[float]]:
    """
    Extract latest SSN, F10.7, and build history for trend.
    Returns: (latest_ssn, latest_f10_7, ssn_history)
    """
    if not records:
        return 0.0, 0.0, []

    # NOAA cycle data: list of dicts with keys like "time-tag", "ssn", "f10.7"
    records_sorted = sorted(
        records,
        key=lambda r: r.get("time-tag", ""),
    )

    latest = records_sorted[-1]
    ssn = float(latest.get("ssn", 0.0) or 0.0)
    f10_7 = float(latest.get("f10.7", 0.0) or 0.0)

    # Build SSN history for trend
    ssn_history = [
        float(r.get("ssn", 0.0) or 0.0)
        for r in records_sorted
        if r.get("ssn") is not None
    ]

    return ssn, f10_7, ssn_history


def _parse_kp_data(records: List[Dict[str, Any]]) -> float:
    """Extract latest Ap from Kp feed."""
    if not records:
        return 0.0
    latest = max(records, key=lambda r: r.get("time_tag", ""))
    return float(latest.get("ap", 0.0) or 0.0)


def _compute_trend(history: List[float], window: int = 12) -> float:
    """
    Simple linear regression slope over last `window` points.
    Returns slope normalized by mean (trend direction & strength).
    """
    if len(history) < 2:
        return 0.0
    y = np.array(history[-window:], dtype=np.float64)
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=np.float64)
    # Linear regression: y = a*x + b
    A = np.vstack([x, np.ones(len(x))]).T
    result = np.linalg.lstsq(A, y, rcond=None)
    a = result[0][0]
    mean_y = np.mean(y)
    if mean_y == 0:
        return 0.0
    return float(a / mean_y)  # normalized slope


def _estimate_days_to_max(
    history: List[float],
    current_ssn: float,
) -> Optional[float]:
    """
    Rough heuristic: if we are in rising phase and below historical max,
    estimate months to maximum using smoothed derivative.
    Returns None if declining or insufficient data.
    """
    if len(history) < 6:
        return None
    recent = np.array(history[-6:], dtype=np.float64)
    if len(recent) < 2:
        return None
    # Check if rising
    if recent[-1] <= recent[0]:
        return None  # declining or flat
    # Very rough: if rising at rate `rate` per month, time to MAX_SSN
    rate = (recent[-1] - recent[0]) / len(recent)
    if rate <= 0:
        return None
    remaining = MAX_SSN - current_ssn
    if remaining <= 0:
        return 0.0
    months = remaining / rate
    return max(0.0, months * 30.0)  # convert to days


def _normalize_score(ssn: float, f10_7: float, ap: float) -> float:
    """
    Composite score [0,1] representing solar activity intensity.
    Weighted: 50% SSN, 30% F10.7, 20% Ap.
    """
    ssn_norm = min(ssn / MAX_SSN, 1.0) if MAX_SSN > 0 else 0.0
    f10_norm = min(f10_7 / MAX_F10_7, 1.0) if MAX_F10_7 > 0 else 0.0
    ap_norm = min(ap / MAX_AP, 1.0) if MAX_AP > 0 else 0.0
    score = 0.5 * ssn_norm + 0.3 * f10_norm + 0.2 * ap_norm
    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────


def run(params: dict = None) -> dict:
    """
    SUBSTRATE layer entrypoint.
    Returns standard LayerResult dict with score ∈ [0, 1].
    """
    try:
        from dll_healing import heal
        heal()
    except ImportError:
        pass

    t0 = time.perf_counter()

    try:
        # Fetch
        cycle_data = _fetch_noaa_cycle()
        kp_data = _fetch_noaa_kp()

        # Parse
        ssn, f10_7, ssn_history = _parse_cycle_data(cycle_data)
        ap = _parse_kp_data(kp_data)

        # Analysis
        score = _normalize_score(ssn, f10_7, ap)
        trend = _compute_trend(ssn_history)
        days_to_max = _estimate_days_to_max(ssn_history, ssn)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Latest time-tag from data
        latest_tag = ""
        if cycle_data:
            sorted_data = sorted(cycle_data, key=lambda r: r.get("time-tag", ""))
            latest_tag = str(sorted_data[-1].get("time-tag", "unknown"))

        return {
            "layer": "solar",
            "score": round(score, 4),
            "data": {
                "source": "NOAA SWPC",
                "last_updated": latest_tag,
                "ssn": round(ssn, 2),
                "f10_7": round(f10_7, 2),
                "ap": round(ap, 2),
                "trend": round(trend, 4),
                "days_to_max_estimated": round(days_to_max, 1) if days_to_max is not None else None,
                "cache_mode": "live" if _is_fresh(_cache_path("cycle")) else "stale",
                "latency_ms": round(latency_ms, 2),
                "synthetic": False,
            },
        }
    except Exception as exc:
        # Fallback: sinusoidal model (same as v1.0)
        from datetime import timezone
        now = datetime.now(tz=timezone.utc)
        year_frac = now.year + (now.timetuple().tm_yday - 1) / 365.25
        cycle_phase = ((year_frac - 2019.96) % 11.0) / 11.0
        solar_index = (1.0 - math.cos(2 * math.pi * cycle_phase)) / 2.0
        score = round(solar_index, 6)
        return {
            "layer": "solar",
            "score": score,
            "data": {
                "cycle_phase": round(cycle_phase, 4),
                "solar_flux_idx": score,
                "year_frac": round(year_frac, 4),
                "model": "11yr_sinusoid_fallback",
                "synthetic": True,
                "error": str(exc),
            },
        }


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint
    print("=== SUBSTRATE Solar Layer v2.0 ===")
    print("Fetching NOAA SWPC data...")
    out = run()
    pprint.pprint(out)
