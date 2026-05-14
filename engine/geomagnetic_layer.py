"""
geomagnetic_layer.py — SUBSTRATE Geomagnetic Layer v2.0

Reads bundled IGRF data from SUBSTRATE/data/geomagnetic/ AND
fetches live NOAA SWPC feeds (Kp, Dst, Alerts, Scales).

Strategy (3-level fallback):
  1. NOAA Live (Kp + Dst + Alerts) blended 60/40 with RandomForest
  2. RandomForest only (cached probe_state.json or live fingerprint)
  3. Synthetic heuristic (last resort)

If NOAA issues alert G3+ (Kp≥7), score is forced ≥0.7 regardless of model.

Target: Arch Linux · Python 3.10+ · Local-first · No API key
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────────
# PATH SETUP (preserve existing module resolution)
# ─────────────────────────────────────────────────────────────

_ROOT        = Path(__file__).resolve().parent.parent          # SUBSTRATE root
_GEO_DATA    = _ROOT / "data" / "geomagnetic"
_PROBE_STATE = _GEO_DATA / "probe_state.json"
_CYCLE_SRC   = _ROOT / "modules" / "cycle_project" / "src"    # forward_probe module
_STALE_AFTER = timedelta(hours=24)

if _CYCLE_SRC.is_dir():
    sys.path.insert(0, str(_CYCLE_SRC))

# ─────────────────────────────────────────────────────────────
# NOAA CONFIG
# ─────────────────────────────────────────────────────────────

NOAA_KP_1M_URL = (
    "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"
)
NOAA_DST_URL = (
    "https://services.swpc.noaa.gov/products/kyoto-dst.json"
)
NOAA_ALERTS_URL = (
    "https://services.swpc.noaa.gov/products/alerts.json"
)
NOAA_SCALES_URL = (
    "https://services.swpc.noaa.gov/products/noaa-scales.json"
)

CACHE_TTL_SECONDS = 3600  # 1 hour (geomagnetic changes faster than solar)
CACHE_DIR = Path.home() / ".cache" / "substrate" / "geomagnetic"


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


def _load_cache(name: str) -> Any | None:
    path = _cache_path(name)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(name: str, data: Any) -> None:
    path = _cache_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────
# NOAA FETCHERS
# ─────────────────────────────────────────────────────────────

def _fetch_json(url: str, cache_name: str) -> Any | None:
    """Fetch JSON from NOAA with cache-first strategy."""
    cached = _load_cache(cache_name)
    if cached is not None and _is_fresh(_cache_path(cache_name)):
        return cached

    try:
        import urllib.request
        import ssl
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(url, context=ctx, timeout=15) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
        _save_cache(cache_name, raw)
        return raw
    except Exception:
        return cached  # stale cache or None


def _fetch_kp() -> Tuple[float, str]:
    """Return (latest_estimated_kp, time_tag)."""
    data = _fetch_json(NOAA_KP_1M_URL, "kp_1m")
    if not data or not isinstance(data, list):
        return 0.0, ""
    latest = max(data, key=lambda r: r.get("time_tag", ""))
    kp = float(latest.get("estimated_kp", latest.get("kp_index", 0.0)) or 0.0)
    tag = str(latest.get("time_tag", ""))
    return kp, tag


def _fetch_dst() -> Tuple[float, str]:
    """Return (latest_dst, time_tag)."""
    data = _fetch_json(NOAA_DST_URL, "dst")
    if not data or not isinstance(data, list):
        return 0.0, ""
    # Post-March 2026 format: list of dicts
    if isinstance(data[0], dict):
        latest = max(data, key=lambda r: r.get("time_tag", ""))
        return float(latest.get("dst", 0.0) or 0.0), str(latest.get("time_tag", ""))
    # Legacy format: list of [time_tag, dst]
    if isinstance(data[0], list) and len(data[0]) > 1:
        latest = max(data, key=lambda r: r[0] if isinstance(r, list) else "")
        return float(latest[1]), str(latest[0])
    return 0.0, ""


def _fetch_alerts() -> List[Dict[str, Any]]:
    """Fetch active geomagnetic alerts (ALTK / WARK / WATA codes)."""
    data = _fetch_json(NOAA_ALERTS_URL, "alerts")
    if not data or not isinstance(data, list):
        return []
    geo_alerts = []
    for alert in data:
        code = str(alert.get("product_id", alert.get("code", ""))).upper()
        if code.startswith(("ALTK", "WARK", "WATA")):
            geo_alerts.append(alert)
    return geo_alerts


def _fetch_g_scale() -> Tuple[int, str]:
    """Fetch current NOAA G-scale from space weather scales."""
    data = _fetch_json(NOAA_SCALES_URL, "scales")
    if not data or not isinstance(data, dict):
        return 0, "none"
    current = data.get("0", {})
    g_info = current.get("G", {})
    scale = int(g_info.get("Scale", 0) or 0)
    text = str(g_info.get("Text", "none"))
    return scale, text


# ─────────────────────────────────────────────────────────────
# EXISTING RANDOMFOREST PIPELINE (preserved from v1.0)
# ─────────────────────────────────────────────────────────────

def _load_cached_probe() -> dict | None:
    """Return probe_state.json contents if the file exists and is < 24 h old."""
    if not _PROBE_STATE.exists():
        return None
    try:
        with _PROBE_STATE.open() as fh:
            state = json.load(fh)
        ts = state.get("generated_at", "")
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - dt < _STALE_AFTER:
            return state
    except Exception:
        pass
    return None


def _run_fingerprint_live() -> dict:
    """Run the RandomForest fingerprint classifier directly (no LSTM training)."""
    from forward_probe.fingerprint import run_fingerprint  # type: ignore[import]

    fp    = run_fingerprint(
        parquet_path=str(_GEO_DATA / "aligned.parquet"),
        output_dir=str(_GEO_DATA),
    )
    prob  = float(fp.get("probability", 0.5))
    state = {
        "pre_excursion_prob":    round(prob, 6),
        "status":                fp.get("status", "STABLE"),
        "loo_accuracy":          float(fp.get("loo_accuracy", float("nan"))),
        "n_excursions":          int(fp.get("n_excursions", 0)),
        "generated_at":          datetime.now(timezone.utc).isoformat(),
    }
    return state


def _get_rf_score() -> Tuple[float, dict]:
    """
    Get RandomForest score via cache or live run.
    Returns (score, metadata_dict).
    """
    # Fast path: cached probe_state.json
    probe = _load_cached_probe()
    if probe is not None:
        score = float(probe.get("pre_excursion_prob", 0.5))
        return score, {**probe, "method": "probe_state_cache"}

    # Slow path: run fingerprint live (~10s)
    try:
        probe = _run_fingerprint_live()
        score = float(probe.get("pre_excursion_prob", 0.5))
        return score, {**probe, "method": "fingerprint_live"}
    except Exception as exc:
        return -1.0, {"error": str(exc), "method": "rf_failed"}


# ─────────────────────────────────────────────────────────────
# NOAA SCORE CALCULATION
# ─────────────────────────────────────────────────────────────

def _kp_to_g_scale(kp: float) -> int:
    """Map estimated Kp to NOAA G-scale (0-5)."""
    if kp < 5:   return 0
    elif kp < 6: return 1
    elif kp < 7: return 2
    elif kp < 8: return 3
    elif kp < 9: return 4
    else:        return 5


def _compute_noaa_score(kp: float, dst: float, g_scale: int, alerts: list) -> float:
    """
    Compute geomagnetic threat score [0,1] from NOAA live data.
    Kp normalized against 9, Dst against -300, G-scale against 5.
    Alert override: G3+ → minimum 0.7, G5 → 1.0.
    """
    kp_norm  = min(kp / 9.0, 1.0)
    dst_norm = min(abs(dst) / 300.0, 1.0)
    g_norm   = g_scale / 5.0

    score = 0.4 * kp_norm + 0.3 * dst_norm + 0.3 * g_norm

    # G-scale override (NOAA alerts take priority)
    if g_scale >= 5: score = 1.0
    elif g_scale >= 4: score = max(score, 0.85)
    elif g_scale >= 3: score = max(score, 0.7)

    return float(np.clip(score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run(params: dict = None) -> dict:
    """
    SUBSTRATE layer entrypoint.

    Strategy:
      1. Fetch NOAA live feeds (Kp, Dst, Alerts, Scales)
      2. Run existing RandomForest pipeline (cache or live)
      3. Blend: 60% NOAA + 40% RF (if both available)
      4. If G-scale ≥ 3 (severe storm), NOAA overrides 100%
      5. If NOAA unavailable, RF-only
      6. If both fail, synthetic fallback
    """
    try:
        from dll_healing import heal
        heal()
    except Exception:
        pass

    t0 = time.perf_counter()

    # ── Step 1: Fetch NOAA data ──────────────────────────────
    kp, kp_tag     = _fetch_kp()
    dst, dst_tag   = _fetch_dst()
    alerts         = _fetch_alerts()
    g_scale_noaa, g_text = _fetch_g_scale()

    # Use max of Kp-derived and NOAA-reported G-scale
    g_scale_kp = _kp_to_g_scale(kp)
    g_scale    = max(g_scale_noaa, g_scale_kp)

    has_noaa = kp > 0 or dst != 0.0 or len(alerts) > 0

    # Cache mode determination
    cache_mode = "none"
    kp_cache = _cache_path("kp_1m")
    if _is_fresh(kp_cache):
        cache_mode = "live"
    elif kp_cache.exists():
        cache_mode = "stale"

    # ── Step 2: Get RF score ─────────────────────────────────
    rf_score, rf_meta = _get_rf_score()
    has_rf = rf_score >= 0.0

    # ── Step 3: Blend ────────────────────────────────────────
    if has_noaa:
        noaa_score = _compute_noaa_score(kp, dst, g_scale, alerts)

        if g_scale >= 3:
            # Severe storm: NOAA overrides completely
            final_score = noaa_score
            blend_method = "noaa_override_g3+"
        elif has_rf:
            # Normal: blend 60% NOAA + 40% RF
            final_score = 0.6 * noaa_score + 0.4 * rf_score
            blend_method = "noaa60_rf40"
        else:
            # NOAA only
            final_score = noaa_score
            blend_method = "noaa_only"
    elif has_rf:
        # NOAA unavailable, RF only
        final_score = rf_score
        blend_method = "rf_only"
    else:
        # Both failed: synthetic fallback
        rng = np.random.default_rng()
        final_score = float(rng.uniform(0.20, 0.75))
        blend_method = "synthetic_fallback"

    latency_ms = (time.perf_counter() - t0) * 1000.0

    # ── Build result ─────────────────────────────────────────
    alert_level = f"G{g_scale}"

    data: dict = {
        "source":        "NOAA SWPC + IGRF/RandomForest",
        "blend_method":  blend_method,
        "synthetic":     blend_method == "synthetic_fallback",
        "cache_mode":    cache_mode,
        # NOAA live metrics
        "kp":            round(kp, 2),
        "kp_time":       kp_tag,
        "dst":           round(dst, 1),
        "dst_time":      dst_tag,
        "g_scale":       g_scale,
        "alert_level":   alert_level,
        "alert_active":  len(alerts) > 0,
        "alert_count":   len(alerts),
        "g_text":        g_text,
        # RF metrics (if available)
        "rf_method":     rf_meta.get("method", "none"),
        "latency_ms":    round(latency_ms, 2),
    }

    # Inject RF-specific fields if available
    if has_rf:
        data["pre_excursion_prob"] = round(rf_score, 6)
        data["rf_status"] = rf_meta.get("status", "unknown")
        if "loo_accuracy" in rf_meta:
            data["loo_accuracy"] = rf_meta["loo_accuracy"]
        if "n_excursions" in rf_meta:
            data["n_excursions"] = rf_meta["n_excursions"]

    return {
        "layer": "geomagnetic",
        "score": round(float(np.clip(final_score, 0.0, 1.0)), 4),
        "data":  data,
    }


if __name__ == "__main__":
    import pprint
    print("=== SUBSTRATE Geomagnetic Layer v2.0 ===")
    print("Fetching NOAA SWPC + RandomForest data...")
    out = run()
    pprint.pprint(out)
