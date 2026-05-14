"""seismic_layer.py — SUBSTRATE Seismic Layer v1.0
Measures global seismic activity over the last 24h via USGS Earthquake API.
Local-first: caches data, works offline with stale cache/synthetic model.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

USGS_API = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
CACHE_TTL_SECONDS = 3 * 3600  # 3 hours
CACHE_DIR = Path.home() / ".cache" / "substrate" / "seismic"
CACHE_FILE = CACHE_DIR / "last_result.json"


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _load_cache() -> dict[str, Any] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_cache(data: dict[str, Any]) -> None:
    _ensure_cache_dir()
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as exc:
        logger.warning(f"Seismic: could not save cache — {exc}")


def run(params: dict[str, Any] = None) -> dict[str, Any]:
    cached = _load_cache()
    fresh = False
    if cached is not None and CACHE_FILE.exists():
        age = time.time() - CACHE_FILE.stat().st_mtime
        if age < CACHE_TTL_SECONDS:
            fresh = True

    data_payload: dict[str, Any] | None = None
    source_used = "USGS Earthquake API"

    if not fresh:
        try:
            import urllib.request
            import ssl

            ctx = ssl.create_default_context()
            with urllib.request.urlopen(USGS_API, context=ctx, timeout=10) as resp:
                raw_data = json.loads(resp.read().decode("utf-8"))

            events = raw_data.get("features", [])
            magnitudes = [
                float(e["properties"]["mag"])
                for e in events
                if e.get("properties") and e["properties"].get("mag") is not None
            ]
            
            data_payload = {
                "magnitudes": magnitudes,
                "timestamp": datetime.utcnow().isoformat(),
            }
            _save_cache(data_payload)
            logger.info(
                f"Seismic: USGS live — {len(magnitudes)} events, "
                f"max_mag={max(magnitudes, default=0):.2f}, "
                f"M5+={sum(1 for m in magnitudes if m >= 5.0)}"
            )
        except Exception as exc:
            logger.warning(f"Seismic: USGS fetch failed ({exc}), falling back to cache/synthetic")

    if data_payload is None:
        if cached is not None:
            data_payload = cached
            source_used = "USGS Earthquake API (cached)"
        else:
            # Plausible synthetic distribution for a normal active day
            rng = np.random.default_rng() if "np" in globals() else None
            # Let's import numpy inside block or use standard random
            import random
            magnitudes = [random.uniform(2.5, 4.5) for _ in range(45)]
            magnitudes += [random.uniform(4.5, 5.8) for _ in range(5)]
            data_payload = {
                "magnitudes": magnitudes,
                "timestamp": datetime.utcnow().isoformat(),
            }
            source_used = "Seismic synthetic fallback"

    magnitudes = data_payload.get("magnitudes", [])
    energies = [10.0 ** (1.5 * m) for m in magnitudes]

    total_energy = sum(energies)
    max_mag = max(magnitudes) if magnitudes else 0.0
    event_count = len(magnitudes)

    # Score: normalized against an active day threshold (e.g. energy of an M6.5)
    energy_threshold = 10.0 ** (1.5 * 6.5)
    score = float(min(total_energy / energy_threshold, 1.0))

    return {
        "layer": "seismic",
        "score": round(score, 4),
        "data": {
            "source": source_used,
            "events_24h": event_count,
            "max_magnitude": round(max_mag, 2),
            "total_energy_relative": round(total_energy / energy_threshold, 4),
            "significant_events": sum(1 for m in magnitudes if m >= 5.0),
            "last_update": data_payload.get("timestamp", datetime.utcnow().isoformat()),
        },
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(run())
