"""Solar wind upstream predictor -- DSCOVR L1 monitor via NOAA SWPC.

ORPHAN MODULE -- experiment-0-lindblad-calibration
===================================================
Status: NOT registered in the SUBSTRATE engine motor.
        NOT wired to GlobalMesh, FFI, or any layer correlation.

Dependency gate: this module connects to the motor ONLY after
Experiment 0 closes with |r(Dst, Y_s)| > 0.4, p < 0.05.

Physical role:
    DSCOVR sits at L1 Lagrange point, ~1.5e6 km sunward.
    Solar wind takes ~30-90 min to propagate from L1 to Earth's magnetopause.
    bz_gsm < 0 (southward IMF) drives dayside reconnection -> Dst depression.

    This layer provides a 30-90 min predictor of Dst perturbation *before*
    NOAA publishes the official Dst index. It does NOT replace geomagnetic_layer.
    It is an upstream input for cross-layer causal validation.

Causal chain (post-Experiment 0):
    DSCOVR bz_gsm < 0  ->  [30-90 min lag]  ->  Dst drops  ->
    DeltaBh perturbs b_earth  ->  magnon Y_s shifts (anisotropic T2 mechanism)

Burton predictor (simplified, no clock-angle):
    dst_estimate = -4.4 * sqrt(p_dyn) * b_z_south
    This is a 1-D approximation. Real Dst requires temporal integration
    with decay constant tau ~= 7.7h. Use dst_estimate as directional
    predictor only, not as a replacement for measured Dst.

Sources:
    Burton et al. (1975) JGR 80(31):4204
    NOAA SWPC DSCOVR product: https://www.swpc.noaa.gov/products/real-time-solar-wind
"""
from __future__ import annotations

import json
import math
import time
import urllib.request
import ssl
from pathlib import Path

# ── NOAA SWPC DSCOVR endpoints ─────────────────────────────────────────────────
_MAG_URL    = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
_PLASMA_URL = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"

# ── Cache ──────────────────────────────────────────────────────────────────────
_CACHE_DIR  = Path.home() / ".cache" / "substrate" / "solar_wind"
_CACHE_FILE = _CACHE_DIR / "dscovr.json"
_CACHE_TTL  = 300    # 5 minutes -- NOAA updates ~1 min but 5 min is sufficient
_CACHE_MAX  = 7200   # 2 hours stale cache accepted before declaring unavailable


def _ensure_cache():
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_age_s():
    if not _CACHE_FILE.exists():
        return float("inf")
    return time.time() - _CACHE_FILE.stat().st_mtime


def _load_cache():
    try:
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(data):
    _ensure_cache()
    _CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ── HTTP fetch ─────────────────────────────────────────────────────────────────

def _fetch_json(url, timeout=15):
    ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ── DSCOVR parser ──────────────────────────────────────────────────────────────

def _latest_row(rows):
    """Return the most recent non-null row from a NOAA time-series list."""
    if not rows or not isinstance(rows, list):
        return None
    # Skip header row if present (first element is a list of field names)
    data_rows = [r for r in rows if isinstance(r, (dict, list))]
    if not data_rows:
        return None

    # NOAA format is list-of-lists with a header, OR list-of-dicts
    if isinstance(data_rows[0], list):
        # list-of-lists: first row is header
        if isinstance(rows[0][0], str) and rows[0][0].lower() == "time_tag":
            header = rows[0]
            data_rows = rows[1:]
        else:
            header = None
        # Find last non-null row
        for row in reversed(data_rows):
            if any(v is not None and v != "" for v in row[1:]):
                if header:
                    return dict(zip(header, row))
                return {"time_tag": row[0], "_raw": row}
    elif isinstance(data_rows[0], dict):
        for row in reversed(data_rows):
            if row.get("time_tag"):
                return row
    return None


def _extract_mag(row):
    """Extract magnetometer fields; return None if critical values missing."""
    if row is None:
        return None
    try:
        bz  = row.get("bz_gsm")
        bt  = row.get("bt")
        bx  = row.get("bx_gsm")
        by  = row.get("by_gsm")
        tag = row.get("time_tag", "")
        if bz is None or bt is None:
            return None
        return {
            "time_tag": str(tag),
            "bx_gsm":   float(bz) if bx is None else float(bx),
            "by_gsm":   0.0       if by is None else float(by),
            "bz_gsm":   float(bz),
            "bt":       float(bt),
        }
    except (TypeError, ValueError):
        return None


def _extract_plasma(row):
    """Extract plasma fields; return None if critical values missing."""
    if row is None:
        return None
    try:
        density = row.get("density")
        speed   = row.get("speed")
        temp    = row.get("temperature")
        tag     = row.get("time_tag", "")
        if density is None or speed is None:
            return None
        return {
            "time_tag":    str(tag),
            "density":     float(density),    # cm^-3
            "speed":       float(speed),      # km/s
            "temperature": float(temp) if temp is not None else None,
        }
    except (TypeError, ValueError):
        return None


# ── Physics ────────────────────────────────────────────────────────────────────

def _compute_derived(mag, plasma):
    """Compute b_z_south, p_dyn, and Burton dst_estimate."""
    bz_gsm   = mag["bz_gsm"]
    bt       = mag["bt"]
    density  = plasma["density"]
    speed    = plasma["speed"]

    # Southward component only (reconnection driver)
    b_z_south = max(-bz_gsm, 0.0)

    # Dynamic pressure [nPa]: P = rho * v^2
    # rho = density [cm^-3] * mp [g] -> convert to nPa
    # P_dyn [nPa] = n [cm^-3] * 1.6726e-24 [g] * v^2 [km/s]^2 * 1e10 -> 1.6726e-6
    p_dyn = density * (speed ** 2) * 1.6726e-6   # nPa

    # Burton et al. simplified Dst predictor (directional, no decay term)
    # dst_estimate < 0 when bz_gsm < 0 (southward, storm-driving)
    dst_estimate = -4.4 * math.sqrt(max(p_dyn, 0.0)) * b_z_south

    # Clock angle (GSM XZ plane, degrees) -- useful diagnostic
    clock_angle = math.degrees(math.atan2(mag["by_gsm"], bz_gsm)) if bt > 0 else 0.0

    return {
        "b_z_south":    round(b_z_south, 4),
        "p_dyn_nPa":    round(p_dyn, 4),
        "dst_estimate": round(dst_estimate, 2),
        "clock_angle":  round(clock_angle, 1),
    }


# ── Score ──────────────────────────────────────────────────────────────────────

def _compute_score(derived, mag, plasma):
    """Storm potential score [0,1].

    Combines bz_south (strongest predictor) with p_dyn and speed.
    Does NOT replicate geomagnetic_layer scoring -- this is upstream potential,
    not a measured disturbance.
    """
    # bz_south normalized to 20 nT (G4-level sustained southward IMF)
    bz_score  = min(derived["b_z_south"] / 20.0, 1.0)
    # p_dyn normalized to 10 nPa (compression threshold)
    pdyn_score = min(derived["p_dyn_nPa"] / 10.0, 1.0)
    # speed normalized to 800 km/s (fast stream threshold)
    v_score   = min(plasma["speed"] / 800.0, 1.0)

    score = 0.60 * bz_score + 0.25 * pdyn_score + 0.15 * v_score
    return round(float(score), 6)


# ── Public entry point ─────────────────────────────────────────────────────────

def run(params=None):
    """SUBSTRATE layer entrypoint (ORPHAN -- not registered in motor).

    Fetch DSCOVR L1 solar wind data, compute Burton Dst predictor,
    and return structured result matching SUBSTRATE layer format.
    """
    import logging
    log = logging.getLogger("substrate.solar_wind")

    t0 = time.perf_counter()

    # ── Try live DSCOVR ────────────────────────────────────────────────────────
    mag_data    = None
    plasma_data = None
    source_tag  = "unknown"
    age_s       = _cache_age_s()

    if age_s > _CACHE_TTL:
        try:
            mag_rows    = _fetch_json(_MAG_URL)
            plasma_rows = _fetch_json(_PLASMA_URL)
            mag_data    = _extract_mag(_latest_row(mag_rows))
            plasma_data = _extract_plasma(_latest_row(plasma_rows))

            if mag_data and plasma_data:
                source_tag = "DSCOVR_live"
                log.info(
                    "DSCOVR live: bz_gsm=%.2f nT  speed=%.0f km/s  density=%.2f cm^-3",
                    mag_data["bz_gsm"], plasma_data["speed"], plasma_data["density"],
                )
            else:
                log.warning("DSCOVR fetch succeeded but data extraction failed.")
        except Exception as exc:
            log.warning("DSCOVR fetch failed: %s", exc)

    # ── Fall back to cache ─────────────────────────────────────────────────────
    if mag_data is None or plasma_data is None:
        cached = _load_cache()
        if cached and isinstance(cached, dict) and "latest" in cached:
            lat = cached["latest"]
            # Reconstruct mag_data and plasma_data from flat cache
            mag_data = {
                "time_tag": lat.get("time_tag", ""),
                "bx_gsm":   lat.get("bx_gsm", 0.0),
                "by_gsm":   lat.get("by_gsm", 0.0),
                "bz_gsm":   lat.get("bz_gsm", 0.0),
                "bt":       lat.get("bt", 0.0),
            }
            plasma_data = {
                "time_tag":    lat.get("time_tag", ""),
                "density":     lat.get("density", 5.0),
                "speed":       lat.get("speed", 400.0),
                "temperature": lat.get("temperature"),
            }
            age_min    = round(age_s / 60.0, 1)
            source_tag = f"DSCOVR_cache_{age_min}min"
            log.warning("DSCOVR unavailable, using cached solar wind (age: %.1f min)", age_min)

    # ── Quiet fallback (no data at all) ───────────────────────────────────────
    if mag_data is None or plasma_data is None:
        log.error("DSCOVR unavailable and cache empty. Using quiet-condition defaults.")
        mag_data    = {"time_tag": "", "bx_gsm": 0.0, "by_gsm": 0.0, "bz_gsm": 0.0, "bt": 5.0}
        plasma_data = {"time_tag": "", "density": 5.0, "speed": 400.0, "temperature": None}
        source_tag  = "quiet_fallback"

    # ── Derived quantities ────────────────────────────────────────────────────
    derived = _compute_derived(mag_data, plasma_data)
    score   = _compute_score(derived, mag_data, plasma_data)

    # ── Cache update (live data only) ─────────────────────────────────────────
    if source_tag == "DSCOVR_live":
        _save_cache({
            "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "latest": {
                "time_tag":    mag_data["time_tag"],
                "bx_gsm":      mag_data["bx_gsm"],
                "by_gsm":      mag_data["by_gsm"],
                "bz_gsm":      mag_data["bz_gsm"],
                "bt":          mag_data["bt"],
                "density":     plasma_data["density"],
                "speed":       plasma_data["speed"],
                "temperature": plasma_data["temperature"],
                **derived,
                "source": "DSCOVR",
            }
        })

    latency_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "layer": "solar_wind",
        "score": score,
        "data": {
            # Raw L1 fields
            "bz_gsm":       mag_data["bz_gsm"],
            "bt":           mag_data["bt"],
            "speed_km_s":   plasma_data["speed"],
            "density_cm3":  plasma_data["density"],
            # Derived
            "b_z_south":    derived["b_z_south"],
            "p_dyn_nPa":    derived["p_dyn_nPa"],
            "dst_estimate": derived["dst_estimate"],
            "clock_angle":  derived["clock_angle"],
            # Provenance
            "source":       source_tag,
            "time_tag":     mag_data["time_tag"],
            "latency_ms":   round(latency_ms, 2),
            # Dependency flag -- enforced by convention, not code
            "connected_to_motor": False,
            "gate_required":      "experiment_0_r>0.4_p<0.05",
        },
    }


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import logging, pprint
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    print("=== SUBSTRATE solar_wind_layer (ORPHAN) ===")
    result = run()
    pprint.pprint(result)

    # Verify sign convention
    bz = result["data"]["bz_gsm"]
    bzs = result["data"]["b_z_south"]
    est = result["data"]["dst_estimate"]
    print(f"\nSign check:")
    print(f"  bz_gsm = {bz:.2f} nT  ->  b_z_south = {bzs:.2f} nT  (should be max(-bz,0))")
    print(f"  dst_estimate = {est:.2f} nT  (negative when bz_gsm < 0)")
    assert bzs >= 0, "b_z_south must be non-negative"
    if bz < 0:
        assert est < 0, "dst_estimate must be negative when bz_gsm < 0"
    print("  Assertions: OK")
