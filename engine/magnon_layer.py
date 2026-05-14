"""Avian-magnetoreception biosensing layer -- Lindblad master equation via terra.

Experiment 0: anisotropic-decoherence calibration
Branch: experiment-0-lindblad-calibration

Physical mechanism (confirmed Gate 3, 2026-05-14):
    Field sensitivity in the 4x4 terra solver arises from anisotropic
    decoherence: T2 Lindblad operators (S1z x I, I x S2z) are fixed in the
    lab z-frame. Rotation of B_earth relative to this axis modulates the
    effective dephasing rate and therefore singlet yield Y_s.
    DeltaY_s(90deg) ~ 1.16e-2 (four orders above noise threshold).
    Delta-g contribution < 1e-15 (K_S >> 1/T_Deltag at Earth-field amplitude).

Cable: geomagnetic cache dst.json -> Dst_nT -> DeltaBh -> b_earth(t)

Attribution: Dst<->Y_s correlation confirms anisotropic-decoherence sensitivity.
It does NOT confirm Delta-g radical pair magnetoreception (Gate 4: 12D + A_N5).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

# ── path setup ─────────────────────────────────────────────────────────────────
_MAGNON_ROOT = Path(__file__).resolve().parent.parent / "modules" / "magnon"
if _MAGNON_ROOT.is_dir():
    sys.path.insert(0, str(_MAGNON_ROOT))

# ── Dst cache (written by geomagnetic_layer.py) ────────────────────────────────
_DST_CACHE = Path.home() / ".cache" / "substrate" / "geomagnetic" / "dst.json"

# ── IGRF-13 baseline for Murcia, Spain (~38N, 1W) ──────────────────────────────
_BX_BASE = 24.5e-6    # North horizontal (T)
_BY_BASE =  0.5e-6    # East horizontal  (T)
_BZ_BASE = -39.0e-6   # Vertical downward (T, negative in Spain)
_BH_BASE = math.sqrt(_BX_BASE**2 + _BY_BASE**2)   # ~24.5006 uT


def _read_dst_nT():
    """Read latest Dst from geomagnetic cache.  Returns (dst_nT, source_tag)."""
    try:
        raw = json.loads(_DST_CACHE.read_text(encoding="utf-8"))
        if isinstance(raw, list) and raw:
            row = raw[0]
            if isinstance(row, dict):
                # Post-March-2026: list[{time_tag, dst}]
                latest = max(raw, key=lambda r: r.get("time_tag", ""))
                return float(latest.get("dst", 0.0) or 0.0), "noaa_dst_cache"
            if isinstance(row, list) and len(row) > 1:
                # Legacy: list[[time_tag, dst], ...]
                latest = max(raw, key=lambda r: r[0] if isinstance(r, list) else "")
                return float(latest[1]), "noaa_dst_cache_legacy"
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return 0.0, "dst_unavailable"


def _dst_to_b_earth(dst_nT):
    """Convert Dst (nT) to perturbed Earth-field vector for Murcia.

    Dst measures the ring-current depression of the horizontal component:
        DeltaBh = Dst * 1e-9 T   (Dst < 0 during storms)
    Applied as a scale factor on (Bx, By); Bz unchanged (ring current is
    horizontal). Physical floor: Bh >= 1e-7 T (field can't reverse from Dst).
    """
    bh_new = max(_BH_BASE + dst_nT * 1e-9, 1.0e-7)
    f = bh_new / _BH_BASE
    return (_BX_BASE * f, _BY_BASE * f, _BZ_BASE)


def run(params=None):
    try:
        from dll_healing import heal
        heal()

        from terra.sensors.noise_tensor import capture_and_tensorize   # type: ignore
        from terra.quantum.radical_pair import measure_decoherence      # type: ignore

        # Capture EM noise (falls back to synthetic urban env without RTL-SDR)
        noise = capture_and_tensorize(sdr_center_freq=100e6, sdr_duration=0.05)

        # ── Experiment 0 cable: Dst -> dynamic B_earth ─────────────────────────
        dst_nT, dst_source = _read_dst_nT()
        b_earth = _dst_to_b_earth(dst_nT)
        # Fallback: dst_nT=0.0 -> b_earth = IGRF-13 Murcia baseline (no change).

        # Lindblad evolution: 200 Euler steps (1 us / 5 ns) on 4x4 density matrix
        obs = measure_decoherence(
            h_noise=noise.hamiltonian,
            b_earth=b_earth,
            t_total=1e-6,
            dt=5e-9,
        )

        score = float(obs.fidelity)
        return {
            "layer": "magnon",
            "score": score,
            "data": {
                "fidelity":            score,
                "singlet_yield":       float(obs.singlet_yield),
                "triplet_yield":       float(obs.triplet_yield),
                "coherence_magnitude": float(obs.coherence_magnitude),
                "t2_effective_us":     float(obs.t2_effective * 1e6),
                "purity":              float(obs.purity),
                "entropy_bits":        float(obs.entropy),
                "bloch_x":             float(obs.bloch_x),
                "bloch_y":             float(obs.bloch_y),
                "bloch_z":             float(obs.bloch_z),
                "compass_sensitivity": float(obs.compass_sensitivity),
                "noise_power_ratio":   float(obs.noise_power_ratio),
                "noise_b_rms_T":       float(noise.b_noise_rms),
                "noise_source":        noise.source,
                "solve_ms":            float(obs.solve_time_ms),
                "method":              "lindblad_terra",
                # ── Experiment 0 provenance ──────────────────────────────────
                "dst_nT":        round(dst_nT, 2),
                "dst_source":    dst_source,
                "b_earth_nT":    [round(v * 1e9, 3) for v in b_earth],
                "mechanism":     "anisotropic_t2_decoherence",
            },
        }
    except Exception as exc:
        import numpy as np
        rng = np.random.default_rng()
        score = float(rng.uniform(0.30, 0.80))
        return {
            "layer": "magnon",
            "score": score,
            "data": {"fidelity": score, "synthetic": True, "error": str(exc)},
        }
