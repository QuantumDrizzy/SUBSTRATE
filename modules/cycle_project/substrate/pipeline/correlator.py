"""
substrate.pipeline.correlator — Cross-instrument correlation engine
===================================================================

Takes multiple SubstrateResult objects and finds:
  · temporal_overlap — synchronous anomaly windows across instruments
  · semantic         — myth ↔ geological event cosine similarity
  · full             — both combined

Returns (data, meta) compatible with SubstrateLab.correlate().
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from substrate.lab import SubstrateResult


def run_correlation(
    results: "list[SubstrateResult]",
    method: str = "temporal_overlap",
    data_root: Path = Path("data/processed"),
    **kwargs: Any,
) -> tuple[Any, dict]:
    meta: dict = {
        "method": method,
        "instruments": [r.instrument for r in results],
        "warnings": [],
    }

    if method == "temporal_overlap":
        return _temporal_overlap(results, meta, **kwargs)
    elif method == "semantic":
        return _semantic(results, meta, **kwargs)
    elif method == "full":
        d1, m1 = _temporal_overlap(results, meta.copy(), **kwargs)
        d2, m2 = _semantic(results, meta.copy(), **kwargs)
        merged = {
            "temporal": d1,
            "semantic": d2,
            "markdown": _merge_markdown(d1, d2),
        }
        meta["warnings"].extend(m1.get("warnings", []))
        meta["warnings"].extend(m2.get("warnings", []))
        return merged, meta
    else:
        raise ValueError(f"Unknown correlation method: {method}")


def _temporal_overlap(
    results: "list[SubstrateResult]",
    meta: dict,
    window_kyr: float = 1.0,
    **_kw,
) -> tuple[Any, dict]:
    """
    Find time windows (kyr BP) where multiple instruments flag anomalies
    simultaneously.  Uses a simple sliding-window intersection.
    """
    all_windows: list[list[float]] = []
    labels: list[str] = []

    for r in results:
        if isinstance(r.data, dict) and "anomaly_windows" in r.data:
            all_windows.append(r.data["anomaly_windows"])
            labels.append(r.instrument)

    if len(all_windows) < 2:
        meta["warnings"].append(
            "Need ≥2 instruments with anomaly_windows for temporal overlap"
        )
        return {"synchronous_events": [], "markdown": "No overlap data available."}, meta

    # Intersection: events within `window_kyr` kyr of each other across all channels
    reference = all_windows[0]
    synchronous = []
    for t_ref in reference:
        match_count = 1
        for other in all_windows[1:]:
            if any(abs(t_ref - t_other) <= window_kyr for t_other in other):
                match_count += 1
        if match_count == len(all_windows):
            synchronous.append(round(t_ref, 2))

    meta["window_kyr"] = window_kyr
    meta["n_synchronous"] = len(synchronous)

    lines = [
        f"## Synchronous Anomaly Windows (±{window_kyr} kyr tolerance)",
        f"**Instruments:** {', '.join(labels)}",
        "",
        "| Event (kyr BP) | Notes |",
        "|---|---|",
    ]
    for t in synchronous:
        label = _known_event_label(t)
        lines.append(f"| {t} | {label} |")

    return {
        "synchronous_events": synchronous,
        "instruments": labels,
        "markdown": "\n".join(lines),
    }, meta


def _semantic(
    results: "list[SubstrateResult]",
    meta: dict,
    **_kw,
) -> tuple[Any, dict]:
    """Extract pre-computed myth correlation markdown from mythology result."""
    myth_results = [r for r in results if r.instrument == "mythology"]
    if not myth_results:
        meta["warnings"].append("No mythology result provided for semantic correlation")
        return {"markdown": "No myth data."}, meta

    myth_result = myth_results[0]
    markdown = (
        myth_result.data.get("markdown", "")
        if isinstance(myth_result.data, dict)
        else ""
    )
    return {"markdown": markdown, "table": myth_result.data.get("table", [])}, meta


def _merge_markdown(d1: dict, d2: dict) -> str:
    return "\n\n---\n\n".join([
        d1.get("markdown", ""),
        d2.get("markdown", ""),
    ])


def _known_event_label(t_kyr: float) -> str:
    known = {
        (12.5, 13.3): "Younger Dryas onset (~12.9 ka)",
        (11.5, 12.0): "YD termination (~11.7 ka)",
        (8.0,  8.5):  "8.2 ka event",
        (4.0,  4.5):  "4.2 ka aridification",
        (40.0, 42.0): "Laschamps geomagnetic excursion (~41 ka)",
        (73.0, 75.0): "Toba supervolcano (~74 ka)",
    }
    for (lo, hi), label in known.items():
        if lo <= t_kyr <= hi:
            return label
    return "—"
