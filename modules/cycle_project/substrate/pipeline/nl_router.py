"""
substrate.pipeline.nl_router — Natural-language query dispatcher
================================================================

Keyword-based fallback dispatcher for the orchestrator's query() bridge.
When the orchestrator's Qwen model is available, it should call SubstrateLab.run()
directly with structured parameters instead of going through here.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from substrate.lab import SubstrateLab, SubstrateResult

_ROUTES = [
    # (pattern, instrument, task, kwargs_factory)
    (r"anomal|geomagneti|proxy|ice.?core|younger dryas|GISP|Vostok",
     "geomagnetic", "anomaly_scan", {}),
    (r"forecast|predict|future|lstm",
     "forecast", "predict", {}),
    (r"simulat|LBM|lithospher|pole.?shift|lattice.?boltzmann",
     "simulation", "run_lbm", {}),
    (r"myth|flood|deucalion|atrahasis|popol|fimbul|manu",
     "mythology", "correlate_events", {}),
    (r"solar|F10\.7|cosmic.?ray|coherence|oulu|SWPC|pole.?accelerat",
     "coherence", "snapshot", {}),
    (r"quantum|radical.?pair|lindblad|tensor.?network|cryptochrome|RF.?noise|spin",
     "quantum_bio", "radical_pair_yield", {}),
]


def dispatch(lab: "SubstrateLab", query: str) -> "SubstrateResult":
    query_lower = query.lower()
    for pattern, instrument, task, kwargs in _ROUTES:
        if re.search(pattern, query, re.IGNORECASE):
            return lab.run(instrument, task=task, **kwargs)

    # Default: geomagnetic overview
    return lab.run("geomagnetic", task="eda_overview")
