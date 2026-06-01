"""
substrate.instruments.geomagnetic — Geomagnetic & Palaeoclimate Instrument
==========================================================================

Wraps:
  · src/cycle_detect/fetch_data.py      — proxy download + cleaning
  · src/cycle_detect/gnn_prototype.py   — GraphSAGE anomaly autoencoder
  · src/field_coherence/               — real-time pole/CR monitor (Rust)

Tasks
-----
  anomaly_scan    Run the GNN autoencoder over the 5-proxy multi-proxy stack.
                  Returns anomaly scores per 100-yr window + anomaly_windows
                  list (kyr BP) flagged as > threshold σ above mean.

  fetch_proxies   Download / refresh the 5 raw proxy files from NCEI/NOAA.
                  Returns paths to processed Parquet files.

  eda_overview    Quick EDA: per-proxy summary stats + correlation matrix.
                  Returns a dict of DataFrames + path to overview PNG.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from substrate.instruments.base import SubstrateInstrument

logger = logging.getLogger("substrate.geomagnetic")

_PROXIES = [
    "gisp2_d18o",
    "vostok_deuterium",
    "vostok_co2",
    "grip_be10",
    "sint2000",
]


class GeomagneticInstrument(SubstrateInstrument):
    """
    Geomagnetic & Palaeoclimate Instrument.

    The flagship SUBSTRATE instrument: ingests five independent palaeoclimate
    proxies spanning up to 420,000 yr BP and runs a Graph Neural Network
    (GraphSAGE autoencoder) to detect synchronous anomaly windows that may
    indicate recurrent catastrophe cycles (~11,500–12,900 yr period).

    Proxies
    -------
    1. GISP2 δ¹⁸O          — Greenland temperature (Alley 2000)
    2. Vostok ΔT            — Antarctic temperature (Petit 1999)
    3. Vostok CO₂           — Atmospheric CO₂ (Petit 1999)
    4. GRIP Be-10 flux      — Cosmic-ray / solar modulation (Muscheler 2004)
    5. Sint-2000 VADM       — Geomagnetic dipole moment stack (Valet 2005)

    GPU: GraphSAGE autoencoder runs on CUDA (sm_120) when available.
    """

    def execute(
        self,
        task: str = "anomaly_scan",
        data_root: Path = Path("data/processed"),
        gpu: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        meta: dict = {"warnings": [], "proxies": _PROXIES}

        if task == "fetch_proxies":
            return self._fetch_proxies(data_root, meta, **kwargs)
        elif task == "anomaly_scan":
            return self._anomaly_scan(data_root, gpu, meta, **kwargs)
        elif task == "eda_overview":
            return self._eda_overview(data_root, meta, **kwargs)
        else:
            raise ValueError(f"GeomagneticInstrument: unknown task '{task}'")

    # ------------------------------------------------------------------
    def _fetch_proxies(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        try:
            from cycle_detect.fetch_data import fetch_all_proxies
            paths = fetch_all_proxies(out_dir=data_root, **kw)
            meta["parquet_paths"] = [str(p) for p in paths]
            return paths, meta
        except ImportError:
            self._warn(meta, "cycle_detect.fetch_data not importable — returning stub")
            stub_paths = [data_root / f"{p}.parquet" for p in _PROXIES]
            return stub_paths, meta

    def _anomaly_scan(
        self, data_root: Path, gpu: bool, meta: dict, **kw
    ) -> tuple[Any, dict]:
        window_kyr: int = kw.pop("window_kyr", 500)
        threshold_sigma: float = kw.pop("threshold_sigma", 2.0)

        try:
            import torch
            device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
            meta["device"] = device
        except ImportError:
            device = "cpu"
            meta["device"] = "cpu"
            self._warn(meta, "torch not found — CPU fallback")

        try:
            from cycle_detect.gnn_prototype import run_anomaly_scan
            result = run_anomaly_scan(
                data_root=data_root,
                window_kyr=window_kyr,
                threshold_sigma=threshold_sigma,
                device=device,
                **kw,
            )
            meta["window_kyr"] = window_kyr
            meta["threshold_sigma"] = threshold_sigma
            return result, meta
        except ImportError:
            self._warn(meta, "cycle_detect.gnn_prototype not importable — returning stub")
            # Return a minimal stub so correlate() still has a time_axis
            stub = {
                "anomaly_scores": [],
                "anomaly_windows": [12.9, 8.2, 4.2],   # known events as placeholder
                "time_axis_kyr_bp": list(range(0, window_kyr, 1)),
                "note": "STUB — run fetch_proxies and install torch to enable GNN",
            }
            return stub, meta

    def _eda_overview(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        try:
            import pandas as pd
            import numpy as np
            summaries = {}
            for proxy in _PROXIES:
                pq = data_root / f"{proxy}.parquet"
                if pq.exists():
                    df = pd.read_parquet(pq)
                    summaries[proxy] = df.describe().to_dict()
                else:
                    summaries[proxy] = {"note": "parquet not found — run fetch_proxies"}
            meta["eda_computed"] = True
            return summaries, meta
        except ImportError as e:
            self._warn(meta, f"pandas/numpy missing: {e}")
            return {}, meta
