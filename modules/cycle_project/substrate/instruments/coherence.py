"""
substrate.instruments.coherence — Field Coherence Monitor Instrument
====================================================================

Wraps the Rust `field_coherence` binary (egui real-time dashboard) via
subprocess, and exposes its live data feeds as Python-queryable snapshots.

Tasks
-----
  snapshot        Fetch current values: pole acceleration, F10.7 flux,
                  Oulu cosmic-ray count rate.  Returns a dict with all three.

  history         Load SQLite timeseries from the Rust binary's database.
                  Returns a DataFrame of the last N days.

  launch_dashboard
                  Spawn the egui dashboard as a background process.
                  Returns the process PID.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

from substrate.instruments.base import SubstrateInstrument

logger = logging.getLogger("substrate.coherence")

_RUST_BINARY = Path(__file__).resolve().parent.parent.parent / "target" / "release" / "field_coherence"


class CoherenceInstrument(SubstrateInstrument):
    """
    Field Coherence Monitor Instrument.

    Provides Python access to the real-time geomagnetic pole acceleration,
    solar F10.7 flux, and Oulu Neutron Monitor cosmic-ray count rates
    maintained by the Rust `field_coherence` egui dashboard.

    Can also directly query NOAA SWPC and NOAA NCEI APIs when the Rust
    binary is not compiled (pure-Python fallback).
    """

    def execute(
        self,
        task: str = "snapshot",
        data_root: Path = Path("data/processed"),
        gpu: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        meta: dict = {"warnings": []}

        if task == "snapshot":
            return self._snapshot(meta, **kwargs)
        elif task == "history":
            return self._history(data_root, meta, **kwargs)
        elif task == "launch_dashboard":
            return self._launch_dashboard(meta, **kwargs)
        else:
            raise ValueError(f"CoherenceInstrument: unknown task '{task}'")

    def _snapshot(self, meta: dict, **kw) -> tuple[Any, dict]:
        """Try Rust binary first; fall back to direct HTTP queries."""
        if _RUST_BINARY.exists():
            try:
                proc = subprocess.run(
                    [str(_RUST_BINARY), "--snapshot", "--json"],
                    capture_output=True, text=True, timeout=10
                )
                data = json.loads(proc.stdout)
                meta["source"] = "rust_binary"
                return data, meta
            except Exception as e:
                self._warn(meta, f"Rust binary failed: {e} — falling back to HTTP")

        # Pure-Python fallback: query NOAA SWPC directly
        return self._http_snapshot(meta, **kw)

    def _http_snapshot(self, meta: dict, **kw) -> tuple[Any, dict]:
        try:
            import requests
            # Solar flux F10.7
            r = requests.get(
                "https://services.swpc.noaa.gov/json/f107_cm_flux.json",
                timeout=8
            )
            f107_data = r.json()
            latest_f107 = f107_data[-1] if f107_data else {}

            data = {
                "f107_sfu": latest_f107.get("flux", None),
                "f107_timestamp": latest_f107.get("time_tag", None),
                "pole_omega_deg_yr": None,    # requires WMM API call
                "cosmic_ray_count": None,     # requires Oulu FTP
                "note": "partial — only F10.7 available via HTTP fallback",
            }
            meta["source"] = "noaa_swpc_http"
            return data, meta
        except Exception as e:
            self._warn(meta, f"HTTP snapshot failed: {e}")
            return {
                "f107_sfu": None,
                "pole_omega_deg_yr": None,
                "cosmic_ray_count": None,
                "note": "STUB — requests failed or not installed",
            }, meta

    def _history(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        n_days: int = kw.pop("n_days", 30)
        db_path = data_root / "field_coherence.sqlite"
        meta["n_days"] = n_days

        if not db_path.exists():
            self._warn(meta, f"SQLite DB not found at {db_path} — run Rust binary first")
            return {}, meta
        try:
            import pandas as pd
            import sqlite3
            con = sqlite3.connect(str(db_path))
            df = pd.read_sql(
                f"SELECT * FROM timeseries ORDER BY ts DESC LIMIT {n_days * 24}",
                con
            )
            con.close()
            return df, meta
        except ImportError:
            self._warn(meta, "pandas/sqlite3 not available")
            return {}, meta

    def _launch_dashboard(self, meta: dict, **kw) -> tuple[Any, dict]:
        if not _RUST_BINARY.exists():
            self._warn(meta, f"Rust binary not found at {_RUST_BINARY}. Run: cargo build --release")
            return {"pid": None, "note": "binary missing"}, meta
        proc = subprocess.Popen([str(_RUST_BINARY)])
        meta["pid"] = proc.pid
        logger.info("field_coherence dashboard launched (PID %d)", proc.pid)
        return {"pid": proc.pid}, meta
