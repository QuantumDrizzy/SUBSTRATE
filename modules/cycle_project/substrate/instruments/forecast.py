"""
substrate.instruments.forecast — Palaeoclimate Forecasting Instrument
======================================================================

Wraps src/forward_probe/ (LSTM ensemble forecaster).

Tasks
-----
  predict         Run LSTM ensemble forward from a given start time.
                  Returns forecast array + confidence intervals.

  backtest        Compare LSTM hindcast against held-out proxy data.
                  Returns RMSE per proxy + a comparison DataFrame.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from substrate.instruments.base import SubstrateInstrument

logger = logging.getLogger("substrate.forecast")


class ForecastInstrument(SubstrateInstrument):
    """
    LSTM Ensemble Palaeoclimate Forecasting Instrument.

    Runs a bidirectional LSTM ensemble (forward_probe module) trained on
    the 5-proxy stack to generate forward projections of palaeoclimate
    state.  Used to estimate future anomaly probability given current
    geomagnetic + cosmic-ray trends.
    """

    def execute(
        self,
        task: str = "predict",
        data_root: Path = Path("data/processed"),
        gpu: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        meta: dict = {"warnings": []}

        if task == "predict":
            return self._predict(data_root, gpu, meta, **kwargs)
        elif task == "backtest":
            return self._backtest(data_root, gpu, meta, **kwargs)
        else:
            raise ValueError(f"ForecastInstrument: unknown task '{task}'")

    def _predict(self, data_root: Path, gpu: bool, meta: dict, **kw) -> tuple[Any, dict]:
        horizon_kyr: int = kw.pop("horizon_kyr", 50)
        n_ensemble: int = kw.pop("n_ensemble", 10)
        meta.update({"horizon_kyr": horizon_kyr, "n_ensemble": n_ensemble})

        try:
            import torch
            device = "cuda" if (gpu and torch.cuda.is_available()) else "cpu"
            meta["device"] = device
        except ImportError:
            device = "cpu"
            self._warn(meta, "torch not found — CPU fallback")

        try:
            from forward_probe.lstm_ensemble import run_forecast
            result = run_forecast(
                data_root=data_root,
                horizon_kyr=horizon_kyr,
                n_ensemble=n_ensemble,
                device=device,
                **kw,
            )
            return result, meta
        except ImportError:
            self._warn(meta, "forward_probe not importable — returning stub")
            stub = {
                "forecast_mean": [],
                "forecast_ci_lo": [],
                "forecast_ci_hi": [],
                "note": "STUB — install torch and ensure forward_probe is in PYTHONPATH",
            }
            return stub, meta

    def _backtest(self, data_root: Path, gpu: bool, meta: dict, **kw) -> tuple[Any, dict]:
        try:
            from forward_probe.lstm_ensemble import run_backtest
            result = run_backtest(data_root=data_root, **kw)
            return result, meta
        except ImportError:
            self._warn(meta, "forward_probe not importable — returning stub")
            return {"rmse_per_proxy": {}, "note": "STUB"}, meta
