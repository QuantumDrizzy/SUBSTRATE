"""
substrate.lab — SubstrateLab unified research interface
========================================================

SubstrateLab is the single Python object that the orchestrator (and humans) use to
interact with every SUBSTRATE instrument.  Think of it as the control room:
you don't go into the detector hall; you call the control room API.

Design principles (GEANT4 / ROOT inspired):
  · Instruments are lazy-loaded — importing SubstrateLab costs nothing
    until you actually run an instrument.
  · All results are returned as SubstrateResult (dict-like) objects that
    carry provenance metadata (instrument, timestamp, parameters, paths).
  · correlate() runs multiple instruments and passes their shared time axis
    through the cross-correlation pipeline — this is the orchestrator bridge.
  · report() renders markdown or JSON summaries without side effects.
  · GPU is optional everywhere; all instruments degrade gracefully to CPU.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("substrate")

# ---------------------------------------------------------------------------
# Instrument registry
# ---------------------------------------------------------------------------
_INSTRUMENT_MAP: dict[str, str] = {
    "geomagnetic":  "substrate.instruments.geomagnetic:GeomagneticInstrument",
    "forecast":     "substrate.instruments.forecast:ForecastInstrument",
    "simulation":   "substrate.instruments.simulation:SimulationInstrument",
    "mythology":    "substrate.instruments.mythology_instrument:MythologyInstrument",
    "coherence":    "substrate.instruments.coherence:CoherenceInstrument",
    "quantum_bio":  "substrate.instruments.quantum_bio:QuantumBioInstrument",
}


def _load_instrument(name: str):
    """Import and instantiate an instrument class by registry name."""
    if name not in _INSTRUMENT_MAP:
        raise ValueError(
            f"Unknown instrument '{name}'. Available: {list(_INSTRUMENT_MAP)}"
        )
    module_path, class_name = _INSTRUMENT_MAP[name].rsplit(":", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls()


# ---------------------------------------------------------------------------
# SubstrateResult
# ---------------------------------------------------------------------------
class SubstrateResult:
    """
    Uniform result container returned by every instrument.

    Attributes
    ----------
    instrument : str
        Which instrument produced this result.
    task : str
        Sub-task or analysis type that was run.
    data : Any
        Primary payload (DataFrame, array, dict, path to Parquet/PNG, …).
    meta : dict
        Provenance: parameters, timestamps, GPU flag, warnings.
    """

    def __init__(
        self,
        instrument: str,
        task: str,
        data: Any,
        meta: dict | None = None,
    ) -> None:
        self.instrument = instrument
        self.task = task
        self.data = data
        self.meta: dict = meta or {}
        self.meta.setdefault("timestamp_utc", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

    # dict-like access for the orchestrator bridge
    def __getitem__(self, key: str) -> Any:
        if key == "data":
            return self.data
        return self.meta[key]

    def keys(self):
        return ["data", *self.meta.keys()]

    def to_json(self) -> str:
        payload = {
            "instrument": self.instrument,
            "task": self.task,
            "meta": self.meta,
        }
        # data may not be JSON-serialisable (DataFrame etc.) — stringify it
        try:
            payload["data_repr"] = str(self.data)[:2000]
        except Exception:
            payload["data_repr"] = "<non-serialisable>"
        return json.dumps(payload, indent=2)

    def __repr__(self) -> str:
        return (
            f"SubstrateResult(instrument={self.instrument!r}, "
            f"task={self.task!r}, "
            f"data={type(self.data).__name__})"
        )


# ---------------------------------------------------------------------------
# SubstrateLab
# ---------------------------------------------------------------------------
class SubstrateLab:
    """
    SUBSTRATE Unified Research Platform.

    Parameters
    ----------
    data_root : str | Path, optional
        Root directory for processed data.  Defaults to
        ``<project_root>/data/processed/``.
    gpu : bool, optional
        Enable GPU acceleration globally.  Each instrument checks this flag
        and falls back to CPU if CUDA is unavailable regardless.
    verbose : bool, optional
        Enable INFO-level logging to stdout.

    Examples
    --------
    Basic single-instrument run::

        lab = SubstrateLab()
        r = lab.run("geomagnetic", task="anomaly_scan", window_kyr=500)
        print(r)

    Cross-instrument correlation (the orchestrator use-case)::

        r_geo  = lab.run("geomagnetic",  task="anomaly_scan")
        r_myth = lab.run("mythology",    task="correlate_events",
                         events=r_geo.data["anomaly_windows"])
        report = lab.correlate([r_geo, r_myth])
        print(report.data["markdown"])

    Quantum biology::

        r_qb = lab.run("quantum_bio", task="radical_pair_yield",
                       B_field_uT=50.0, rf_freq_MHz=1.4)
        print(r_qb.data["singlet_yield"])
    """

    def __init__(
        self,
        data_root: str | Path | None = None,
        gpu: bool = True,
        verbose: bool = False,
    ) -> None:
        if verbose:
            logging.basicConfig(level=logging.INFO,
                                format="[SUBSTRATE] %(message)s")

        # Resolve data root relative to this file's project structure
        self._project_root = Path(__file__).resolve().parent.parent
        self._data_root = (
            Path(data_root) if data_root else self._project_root / "data" / "processed"
        )
        self._data_root.mkdir(parents=True, exist_ok=True)

        self.gpu = gpu
        self._instruments: dict[str, Any] = {}   # lazy cache

        logger.info("SubstrateLab initialised  data_root=%s  gpu=%s",
                    self._data_root, self.gpu)

    # ------------------------------------------------------------------
    # Instrument access
    # ------------------------------------------------------------------
    def instrument(self, name: str):
        """Return (and cache) a live instrument instance."""
        if name not in self._instruments:
            inst = _load_instrument(name)
            inst._lab = self          # back-reference for shared config
            self._instruments[name] = inst
            logger.info("Loaded instrument: %s", name)
        return self._instruments[name]

    @property
    def available(self) -> list[str]:
        """List all registered instrument names."""
        return list(_INSTRUMENT_MAP)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def run(self, instrument_name: str, task: str = "default", **kwargs) -> SubstrateResult:
        """
        Run a single instrument task.

        Parameters
        ----------
        instrument_name : str
            One of: geomagnetic, forecast, simulation, mythology,
            coherence, quantum_bio.
        task : str
            Instrument-specific task name (see each instrument's docstring).
        **kwargs
            Passed verbatim to the instrument's execute() method.

        Returns
        -------
        SubstrateResult
        """
        inst = self.instrument(instrument_name)
        t0 = time.perf_counter()
        logger.info("run  instrument=%s  task=%s", instrument_name, task)

        data, meta = inst.execute(task=task, data_root=self._data_root,
                                  gpu=self.gpu, **kwargs)
        elapsed = time.perf_counter() - t0
        meta["elapsed_s"] = round(elapsed, 3)
        meta["params"] = kwargs

        result = SubstrateResult(instrument_name, task, data, meta)
        logger.info("done instrument=%s  task=%s  %.2fs", instrument_name, task, elapsed)
        return result

    def correlate(
        self,
        results: list[SubstrateResult],
        method: str = "temporal_overlap",
        **kwargs,
    ) -> SubstrateResult:
        """
        Cross-instrument correlation analysis.

        Takes a list of SubstrateResult objects (from different instruments)
        and finds temporal / thematic overlaps.  The output is a new
        SubstrateResult with a markdown correlation table and a shared
        event timeline.

        Parameters
        ----------
        results : list[SubstrateResult]
            At least two results to correlate.  Each must expose a
            ``time_axis`` key in its meta dict (kyr BP array or similar).
        method : str
            'temporal_overlap' (default) — find synchronous anomaly windows.
            'semantic'         — myth ↔ geological event cosine similarity.
            'full'             — both methods combined.
        **kwargs
            Forwarded to the correlation engine.
        """
        from substrate.pipeline.correlator import run_correlation
        data, meta = run_correlation(results, method=method,
                                     data_root=self._data_root, **kwargs)
        return SubstrateResult("__correlator__", method, data, meta)

    def report(
        self,
        result: SubstrateResult,
        fmt: str = "markdown",
        out: str | Path | None = None,
    ) -> SubstrateResult:
        """
        Render a SubstrateResult into a human-readable report.

        Parameters
        ----------
        result : SubstrateResult
            Any result returned by run() or correlate().
        fmt : str
            'markdown' (default), 'json', or 'html'.
        out : str | Path, optional
            Write the report to this path.  If None, returns the text in
            result.data['text'].

        Returns
        -------
        SubstrateResult  (instrument='__reporter__')
        """
        from substrate.pipeline.reporter import render
        text = render(result, fmt=fmt)
        if out:
            Path(out).write_text(text, encoding="utf-8")
            logger.info("Report written to %s", out)
        return SubstrateResult("__reporter__", fmt, {"text": text},
                               {"source_instrument": result.instrument,
                                "source_task": result.task})

    # ------------------------------------------------------------------
    # Convenience: the orchestrator bridge
    # ------------------------------------------------------------------
    def query(self, natural_language: str) -> SubstrateResult:
        """
        High-level natural-language interface for the orchestrator.

        Parses a query like "show anomaly windows in the last 500 kyr" and
        dispatches to the appropriate instrument(s).  Returns a merged result
        suitable for the orchestrator's response formatter.

        Note: full NL routing requires the orchestrator's Qwen model.  Without the orchestrator,
        falls back to a keyword-based dispatcher.
        """
        from substrate.pipeline.nl_router import dispatch
        return dispatch(self, natural_language)

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        loaded = list(self._instruments.keys()) or ["none"]
        return (
            f"SubstrateLab(gpu={self.gpu}, "
            f"data_root={self._data_root}, "
            f"loaded_instruments={loaded})"
        )
