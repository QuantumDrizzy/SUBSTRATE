"""
substrate.instruments — Instrument adapter layer
================================================

Each instrument is a thin adapter that:
  1. Wraps the underlying science module in src/
  2. Exposes a uniform execute(task, data_root, gpu, **kwargs) interface
  3. Returns (data, meta) — unpacked by SubstrateLab.run()

Instruments never import each other.  Cross-instrument logic lives in
substrate.pipeline.correlator and substrate.pipeline.nl_router.
"""

from substrate.instruments.base import SubstrateInstrument

__all__ = ["SubstrateInstrument"]
