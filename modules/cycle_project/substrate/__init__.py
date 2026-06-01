"""
SUBSTRATE — Sovereign Quantum Research Platform
================================================

Single entry point for all SUBSTRATE instruments and analysis pipelines.

    from substrate import SubstrateLab

    lab = SubstrateLab()
    results = lab.run("geomagnetic", task="anomaly_scan", window_kyr=500)
    report  = lab.report(results)

SUBSTRATE integrates:
  · Geomagnetic proxy analysis (GNN anomaly detection over 5 ice-core proxies)
  · Palaeoclimate forecasting  (LSTM ensemble, 100-yr resolution)
  · Lithospheric simulation    (Lattice Boltzmann D3Q19, U(1) gauge field)
  · Mythology corpus RAG       (ChromaDB + LangChain, 7 myth traditions)
  · Field coherence monitor    (real-time NOAA SWPC + Oulu NM Rust dashboard)
  · Quantum biology engine     (Lindblad OQS, radical pairs, tensor networks)

All instruments share a common data bus (Apache Parquet + metadata registry)
and expose a uniform Python API so the orchestrator can query any instrument in one call.
"""

from substrate.lab import SubstrateLab

__all__ = ["SubstrateLab"]
__version__ = "0.1.0"
