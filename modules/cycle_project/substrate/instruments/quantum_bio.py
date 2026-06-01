"""
substrate.instruments.quantum_bio — Quantum Biology Engine Instrument
======================================================================

Wraps src/quantum_bio/ — the new SUBSTRATE science module.

This is the CryptoTN-GPU + TERRA-QCI capability, absorbed natively into
SUBSTRATE.  No separate project; one unified engine.

Tasks
-----
  radical_pair_yield
                  Compute singlet/triplet spin state yield for a
                  cryptochrome radical pair under static B field + RF noise.
                  Returns yield curves as a function of field strength.

  lindblad_evolve
                  Solve the Lindblad master equation for an open quantum
                  system density matrix.  Returns ρ(t) trajectory.

  tensor_network_compress
                  Compress a many-body quantum state to MPS form.
                  Returns bond entropy profile + truncation error.

  rf_sensitivity_scan
                  Sweep RF frequency and amplitude; compute coherence
                  lifetime and yield perturbation ΔΦ_S.
                  Returns 2-D heatmap array.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from substrate.instruments.base import SubstrateInstrument

logger = logging.getLogger("substrate.quantum_bio")


class QuantumBioInstrument(SubstrateInstrument):
    """
    Quantum Biology Engine Instrument.

    Simulates quantum spin dynamics in biological radical pair systems
    (cryptochrome avian compass mechanism) using:

      · Lindblad master equation  — open quantum system time evolution
      · Radical pair mechanism    — FAD–FADH• spin-correlated pair yield
      · GPU tensor networks (MPS) — bond dimension χ up to ~2500 on RTX 5060 Ti
      · RF noise modelling        — perturbation theory on coherence lifetime

    Science basis
    -------------
    Ritz et al. 2004 (radical pair in cryptochrome)
    Schulten et al. 1978 (radical pair mechanism)
    Schulten & Tillet 2009 (RF sensitivity)
    Muheim et al. 2019 (in vivo RF experiments)

    REPLIQA relevance
    -----------------
    Directly targets quantum spin in biology — the core science
    of REPLIQA's $10M program.  SUBSTRATE positions this as a
    GPU-accelerated platform contribution, not a single experiment.
    """

    def execute(
        self,
        task: str = "radical_pair_yield",
        data_root: Path = Path("data/processed"),
        gpu: bool = True,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        meta: dict = {"warnings": []}

        if task == "radical_pair_yield":
            return self._radical_pair_yield(gpu, meta, **kwargs)
        elif task == "lindblad_evolve":
            return self._lindblad_evolve(gpu, meta, **kwargs)
        elif task == "tensor_network_compress":
            return self._tn_compress(gpu, meta, **kwargs)
        elif task == "rf_sensitivity_scan":
            return self._rf_scan(data_root, gpu, meta, **kwargs)
        else:
            raise ValueError(f"QuantumBioInstrument: unknown task '{task}'")

    def _radical_pair_yield(self, gpu: bool, meta: dict, **kw) -> tuple[Any, dict]:
        B_field_uT: float = kw.pop("B_field_uT", 50.0)
        rf_freq_MHz: float = kw.pop("rf_freq_MHz", 0.0)   # 0 = no RF
        t_max_us: float = kw.pop("t_max_us", 10.0)
        n_steps: int = kw.pop("n_steps", 1000)
        meta.update({"B_field_uT": B_field_uT, "rf_freq_MHz": rf_freq_MHz,
                     "t_max_us": t_max_us, "n_steps": n_steps})

        try:
            from quantum_bio.radical_pair import RadicalPairSystem
            rp = RadicalPairSystem(B_field_uT=B_field_uT, rf_freq_MHz=rf_freq_MHz,
                                   use_gpu=gpu)
            result = rp.compute_yield(t_max_us=t_max_us, n_steps=n_steps)
            meta["gpu_used"] = rp.gpu_used
            return result, meta
        except ImportError as e:
            self._warn(meta, f"quantum_bio.radical_pair not importable: {e}")
            return self._stub_yield(B_field_uT, meta), meta

    def _stub_yield(self, B: float, meta: dict) -> dict:
        """Analytical approximation for stub mode (Haberkorn recombination)."""
        import math
        # Rough singlet yield estimate from hyperfine coupling only
        hfc_mT = 1.5   # typical FAD hyperfine (mT)
        B_mT = B * 1e-3
        phi_s = 0.25 * (1 + 1 / (1 + (B_mT / hfc_mT) ** 2))
        meta["stub_mode"] = True
        meta["approximation"] = "Haberkorn analytical"
        return {"singlet_yield": round(phi_s, 4), "triplet_yield": round(1 - phi_s, 4),
                "note": "Analytical stub — install quantum_bio for full simulation"}

    def _lindblad_evolve(self, gpu: bool, meta: dict, **kw) -> tuple[Any, dict]:
        n_qubits: int = kw.pop("n_qubits", 2)
        t_max: float = kw.pop("t_max", 1.0)
        dt: float = kw.pop("dt", 0.01)
        meta.update({"n_qubits": n_qubits, "t_max": t_max, "dt": dt})

        try:
            from quantum_bio.lindblad import LindbladSolver
            solver = LindbladSolver(n_qubits=n_qubits, use_gpu=gpu, **kw)
            rho_t = solver.evolve(t_max=t_max, dt=dt)
            return {"rho_t": rho_t, "times": solver.times}, meta
        except ImportError as e:
            self._warn(meta, f"quantum_bio.lindblad not importable: {e}")
            return {"rho_t": None, "times": None,
                    "note": "STUB — quantum_bio module required"}, meta

    def _tn_compress(self, gpu: bool, meta: dict, **kw) -> tuple[Any, dict]:
        n_sites: int = kw.pop("n_sites", 20)
        chi: int = kw.pop("chi", 64)   # bond dimension
        meta.update({"n_sites": n_sites, "chi": chi})

        try:
            from quantum_bio.tensor_network import MPSEngine
            eng = MPSEngine(n_sites=n_sites, chi=chi, use_gpu=gpu)
            result = eng.compress(**kw)
            meta["truncation_error"] = result.get("truncation_error")
            return result, meta
        except ImportError as e:
            self._warn(meta, f"quantum_bio.tensor_network not importable: {e}")
            return {"mps": None, "truncation_error": None,
                    "note": f"STUB — quantum_bio required (chi={chi})"}, meta

    def _rf_scan(self, data_root: Path, gpu: bool, meta: dict, **kw) -> tuple[Any, dict]:
        freq_range_MHz: tuple = kw.pop("freq_range_MHz", (0.1, 100.0))
        n_freq: int = kw.pop("n_freq", 50)
        B_field_uT: float = kw.pop("B_field_uT", 50.0)
        meta.update({"freq_range_MHz": freq_range_MHz,
                     "n_freq": n_freq, "B_field_uT": B_field_uT})

        try:
            from quantum_bio.rf_noise import RFSensitivityScanner
            scanner = RFSensitivityScanner(use_gpu=gpu)
            heatmap = scanner.scan(
                freq_range_MHz=freq_range_MHz,
                n_freq=n_freq,
                B_field_uT=B_field_uT,
                **kw,
            )
            out_path = data_root / "quantum_bio_rf_scan.npy"
            import numpy as np
            np.save(str(out_path), heatmap)
            meta["npy_path"] = str(out_path)
            return {"heatmap": heatmap, "npy_path": str(out_path)}, meta
        except ImportError as e:
            self._warn(meta, f"quantum_bio.rf_noise not importable: {e}")
            return {"heatmap": None, "note": "STUB"}, meta
