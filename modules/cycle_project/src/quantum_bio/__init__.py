"""
quantum_bio — Quantum Biology Engine
=====================================

GPU-accelerated simulation of quantum spin dynamics in biological systems.

Modules
-------
lindblad        Open quantum systems — Lindblad master equation solver
radical_pair    Cryptochrome radical pair mechanism (FAD–FADH•)
tensor_network  MPS/tensor network state compression (CuPy GPU)
rf_noise        RF field perturbation on spin coherence lifetime

Science basis
-------------
· Ritz et al. 2004 — radical pair in avian cryptochrome
· Schulten 1978    — original radical pair mechanism
· Haberkorn 1976   — recombination rate formalism
· Muheim 2019      — in vivo RF disruption of avian compass
"""

from quantum_bio.lindblad import LindbladSolver
from quantum_bio.radical_pair import RadicalPairSystem
from quantum_bio.tensor_network import MPSEngine
from quantum_bio.rf_noise import RFSensitivityScanner

__all__ = [
    "LindbladSolver",
    "RadicalPairSystem",
    "MPSEngine",
    "RFSensitivityScanner",
]
