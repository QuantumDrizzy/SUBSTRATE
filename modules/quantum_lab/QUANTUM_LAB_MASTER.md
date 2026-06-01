# Quantum/AI Particle Physics Laboratory

This repository contains three independent particle physics projects designed to run on a heterogeneous local cluster (16GB GPU, multi-core CPU, 20-qubit quantum simulator).

## ⚛️ PROJECT 1: Anomaly Hunting in the B Meson (LHCb)
**Objective:** Detect signals of new physics in $b \to s\ell\ell$ decays using CERN Open Data.
*   **Classical Pipeline:** Ingestion with `uproot`, dimensionality reduction with GNN Autoencoder (`PyTorch Geometric`).
*   **Quantum Pipeline:** Generative background modeling in latent space using Quantum Boltzmann Machines (`PennyLane` + `qiskit.aer` + `cuQuantum`).
*   **Directory:** `P1_LHCB/`

## 💎 PROJECT 2: Quantum Unfolding of the W Boson
**Objective:** Validate a sequential quantum unfolding architecture to recover kinematic distributions.
*   **Classical Pipeline:** Toy data generation with Jacobian peak, differentiable fits with `JAX`.
*   **Quantum Pipeline:** Unfolding via sliding window mapped to QUBO solved with `cirq`/`PennyLane`.
*   **Directory:** `P2_WBOSON/`

## 🌀 PROJECT 3: Tensor Networks for Vacuum Polarization (g-2)
**Objective:** Validate computational methods (Normalizing Flows + Tensor Networks) in U(1) and Z2 gauge theories.
*   **AI Pipeline:** Gauge configuration generation bypassing critical slowing down via Equivariant Continuous Normalizing Flows (`PyTorch` + Neural ODEs).
*   **Tensor Networks Pipeline:** Vacuum representation as 2+1D PEPS, approximate contraction via CTMRG or BMPS using `quimb` on GPU (`CuPy`).
*   **Directory:** `P3_G2/`
