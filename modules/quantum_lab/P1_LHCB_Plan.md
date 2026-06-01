# Implementation Plan: P1_LHCB (B-Meson Anomaly)

The objective is to build the pipeline for Project 1, isolating anomalous $b \to s\ell\ell$ events using a classical GNN Autoencoder and modeling the background with a Quantum Boltzmann Machine (QBM).

## 🛠️ Phase 0: Infrastructure & Stack
Definition of the exact dependency stack for local GPU:
- `torch`, `torch-geometric` (Classical AI)
- `pennylane`, `qiskit-aer[gpu]`, `cuquantum` (Quantum AI)

## 📥 Phase 1: Ingestion (`ingestion.py`)
Module responsible for reading CERN Open Data ntuples using `uproot`. It will transform kinematic variables of muons and kaons into PyTorch tensors.
**Tactical Adjustment:** As the real `lhcb_data.root` file is not currently available, I will include a *mock* mode (generating a synthetic DataFrame with the same expected kinematic structure: $q^2$, $p_T$, $\eta$, $\phi$). This allows us to test the pipeline end-to-end immediately.

## 🕸️ Phase 2: Topology (`gnn_autoencoder.py`)
GNN model implementation using `torch_geometric`.
- **Encoder:** Reduces the event graph (nodes = particles, edges = $\Delta R$ distances) to a 12-dimensional latent space.
- **Decoder:** Reconstructs the kinematics. Calculates Mean Squared Error (MSE) as the "Anomaly Score".

## ⚛️ Phase 3: Generative Quantum (`qbm_pennylane.py`)
Quantum Boltzmann Machine implementation on 12 qubits using `PennyLane`.
- It will train on the GNN latent space to generate Standard Model background samples using a parameterized variational ansatz.

## 🚀 Phase 4: Orchestrator (`main.py`)
Main orchestrator. It will execute the sequential pipeline:
1. Data loading $\to$ 2. GNN training $\to$ 3. Latent extraction $\to$ 4. QBM training $\to$ 5. Anomaly evaluation.
