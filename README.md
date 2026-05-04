# QUANTUM-LAB

Lattice field theory research facility. Three independent particle physics projects sharing a common infrastructure: CUDA-accelerated gauge computations, normalizing flow sampling, and a sovereign experiment tracking system.

**Target hardware:** RTX 5060 Ti 16 GB (sm_120, Blackwell). No cloud. No notebooks-as-a-service. Bare metal.

---

## results

### P3 — U(1) gauge theory (CUDA reactor)

| engine | action value | time | speedup |
|--------|-------------|------|---------|
| JAX (CPU/GPU) | -5.1589670 | 829 ms | baseline |
| **CUDA C++ kernel** | **-5.1589675** | **0.578 ms** | **354×** |
| precision diff | 4.77e-07 | — | bit-perfect |

HMC thermalization: 200 configs on 8×8 lattice, β=1.0, acceptance rate 0.98.
Plaquette energy: 0.4505 ± 0.0779.

### P1 — LHCb anomaly detection

GNN Autoencoder on CERN Open Data (b→sℓℓ decays). ROC-AUC: **0.765** on 4-feature latent space. Quantum Boltzmann Machine background model via PennyLane.

### P2 — W-boson quantum unfolding

Sequential QUBO unfolding of Jacobian peak distributions. Sliding window → Cirq/PennyLane solver. Benchmarked against classical SVD baseline.

---

## architecture

```
QUANTUM_LAB/
├── P1_LHCB/
│   ├── main.py                 ← orchestrator (ingestion → GNN → QBM)
│   ├── ingestion.py            ← uproot + graph construction
│   ├── gnn_autoencoder.py      ← PyTorch Geometric autoencoder
│   └── qbm_pennylane.py        ← Quantum Boltzmann Machine
│
├── P2_WBOSON/
│   ├── data_generator.py       ← Jacobian peak toy data
│   ├── quantum_unfolder.py     ← QUBO sliding window solver
│   └── benchmark.py            ← SVD vs quantum comparison
│
├── P3_G2/
│   ├── main.py                 ← orchestrator (HMC → CNF → TN → CUDA)
│   ├── lattice_hmc.py          ← JAX Hybrid Monte Carlo for U(1)
│   ├── cnf_flow.py             ← Continuous Normalizing Flow (PyTorch)
│   ├── tn_quimb.py             ← Tensor network contraction (quimb)
│   ├── cuda_accelerator.py     ← CuPy RawKernel C++ plaquette action
│   └── visualization_3d.py     ← PyVista/VTK bare-metal 3D rendering
│
├── tracking.py                 ← SQLite sovereign experiment tracker
├── query.py                    ← CLI experiment query interface
├── plot_metrics.py             ← Offline metric visualization
└── experiments.db              ← Local experiment database
```

---

## stack

| layer | technology |
|-------|-----------|
| gauge sampling | JAX (HMC with leapfrog integrator) |
| generative model | PyTorch + torchdiffeq (Neural ODE / CNF) |
| tensor networks | quimb + cotengra (PEPS, CTMRG, BMPS) |
| GPU acceleration | CuPy RawKernel (native C++ CUDA) |
| particle data | uproot (ROOT → NumPy) |
| quantum circuits | PennyLane + Cirq |
| 3D visualization | PyVista / VTK (bare-metal OpenGL) |
| experiment tracking | SQLite (no MLflow, no cloud) |

---

## the CUDA kernel

The plaquette action kernel maps each lattice site to a CUDA thread. The gauge field U(1) action is computed as:

```
S = -β Σ Re(U_μ(n) · U_ν(n+μ) · U_μ(n+ν)* · U_ν(n)*)
```

Each thread computes one plaquette. Block reduction via `__syncthreads()`. The kernel achieves **354× speedup** over JAX with sub-microsecond precision (diff < 5e-07).

```cpp
__global__ void compute_plaquette_kernel(
    const float* field_cos, const float* field_sin,
    float* block_results, int L, float beta)
```

---

## sovereign tracking

Every experiment run is logged to `experiments.db` (SQLite) with:
- Parameters (JSON): lattice size, β, epochs, bond dimension
- Metrics (JSON): plaquette energy, acceptance rate, loss curves
- Artifacts: paths to generated plots and configs
- Timestamp and run notes

No MLflow. No localhost servers. No web dashboards. Query with `python query.py`.

---

## run

```bash
# full pipeline (HMC → CNF → TN → CUDA benchmark)
python P3_G2/main.py

# LHCb anomaly detection
python P1_LHCB/main.py

# W-boson unfolding benchmark
python P2_WBOSON/benchmark.py

# query experiment history
python query.py --last 10
```

---

## requirements

```
jax[cuda12]
torch
torchdiffeq
torch_geometric
quimb
cotengra
cupy-cuda12x
pennylane
cirq
uproot
pyvista
```

---

## related

- [CryptoTN-GPU](https://github.com/QuantumDrizzy/CryptoTN-GPU) — GPU tensor networks for quantum biology
- [KHAOS](https://github.com/QuantumDrizzy/KHAOS) — BCI kernel with CUDA DSP
- [quantum-geo-metrology](https://github.com/QuantumDrizzy/quantum-geo-metrology) — geophysical quantum computing

---

*Antonio Rodríguez (QuantumDrizzy) · research software engineer*
