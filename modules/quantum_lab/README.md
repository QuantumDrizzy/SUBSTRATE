# QUANTUM-LAB

Lattice field theory research facility. Three independent particle physics projects sharing a common infrastructure: CUDA-accelerated gauge computations, normalizing flow sampling, and a sovereign experiment tracking system.

**Target hardware:** RTX 5060 Ti 16 GB (sm_120, Blackwell). No cloud. No notebooks-as-a-service. Bare metal.

---

## results

### P3 — U(1) gauge theory (CUDA reactor)

Hand-written CUDA kernel (CuPy `RawKernel`, sm_120 Blackwell) for the U(1)
plaquette/Wilson action, benchmarked against a `@jax.jit` reference. Timing is
honest: both sides warmed up, JAX `block_until_ready()`, CUDA-event timing for
kernel-only GPU work, 1000-iteration medians. **JAX runs on CPU in this build**
(no GPU backend installed), so these are GPU-vs-CPU numbers — stated as such.

| L | cells | JAX-CPU (ms) | CUDA kernel (ms) | CUDA e2e (ms) | kernel speedup | e2e speedup | max\|Δ\| |
|----:|------:|-------------:|-----------------:|--------------:|---------------:|------------:|--------:|
| 8   | 128    | 0.0272 | 0.00819 | 0.3767 | 3.3×   | 0.07× | 0.0     |
| 16  | 512    | 0.0303 | 0.00816 | 0.4207 | 3.7×   | 0.07× | 9.5e-07 |
| 32  | 2048   | 0.0367 | 0.00819 | 0.4054 | 4.5×   | 0.09× | 4.8e-06 |
| 64  | 8192   | 0.1812 | 0.00822 | 0.4164 | 22.0×  | 0.44× | 1.9e-05 |
| 128 | 32768  | 0.4183 | 0.00845 | 0.4470 | 49.5×  | 0.94× | 1.9e-05 |
| 256 | 131072 | 0.6252 | 0.00870 | 0.4777 | 71.8×  | 1.31× | 3.1e-05 |
| 512 | 524288 | 2.0204 | 0.01456 | 0.8433 | 138.8× | 2.40× | 4.1e-04 |

**The honest story is the crossover, not a single hero number.** The CUDA kernel
itself is a stable 8–15 µs across all sizes; the kernel-only *speedup* rises from
~3× (small L) to roughly 139–157× at L=512 across runs (this table's run: 139×).
That top-end spread is JAX-CPU **baseline** noise (CPU wall-clock), not kernel
variance — the table above is one representative run. End-to-end (including H2D/D2H PCIe transfers) only
breaks even around **L≈128** — below that the workload is transfer/launch-bound
and the GPU loses. Numerical agreement with JAX is < 1e-3 (fp32 accumulation;
< 1e-4 up to L≈256, growing to ~4e-4 at L=512 as more terms are summed in fp32).
Reproduce: `python P3_G2/benchmark_plaquette.py` → `benchmark_plaquette_results.json`.

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

The plaquette action kernel maps each lattice site to a CUDA thread. The gauge
field U(1) action is computed as:

```
S = -β Σ Re(U_μ(n) · U_ν(n+μ) · U_μ(n+ν)* · U_ν(n)*)
```

Each thread computes one plaquette and accumulates into shared memory; a
block-level reduction (`__syncthreads()`, tree reduction over an 8×8 tile)
produces one partial sum per block, summed on-device with `cp.sum`. Keeping the
reduction in shared memory minimizes global-memory traffic, and the 8×8 block
maps cleanly to the 2D lattice for coalesced loads.

```cpp
extern "C" __global__ void compute_plaquette_kernel(
    const float* theta,        // gauge angles, shape (2, L, L) flattened
    float* block_sums,          // one partial sum per block
    int L)                      // lattice size
```

**Why kernel-only ≠ end-to-end.** The kernel itself is tiny (8–15 µs across all
sizes tested). At small L the runtime is dominated by the fixed cost of the H2D
copy of `theta` plus the D2H copy of the scalar result — that is why the
end-to-end column only beats JAX-CPU past L≈128. For a single action evaluation
this is the launch-overhead-bound regime; the win comes when the field already
lives on device (as it does inside an HMC trajectory, where the same buffer is
reused across leapfrog steps and no per-step transfer is paid).

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
- [quantum-geo-metrology](https://github.com/QuantumDrizzy/quantum-geo-metrology) — geophysical quantum computing

---

*Antonio Rodríguez (QuantumDrizzy) · research software engineer*
