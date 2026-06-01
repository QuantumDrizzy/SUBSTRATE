# khaos-core

Closed-loop BCI kernel designed for hard real-time operation. The core loop runs at sub-millisecond latency on FPGA; the neural decoding and feedback synthesis pipeline is quantum-classical hybrid.

**Status:** FPGA stub complete (Verilog/HLS). Next phase: hardware bring-up on dev board + OptiX-based volumetric neural visualization.

---

## what it does

khaos-core is the signal processing and control backbone for a closed-loop brain-computer interface. "Closed-loop" means the system reads neural signals, decodes intent or state, and generates feedback (stimulation, visual, haptic) within a deterministic time budget — the loop completes before the brain's next relevant oscillation cycle.

The architecture separates three concerns:

- **Acquisition layer** — FPGA handles analog front-end timing, spike detection, and DMA to host. Hard real-time, deterministic latency.
- **Decode layer** — neural state estimation on CPU/GPU. Currently: adaptive Kalman filter + learned latent space (Rust/CUDA). The "quantum" side lives here: QNN-assisted state classifiers for high-dimensional neural population codes.
- **Feedback layer** — OptiX volumetric renderer for real-time 3D visualization of decoded neural state, plus stimulation parameter synthesis.

---

## architecture

```
┌─────────────────────────────────────────────────────────┐
│                    khaos-core pipeline                  │
├──────────────────┬──────────────────┬───────────────────┤
│  ACQUISITION     │  DECODE          │  FEEDBACK         │
│  FPGA            │  CPU/GPU         │  GPU + stim       │
│                  │                  │                   │
│  • analog AFE    │  • Kalman filter │  • OptiX render   │
│  • spike detect  │  • latent space  │  • volumetric viz │
│  • DMA → host    │  • QNN classify  │  • stim synth     │
│  • <100 µs       │  • Rust/CUDA     │  • feedback ctrl  │
└──────────────────┴──────────────────┴───────────────────┘
```

The FPGA stub implements the acquisition layer in Verilog/HLS. The decode and feedback layers run on the host GPU. Inter-layer communication is via shared memory ring buffers with lock-free producers/consumers.

---

## quantum-classical hybrid

The QNN classifiers in the decode layer operate on projected neural population vectors. The motivation is that neural state spaces for motor/cognitive BCI are high-dimensional and non-stationary — classical linear decoders generalize poorly across sessions. Variational quantum circuits (parameterized unitaries on a small qubit register) serve as the nonlinear kernel in the latent space classifier, with the rest of the pipeline classical.

This is **not** "quantum advantage" marketing. It's a testable hypothesis: does a QNN kernel generalize better than an RBF kernel on neural population data? The architecture is designed to let you swap the classifier without touching the acquisition or feedback layers.

---

## current state

| component | status | notes |
|-----------|--------|-------|
| FPGA stub (Verilog/HLS) | complete | simulated, not yet on hardware |
| Spike detection kernel | complete | threshold + template matching |
| Adaptive Kalman filter | complete | Rust, benchmarked |
| Latent space encoder | in progress | Rust/CUDA |
| QNN classifier | in progress | variational circuit, JAX |
| OptiX volumetric viz | planned | next milestone |
| Hardware bring-up | planned | dev board selection TBD |
| Closed-loop validation | planned | phantom + in-vitro target |

---

## stack

| layer | technology |
|-------|-----------|
| FPGA | Verilog / Vivado HLS |
| Real-time kernel | Rust (no_std compatible) |
| GPU decode | CUDA / CuPy |
| QNN | JAX + PennyLane (variational circuits) |
| Visualization | NVIDIA OptiX 8 |
| Build | Cargo + CMake |

---

## repo structure

```
khaos-core/
├── fpga/            Verilog/HLS sources, simulation testbenches
├── kernel/          Rust real-time kernel (acquisition, ring buffers)
├── decode/          Kalman filter, latent space encoder (Rust/CUDA)
├── qnn/             Variational quantum classifiers (JAX/PennyLane)
├── viz/             OptiX volumetric renderer
├── tests/           Unit + integration tests
└── docs/            Architecture notes, latency budget analysis
```

---

## getting started

```bash
git clone https://github.com/QuantumDrizzy/khaos-core
cd khaos-core

# Rust kernel + decode layer
cargo build --release
cargo test

# FPGA simulation (requires Vivado or xsim)
cd fpga && make sim

# QNN classifier
pip install jax pennylane
python qnn/train.py --dataset synthetic
```

GPU decode requires CUDA 12+. OptiX visualization requires driver ≥ 535 and OptiX SDK 8.

---

## roadmap

- [x] FPGA acquisition stub (Verilog/HLS, simulated)
- [x] Spike detection kernel
- [x] Adaptive Kalman filter (Rust)
- [ ] Latent space encoder (Rust/CUDA) — in progress
- [ ] QNN state classifier (JAX) — in progress
- [ ] Hardware bring-up on FPGA dev board
- [ ] OptiX volumetric neural visualization
- [ ] Closed-loop validation (phantom electrode array)

---

## related work

- Q-NAA (Quantum Neural Attention Analyzer) — higher-level attention modeling on top of khaos-core decoded states, Rust/CUDA
- eeg-epilepsy — clinical EEG signal processing pipeline (F1 = 0.945 on CHB-MIT dataset), shares preprocessing with khaos-core acquisition layer

---

## license

MIT
