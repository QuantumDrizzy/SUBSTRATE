# SUBSTRATE

> *From quantum vacuum to geomagnetic civilization risk — a multi-scale computational framework.*

SUBSTRATE is a unified research platform analyzing electromagnetic fields across all physical scales:
from quantum field theory and quantum biological sensing, through geomagnetic dynamics and solar physics,
to cosmological structure.

## The thesis

At every scale, the same question: **what is the substrate generating the observable pattern?**

- Quantum vacuum fluctuations → fields → forces
- Radical pair coherence → biological magnetoreception
- Geomagnetic field → civilizational protection layer
- Heliospheric dynamics → cosmic ray modulation
- CMB fluctuations → large-scale structure

## Modules

| Module | Status | Description |
|--------|--------|-------------|
| `quantum_lab` | ✅ Operational | Lattice QCD, tensor networks, 354× CUDA speedup |
| `cryptotn_gpu` | ✅ Operational | Radical pair quantum biology, χ tensor engine |
| `magnon` | ✅ Operational | Avian magnetoreception, Lindblad dynamics |
| `cycle_project` | ✅ Operational | Geomagnetic field: detection, simulation, RAG, monitor, forecast |
| `heliospheric` | 📋 Planned | Solar dynamo, Be-10 proxy, heliosphere coupling |
| `cosmological` | 📋 Planned | CMB anomaly detection (Planck data, GNN) |

## Quick Start (Arch Linux / iNFAMØUS)

```bash
git clone <repo> && cd SUBSTRATE
bash install.sh        # install deps, build Rust + CUDA kernels, run smoke tests
bash validate.sh       # pre-switch checklist — must be all-green before deploying
./target/release/substrate run
```

> **Windows note:** `install.sh` / `validate.sh` are Bash scripts for the Arch target.
> On the dev machine use Git Bash or WSL to inspect them; `chmod +x` is a no-op on NTFS
> but the execute bit is preserved when pushed and cloned on Linux.

## Running the pipeline

```bash
python pipeline.py --module cycle_project
python pipeline.py --module all
```

## Hardware

- GPU: RTX 5060 Ti (sm_120, Blackwell, 16GB)
- PyTorch nightly cu128
- Rust 2021 edition
