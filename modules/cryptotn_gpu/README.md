# cryptotn-gpu

GPU-accelerated MPS/MPO tensor network engine for cryptochrome radical pair spin dynamics.

**Target:** bond dimension χ ≥ 2500 on RTX 5060 Ti 16 GB (sm_120, Blackwell), breaking the χ ≤ 1500 CPU bottleneck of Hino et al. (arXiv:2509.22104, 2025).

**Status:** Phase A (CPU/quimb) complete and validated. Phase B (cuTensorNet GPU engine) functional; χ-convergence benchmarks running.

---

## why this matters

The avian magnetic compass is thought to operate via the radical pair mechanism in flavin adenine dinucleotide (FAD) and tryptophan radicals in cryptochrome 4a. Simulating the spin dynamics of 30–60 coupled nuclear spins requires working in a Hilbert space of dimension 2⁶² ≈ 10¹⁸ — tractable only with matrix product state (MPS) methods.

Hino et al. (2025) established the state of the art: χ = 1500 on CPU, ~6 hours per trajectory for a 62-spin system. That wall limits the accuracy of compass sensitivity models. Higher χ → better truncation fidelity → more reliable predictions for field-dependent singlet yield Φ_S(B) — the observable that links quantum spin physics to actual bird navigation.

This project moves the ceiling by running the TDVP sweep on GPU with cuTensorNet, targeting χ = 2500 for N ≤ 40 spins and χ = 1024 for the full 62-spin ErCry4a system.

---

## architecture

```
Phase A (CPU prototype)              Phase B (GPU hot path)
────────────────────────────         ─────────────────────────────────────
quimb + scipy                        cuTensorNet (cuQuantum Python)
ExactSolver  — dense expm, N ≤ 20   CupyKrylovSolver — GPU Arnoldi Krylov
MpsSolver    — TDVP, χ ≤ 500        CuTDVPSolver     — MPO-MPS TDVP, χ ≤ 2500
~minutes (N ≤ 20)                    ~minutes (χ = 1024, N = 62, RTX 5060 Ti)
```

```
cryptotn/
├── hamiltonian.py    spin Hamiltonians (hyperfine, Zeeman, exchange, Haberkorn)
├── radical_pair.py   system configs: FAD-W, ErCry4a, Tetrad-Trp (AtCry1)
├── tdvp.py           integrators: ExactSolver, MpsSolver
├── observables.py    singlet/triplet yields, compass sensitivity ΔΦ_S
└── cuda/
    └── engine.py     CupyKrylovSolver + CuTDVPSolver (Phase B)
```

**Key implementation note:** `CuTDVPSolver` uses a left-to-right-only 2-site sweep. The FSM MPO accumulates Liouvillian contributions L→R; adding a reverse pass causes N-fold trace double-counting. This is documented in `cuda/engine.py §7`.

---

## benchmark results (Phase A)

Three mandatory benchmarks against published literature:

| # | system | reference | status |
|---|--------|-----------|--------|
| 1 | FMO 7-site complex, 77 K / 300 K | Dunnett et al., *J. Chem. Phys.* **163**, 104109 (2025) | passing — P₁(t) RMSE < 1e-4 |
| 2 | ErCry4a, 3–4 nuclei (exact) | Hino et al., arXiv:2509.22104 (2025) | Φ_S validated; χ-convergence sweep running |
| 3 | Tetrad-Trp AtCry1 | Babcock et al., *JPCB* **128**, 4035 (2024) | scheduled |

**ErCry4a χ-convergence (Phase A, CPU, n_nuc=3):**

| χ | Φ_S | abs error vs exact | wall time |
|---|-----|--------------------|-----------|
| 2 | 0.128665 | 0.0 | 190 s |
| 4 | 0.128665 | 0.0 | 709 s |
| 8 | 0.128654 | 1.1e-5 | 1158 s |
| 16 | 0.128665 | 0.0 | 2274 s |

Phase B GPU timing logs available in `benchmarks/results/gpu_timing.jsonl`.

---

## quickstart

```bash
# Phase A — CPU, no GPU required
git clone https://github.com/[your-username]/cryptotn-gpu
cd cryptotn-gpu
pip install -e ".[dev]"
pytest tests/ -v

# smoke-test all benchmarks (~2 min)
python benchmarks/run_all.py --fast

# χ-convergence sweep (paper Figure §4.2)
python benchmarks/bench_chi.py --n-nuc 10

# ErCry4a field sweep
python benchmarks/bench_ercry4a.py --n-nuc 10 --b-fields 0.0 0.05 0.1 0.5

# FMO 7-site (77 K and 300 K)
python benchmarks/bench_fmo.py
```

WSL2 note: if running from Windows, copy the repo to `/opt/` for full filesystem performance:
```bash
sudo cp -r cryptotn-gpu/ /opt/ && sudo chown -R $USER /opt/cryptotn-gpu
```

---

## Phase B (GPU) setup

Requires CUDA 12.8+ and an NVIDIA GPU with sm_86 or newer (sm_120 for Blackwell):

```bash
pip install ".[gpu]"
# verifies both cupy and cuquantum
python -c "import cupy; import cuquantum; print('GPU ready:', cupy.cuda.runtime.getDeviceProperties(0)['name'])"
python verify_gpu.py
```

VRAM budget at χ = 2500:
- N = 40 spins: MPS tensors ~4 GB + environments ~28 GB → fits on 16 GB with selective recompute
- N = 62 spins: practical limit χ ≈ 1024 on 16 GB

---

## key parameters

| parameter | symbol | default | notes |
|-----------|--------|---------|-------|
| bond dimension | χ | 64 (A) / 2500 (B) | primary accuracy lever |
| singlet rate | k_S | 0.263 μs⁻¹ | ErCry4a (Hino 2025) |
| earth field | B | 0.05 mT | Helsinki latitude |
| integration time | t_max | 10 μs | ~5 / k_S |
| time steps | n_steps | 1000 | dt = t_max / n_steps |

---

## roadmap

- [x] Phase A: CPU reference solvers (ExactSolver, MpsSolver)
- [x] Phase A: ErCry4a, FMO benchmarks validated
- [x] Phase B: CupyKrylovSolver (GPU sparse Krylov, RMSE < 1e-4 vs exact)
- [x] Phase B: CuTDVPSolver skeleton + LR-only 2-site sweep
- [ ] Phase B: full χ = 2500 sweep on RTX 5060 Ti, N = 40
- [ ] Tetrad-Trp benchmark (Babcock 2024)
- [ ] arXiv preprint (physics.bio-ph + quant-ph)

---

## references

1. Hino et al., *arXiv:2509.22104* (2025) — ErCry4a TDVP at χ = 1500
2. Dunnett et al., *J. Chem. Phys.* **163**, 104109 (2025) — TENSO FMO benchmark
3. Babcock et al., *JPCB* **128**, 4035 (2024) — Tetrad-Trp superradiance in AtCry1
4. Maeda et al., *Nature* **453**, 387 (2008) — FAD-W radical pair spin selectivity
5. Haberkorn, *Mol. Phys.* **32**, 1491 (1976) — recombination kinetics master equation

---

## license

Apache License 2.0, inherited from the upstream project this is a fork of:
[KenHino/radicalpair-tensornetwork](https://github.com/KenHino/radicalpair-tensornetwork).
See [LICENSE](LICENSE). This directory is carved out of the proprietary
licence covering the rest of the repository.
