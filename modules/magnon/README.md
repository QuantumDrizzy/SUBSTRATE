# MAGNON

RF noise characterization of avian magnetic compass (cryptochrome radical pairs) via Lindblad master equation. Measures how urban EM environment degrades quantum coherence in the FAD•⁻—TrpH•⁺ system. Related to [CryptoTN-GPU](https://github.com/QuantumDrizzy/CryptoTN-GPU) (same phenomenon, GPU tensor network approach).

---

## what exists

| component | file | status |
|-----------|------|--------|
| Lindblad solver | `terra/quantum/radical_pair.py` | complete — 4D two-electron Hilbert space, Euler integration, 11 observables |
| Sensor bridge | `terra/sensors/noise_tensor.py` | complete — RTL-SDR or synthetic fallback, NOAA SWPC fetch, PSD → H_noise |
| Database layer | `terra/db.py` | complete — SQLite WAL, 5-table schema, SHA-256 audit chain |
| Orchestrator | `main.py` | complete — 1 Hz loop, SHM write, audit log |
| Rust dashboard | `rust/` | complete — egui 0.31, Bloch sphere, fidelity timeline, reads SHM at 60 FPS |
| IPC protocol | `rust/terra_protocol.rs` | complete — 80-byte binary struct (little-endian) |
| Schema | `migrations/0001_initial_schema.sql` | complete |
| Database | `data/terra_qci.sqlite` | active — 483 MB of real simulation runs |
| GPU Lindblad (cuTensorNet) | — | **stub** — cupy in deps, zero calls in code |
| Hyperfine expansion | — | **stub** — 4D only, full 324D FAD model not started |
| Web dashboard (Plotly/Dash) | — | **stub** — deps installed, no implementation |

~48% complete by original scope. Core physics pipeline runs end-to-end.

---

## the problem

Cryptochrome-based magnetoreception depends on coherent singlet-triplet interconversion in a radical pair. The FAD•⁻—TrpH•⁺ pair behaves as a compass because Earth's field (~50 µT) mixes spin states at the Larmor frequency (~1.4 MHz for electrons). RF noise at comparable frequencies drives random S↔T transitions and degrades directional sensitivity.

TERRA-QCI quantifies this degradation: given a measured RF environment, how much does the compass fail?

---

## pipeline

```
RTL-SDR (RTL2832U) ──────────────────────────────────────────────────────┐
  or synthetic noise                                                       │
  (urban EM: FM, TETRA, 5G, power line harmonics)                        │
                                                                           │
         PSD (Welch, nperseg=4096)                                        │
           │                                                               │
           ▼                                                               │
NOAA SWPC ──→ B_earth [24.5, 0.5, −39.0] µT                              │
  Kp index       (Aljucer, Murcia; fallback if offline)                   │
                                                                           │
           ▼                                                               │
     noise_tensor.py                                                       │
       PSD → B_noise_rms = √(2µ₀ · S_psd · Δf) / c                       │
       B_noise → H_noise (4×4 Hermitian, Zeeman coupling)                 │
                                                                           │
           ▼                                                               │
     radical_pair.py                                                       │
       H = H_Zeeman(B_earth) + H_exchange + H_noise                       │
       dρ/dt = -i[H,ρ] + Σ γ_k (L_k ρ L_k† − ½{L_k†L_k, ρ})            │
       ρ(0) = |S⟩⟨S|  →  ρ(1 µs)  in 1 ns steps                         │
                                                                           │
       Lindblad operators:                                                 │
         L₁ = √k_S · P_S     (singlet recombination, k_S = 1 MHz)        │
         L₂ = √k_T · P_T     (triplet recombination)                      │
         L₃ = √γ_T₂ · σ_z¹  (spin-spin dephasing, T₂⁻¹ = 0.5 MHz)      │
         L₄ = √γ_T₂ · σ_z²                                               │
                                                                           │
       observables: singlet yield, fidelity F(ρ_clean, ρ_noisy),         │
         coherence magnitude, T₂_eff, purity, von Neumann entropy,        │
         Bloch vector, compass sensitivity Δ(Φ_S) at θ=45°, NPR          │
                                                                           ▼
                                                                    db.py (SQLite)
                                                                    terra_qci.shm (80 bytes)
                                                                           │
                                                                           ▼
                                                                    Rust/egui dashboard
                                                                      Bloch sphere
                                                                      fidelity timeline
                                                                      COHERENT / DECOHERING
                                                                      / COLLAPSED status
```

Loop cadence: 1 Hz (Python), 60 FPS (dashboard reads SHM independently).

---

## physics parameters

Hyperfine coupling constants (Maeda 2008, Hiscock 2016):

| nucleus | A (MHz) |
|---------|---------|
| N5 (isoalloxazine) | 1.09 |
| N10 | 0.61 |
| N1 | 0.14 |
| N3 | 0.08 |

Currently not integrated into the Hamiltonian — solver is pure two-electron (4D). Full model requires 4 × 3⁴ = 324D Hilbert space via tensor product with ¹⁴N spins. That's the GPU branch.

Fidelity threshold: F < 0.7 → COLLAPSED. F ∈ [0.7, 0.9] → DECOHERING.

---

## stack

Python 3.10 · NumPy · SciPy · pyrtlsdr · requests · SQLite (WAL) · Rust 2021 · egui 0.31 · eframe · memmap2 · cupy-cuda12x (dep only, unused)

---

## build

**Python:**
```bash
pip install -e .
# or minimal
pip install numpy scipy pyrtlsdr requests
```

**Rust dashboard:**
```bash
cd rust
cargo build --release
```

Schema applies automatically on first run via `main.py`. No migration tool needed.

---

## run

```bash
# Python pipeline — 1 Hz loop, writes to data/terra_qci.sqlite and data/terra_qci.shm
python main.py

# Dashboard — separate terminal, reads SHM at 60 FPS
./rust/target/release/terra-qci-dashboard
```

RTL-SDR dongle optional. Without hardware, `noise_tensor.py` synthesizes a realistic urban EM environment (FM broadcast, TETRA pulsed at 17.65 Hz, 5G NR sub-6, 50 Hz power line harmonics, burst interference). NOAA fetch optional — falls back to nominal Aljucer field values.

---

## file tree

```
MAGNON/
├── main.py                          orchestrator loop
├── pyproject.toml
├── data/
│   ├── terra_qci.sqlite             483 MB, active runs
│   ├── terra_qci.sqlite-shm
│   ├── terra_qci.sqlite-wal
│   └── terra_qci.shm                80-byte IPC state
├── migrations/
│   └── 0001_initial_schema.sql
├── terra/
│   ├── db.py
│   ├── quantum/
│   │   └── radical_pair.py          Lindblad solver
│   └── sensors/
│       └── noise_tensor.py          SDR bridge + NOAA
└── rust/
    ├── Cargo.toml
    ├── Cargo.lock
    └── src/
        ├── main.rs                  egui app
        ├── terra_protocol.rs        80-byte SHM struct
        ├── bloch_widget.rs          Bloch sphere widget
        └── fidelity_plot.rs         timeline chart
```

---

## related

- [CryptoTN-GPU](https://github.com/QuantumDrizzy/CryptoTN-GPU) — GPU tensor networks for quantum biology (arXiv). Same radical pair physics, tensor network approach for large Hilbert spaces.
- [Q-NAA](https://github.com/QuantumDrizzy/Q-NAA) — Quantum Neural Attention Analyzer.

---

## what's next

- GPU Lindblad via cuTensorNet — expand to 324D with full hyperfine coupling
- Stochastic master equation — time-varying H_noise (TETRA framing, burst events)
- Dual-polarization SDR — replace isotropic B_noise assumption with polarization-resolved field
- Adaptive integration — replace Euler with RK45 or Krylov subspace methods

---

*Antonio Zambudio Rodríguez (QuantumDrizzy) · research software engineer*
