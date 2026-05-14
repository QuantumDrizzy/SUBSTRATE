# cycle_project — Architecture

> Investigates the crustal displacement / catastrophic cycle hypothesis
> (Younger Dryas ~12,900 BP; Gothenburg paleomagnetic excursion; Be-10 production
> spikes; global flood-myth convergence) using public geological data and
> GPU-accelerated ML.

---

## Directory Tree

```
cycle_project/
├── ARCHITECTURE.md
├── pyproject.toml                    # Python workspace (uv / pip)
├── Cargo.toml                        # Rust workspace (field_coherence)
├── configs/
│   └── data_sources.yaml             # All URLs + checksums
├── data/
│   ├── raw/                          # Downloaded originals (gitignored)
│   └── processed/                    # Parquet + PNG outputs
├── notebooks/
│   ├── 01_proxy_eda.ipynb
│   ├── 02_gnn_training.ipynb
│   └── 03_lbm_validation.ipynb
├── scripts/
│   ├── run_fetch.sh
│   ├── run_gnn.sh
│   └── ingest_myths.sh
└── src/
    ├── cycle_detect/                  # MODULE 1
    │   ├── __init__.py
    │   ├── fetch_data.py              # ← delivered here
    │   └── gnn_prototype.py           # ← delivered here
    ├── pole_shift_sim/                # MODULE 2
    │   ├── __init__.py
    │   ├── lbm_lithosphere.py         # Lattice Boltzmann sim (CuPy/CUDA)
    │   ├── gauge_coupling.py          # U(1) asthenosphere viscosity field
    │   └── visualize.py               # PyVista 3-D render pipeline
    ├── myth_rag/                      # MODULE 3
    │   ├── __init__.py
    │   ├── ingest.py                  # Corpus → ChromaDB
    │   ├── rag_agent.py               # RAG query + geological correlation
    │   └── corpus/                    # Plain-text myth sources (CC/PD)
    └── field_coherence/               # MODULE 4 (Rust)
        ├── Cargo.toml
        └── src/
            ├── main.rs                # egui dashboard entry point
            ├── noaa_api.rs            # NOAA Space Weather API poller
            ├── pole_tracker.rs        # Geomagnetic north acceleration
            └── cosmic_ray.rs          # Oulu / IZMIRAN cosmic-ray feed
```

---

## Module 1 — CYCLE_DETECT

**Goal:** GNN over 5 paleoclimate proxies → detect synchronous anomaly windows
indicative of a recurrent ~11,500–12,900 yr catastrophe cycle.

| Component | Choice | Justification |
|-----------|--------|---------------|
| Data fetch | `requests` + `urllib` + `tqdm` | No external dep; FTP redirect to HTTPS at NCEI |
| DataFrame | `pandas` + `numpy` | Standard; parquet via `pyarrow` |
| Graph ML | `torch_geometric` (PyG) | Best PyTorch-native temporal GNN ecosystem |
| GNN arch | `SAGEConv` autoencoder | Inductive; handles variable-density graphs |
| GPU | PyTorch `.to(device)` sm_120 | RTX 5060 Ti — needs `torch>=2.5` nightly for sm_120 |
| Anomaly | Reconstruction MSE per window | Unsupervised; no labeled YD events needed |
| Storage | Apache Parquet | Columnar; fast read for 150k rows × 5 proxies |

**Proxies ingested:**
1. GISP2 δ¹⁸O — Greenland temperature proxy (Alley 2000)
2. Vostok ΔT — Antarctic temperature (deuterium)
3. Vostok CO₂ — atmospheric greenhouse forcing
4. GRIP Be-10 flux — cosmic-ray / solar modulation proxy
5. Sint-2000 VADM — geomagnetic dipole moment stack

---

## Module 2 — POLE_SHIFT_SIM

**Goal:** 3-D Lattice Boltzmann simulation of lithospheric slab sliding over
a low-viscosity asthenosphere, driven by an asymmetric ice-load torque.
U(1) gauge coupling encodes the asthenosphere as a viscosity field.

| Component | Choice | Justification |
|-----------|--------|---------------|
| LBM kernel | CuPy (D3Q19) | Direct CUDA array ops; matches QUANTUM_LAB CUDA style |
| Gauge field | NumPy/CuPy U(1) link matrices | Re-uses HMC infrastructure from QUANTUM_LAB |
| Geometry | Spherical lat-lon grid (512×256×32) | Sufficient for continental-scale flow |
| Viz | PyVista + VTK | Interactive 3-D; export to .vtp for ParaView |
| Validation | Compare to Steinberger & O'Connell 1997 | Asthenosphere flow benchmark |

**Physics encoded:**
- Incompressible Navier-Stokes (LBM BGK collision)
- Yield-stress rheology for lithosphere (Bingham fluid)
- U(1) gauge field: asthenosphere viscosity η(x) as a scalar Wilson field
- Ice-sheet gravitational torque as external body force

---

## Module 3 — MYTH_RAG

**Goal:** ChromaDB vector store of primary flood/catastrophe myth texts
(Sumerian Atrahasis, Egyptian Papyrus of Ipuwer, Rigveda Manu, Norse Fimbulwinter,
Aztec Suns cosmogony, Mayan Popol Vuh, Greek Deucalion) + LangChain RAG agent
that queries geological events and retrieves thematically aligned myth passages.

| Component | Choice | Justification |
|-----------|--------|---------------|
| Vector DB | ChromaDB (local, persistent) | Already used in the orchestrator; sovereign, no cloud |
| Embeddings | `sentence-transformers` (all-mpnet-base-v2) | Best semantic similarity for ancient texts |
| LLM backend | Ollama (llama3/mistral local) | Matches the orchestrator stack; no API dependency |
| RAG orchestration | LangChain (minimal) | Familiar; swap for raw similarity search if needed |
| Corpus | Plain-text from Project Gutenberg + ETCSL | Public domain; licensed for research |
| Output | Markdown report: myth ↔ geological event correlation table | |

**Correlation targets:**
- 12,900 BP → YD onset: abrupt cooling, Atrahasis flood narrative
- 11,700 BP → YD termination: civilizational restart myths
- 74,000 BP → Toba event: population bottleneck in Hindu cosmogony
- ~41,000 BP → Laschamps excursion: aurora/sky-fire myths

---

## Module 4 — FIELD_COHERENCE_MONITOR

**Goal:** Real-time Rust + egui dashboard showing geomagnetic pole acceleration,
solar flux (F10.7), and cosmic-ray flux from Oulu Neutron Monitor.

| Component | Choice | Justification |
|-----------|--------|---------------|
| GUI | `egui` + `eframe` | Immediate-mode; no X11 dep; renders on bare metal |
| HTTP polling | `reqwest` (tokio async) | Non-blocking; update every 60s |
| Geomagnetic pole | NOAA NCEI WMM API | `https://www.ngdc.noaa.gov/geomag-web/calculators/calculateDeclination` |
| Solar activity | NOAA SWPC JSON | `https://services.swpc.noaa.gov/json/f107_cm_flux.json` |
| Cosmic rays | Oulu NM FTP | `http://cosmicrays.oulu.fi/` (hourly count rates) |
| Plotting | `egui_plot` | Real-time streaming chart; ring buffer 24h |
| Persistence | SQLite via `rusqlite` | Local timeseries storage; no cloud |

**Alerts:** Configurable threshold triggers (Ω > 2σ from 30-day mean)
emit desktop notification via `notify-rust`.

---

## Data Sources (exact paths)

```yaml
# configs/data_sources.yaml

gisp2_d18o:
  url: "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/isotopes/gisp2_d18o_accum_alley2000.txt"
  description: "GISP2 δ18O + accumulation rate, Alley 2000, 0–110,000 yr BP"
  verified: true

vostok_deuterium:
  url: "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/deutnat.txt"
  description: "Vostok deuterium + ΔT, Petit et al. 1999, 0–420,000 yr BP"
  verified: true

vostok_co2:
  url: "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/co2nat.txt"
  description: "Vostok CO2, Petit et al. 1999, 0–420,000 yr BP"
  verified: true

grip_be10:
  url: "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/beryllium/grip_be10_muscheler2004.txt"
  description: "GRIP Be-10 flux, Muscheler et al. 2004, 0–110,000 yr BP"
  verified: false  # ⚠️ browse dir to confirm filename
  fallback_dir: "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/beryllium/"

sint2000:
  url: "https://www.ngdc.noaa.gov/geomag/paleo_mag_datasets/Sint-2000.txt"
  description: "Sint-2000 VADM stack, Valet et al. 2005, 0–2,000,000 yr BP at 1-ka res"
  verified: false  # ⚠️ fallback: PANGAEA DOI:10.1594/PANGAEA.186810
  fallback_pangaea: "https://doi.pangaea.de/10.1594/PANGAEA.186810"

# Additional (Phase 2+)
sea_level_spratt:
  url: "https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/spratt2016/spratt2016.txt"
  description: "Sea level stack, Spratt & Lisiecki 2016, 0–800,000 yr BP"
  verified: true

lisiecki_lr04:
  url: "https://www.ncei.noaa.gov/pub/data/paleo/contributions_by_author/lisiecki2005/lisiecki2005.txt"
  description: "LR04 benthic δ18O stack, Lisiecki & Raymo 2005"
  verified: true
```

---

## 4-Phase Implementation Roadmap

### Phase 1 — Data Foundation (2–3 weeks)
- [x] Project scaffold + `pyproject.toml`
- [ ] `fetch_data.py`: download all proxies, clean, resample 100-yr grid
- [ ] Manual verification of ⚠️ URLs; fallback downloads if needed
- [ ] `gnn_prototype.py`: baseline GraphSAGE autoencoder, anomaly timeline
- [ ] Notebook 01: EDA — confirm YD spike at 12,900 BP in all proxies
- Deliverable: `data/processed/overview.png` showing 5-proxy alignment

### Phase 2 — GNN + LBM Core (4–6 weeks)
- [ ] GNN: hyperparameter sweep (window size 1k–10k yr, edge threshold)
- [ ] GNN: compare anomaly scores to known events (YD, 8.2ka, Laschamps)
- [ ] LBM: D3Q19 kernel in CuPy; benchmark on RTX 5060 Ti
- [ ] LBM: U(1) viscosity coupling; polar ice-torque forcing
- [ ] `visualize.py`: PyVista animation of slab displacement
- Deliverable: LBM simulation video + GNN anomaly heatmap

### Phase 3 — RAG + Real-Time Monitor (3–4 weeks)
- [ ] Corpus ingestion: 7 myth traditions → ChromaDB
- [ ] RAG agent: geological event → myth passage retrieval
- [ ] Correlation report: markdown table (event × myth × similarity score)
- [ ] Rust `field_coherence`: egui dashboard compiling on Arch bare metal
- [ ] Live data feeds: NOAA SWPC + Oulu NM + NCEI WMM
- Deliverable: Running dashboard + myth correlation report

### Phase 4 — Integration + Publication (2–3 weeks)
- [ ] Cross-module: GNN anomaly dates → RAG queries → myth passages
- [ ] Statistical validation: bootstrap significance test on synchrony scores
- [ ] Full 150,000-yr multi-proxy animation (PyVista + matplotlib)
- [ ] LaTeX report / preprint draft
- [ ] Open dataset release (processed parquets on Zenodo)
- Deliverable: Reproducible research archive

**Total estimated time: 11–16 weeks** (solo researcher, part-time alongside QUANTUM_LAB)

---

## GPU Notes (RTX 5060 Ti — sm_120)

```toml
# Requires torch >= 2.5 with sm_120 support
# Install nightly if stable doesn't build for Ada Lovelace successor arch:
# pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu124

# CuPy: build from source or use cupy-cuda12x ≥ 13.x
# CUDA-Q / PennyLane: sm_120 PTX should JIT-compile fine
```

All CUDA kernels in POLE_SHIFT_SIM target `sm_120` via `nvcc -arch=sm_120`.
PyTorch ops fall back to CPU automatically if sm_120 is not yet supported
in the installed build (guarded by `device = torch.device("cuda" if
torch.cuda.is_available() else "cpu")`).
