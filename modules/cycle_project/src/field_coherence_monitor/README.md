# FIELD_COHERENCE_MONITOR

**cycle_project / Module 4** — Real-time geomagnetic field coherence dashboard.

Connects the geological past (Laschamp excursion 41 ka, Younger Dryas 12.9 ka)  
to present-day geomagnetic field state via live NOAA SWPC data.

---

## Dashboard panels

| Panel | Source | Signal |
|-------|--------|--------|
| **Coherence Index** | Composite | 0.0 = Laschamp-like · 1.0 = stable |
| **Kp Index** | NOAA SWPC (live JSON) | Geomagnetic disturbance, 3h intervals |
| **F10.7 Solar Flux** | NOAA SWPC (live JSON) | Solar activity, monthly |
| **CR Proxy** | Inverse F10.7 (normalized) | Cosmic ray exposure estimate |
| **Pole Drift** | WMM 2000–2025 (NOAA NCEI) | Magnetic north latitude trend |
| **Laschamp Bar** | Composite | % toward Laschamp event conditions |

### Real cosmic ray data
→ Oulu Neutron Monitor: http://cosmicrays.oulu.fi  
→ NMDB database: https://www.nmdb.eu

---

## Build & run

```bash
# From cycle_project/src/field_coherence_monitor/
cargo build --release       # ~3-5 min first build (downloads egui deps)
cargo run --release         # live data mode
cargo run --release -- --offline  # use cached data only
```

**Requirements:** Rust 1.75+ (`rustup update stable`)

First run fetches live data from NOAA SWPC and caches it locally.  
The dashboard auto-refreshes every 15 minutes.

---

## Coherence score formula

```
disturbance = 0.40 × Kp_norm + 0.40 × CR_proxy + 0.20 × drift_norm
coherence   = 1.0 − disturbance

Laschamp conditions → coherence ≈ 0.0
Modern quiet-time  → coherence ≈ 0.85–0.95
```

### Laschamp reference (41,000 BP)
- VADM = 15–25% of reference (8 × 10²² Am²)  
- Kp_effective ≈ 9 (aurora visible at equatorial latitudes)  
- Be-10 spike: +300% above baseline (→ GRIP_Be10 anomaly in CYCLE_DETECT)  
- Pole drift: chaotic / undefined  

**Correlation with POLE_SHIFT_SIM:** VADM at Laschamp = 2.70 → force amplification 2.96×  
→ Peak lithosphere velocity amplification 1.394× at 39–40 ka BP

---

## NOAA SWPC endpoints

```
Solar cycle indices:
  https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json

Planetary Kp-index:
  https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json

WMM pole position updates:
  https://www.ncei.noaa.gov/products/world-magnetic-model
```

All endpoints are public, no API key required.

---

## Integration with cycle_project pipeline

```
CYCLE_DETECT         → detects anomaly at Laschamp (score 0.013) and YD (0.011)
POLE_SHIFT_SIM       → quantifies 2.96× force amplification during VADM collapse  
MYTH_RAG             → 21 cultures with flood/fire myths near geological events
FIELD_COHERENCE_MON  → answers: is anything similar happening NOW?
```
