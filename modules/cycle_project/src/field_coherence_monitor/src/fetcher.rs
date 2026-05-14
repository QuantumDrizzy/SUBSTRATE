//! Data fetching, caching, and shared state for FIELD_COHERENCE_MONITOR.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Duration;

// ── Data types ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SolarReading {
    pub time_tag: String,  // "YYYY-MM"
    pub ssn:  f64,         // smoothed sunspot number
    pub f107: f64,         // F10.7 solar flux index (sfu)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct KpReading {
    pub time_tag: String,  // "YYYY-MM-DD HH:MM:SS"
    pub kp: f64,           // planetary K-index, 0..9
}

/// Magnetic north pole position from NOAA WMM annual reports.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolePosition {
    pub year: f64,
    pub lat:  f64,   // degrees N
    pub lon:  f64,   // degrees E (negative = W)
}

/// Shared mutable application state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DataState {
    pub solar:      Vec<SolarReading>,
    pub kp:         Vec<KpReading>,
    pub pole:       Vec<PolePosition>,
    pub last_fetch: Option<String>,
    pub fetch_ok:   bool,
    pub fetch_msg:  String,
}

// ── WMM Magnetic North Pole History (2000–2025) ──────────────────────────────
// Source: NOAA NCEI World Magnetic Model
// https://www.ncei.noaa.gov/products/world-magnetic-model
//
// Note the dramatic acceleration: the pole drifted ~10 km/yr before 1990,
// ~55 km/yr by 2019. It crossed from Canadian Arctic toward Siberia ~2018.
pub fn wmm_pole_history() -> Vec<PolePosition> {
    vec![
        PolePosition { year: 2000.0, lat: 81.3, lon: -110.8 },
        PolePosition { year: 2001.0, lat: 81.6, lon: -111.6 },
        PolePosition { year: 2002.0, lat: 82.0, lon: -112.4 },
        PolePosition { year: 2003.0, lat: 82.3, lon: -113.4 },
        PolePosition { year: 2004.0, lat: 82.6, lon: -114.6 },
        PolePosition { year: 2005.0, lat: 83.0, lon: -115.9 },
        PolePosition { year: 2006.0, lat: 83.4, lon: -117.4 },
        PolePosition { year: 2007.0, lat: 83.7, lon: -118.8 },
        PolePosition { year: 2008.0, lat: 84.0, lon: -120.4 },
        PolePosition { year: 2009.0, lat: 84.4, lon: -122.7 },
        PolePosition { year: 2010.0, lat: 84.9, lon: -130.0 },
        PolePosition { year: 2011.0, lat: 85.1, lon: -133.0 },
        PolePosition { year: 2012.0, lat: 85.5, lon: -140.0 },
        PolePosition { year: 2013.0, lat: 85.9, lon: -147.0 },
        PolePosition { year: 2014.0, lat: 86.2, lon: -153.0 },
        PolePosition { year: 2015.0, lat: 86.3, lon: -160.1 },
        PolePosition { year: 2016.0, lat: 86.5, lon: -165.0 },
        PolePosition { year: 2017.0, lat: 86.7, lon: -170.0 },
        PolePosition { year: 2018.0, lat: 86.9, lon: -175.0 },
        PolePosition { year: 2019.0, lat: 87.1, lon:  175.5 }, // crossed antimeridian
        PolePosition { year: 2020.0, lat: 87.2, lon:  170.0 },
        PolePosition { year: 2021.0, lat: 87.3, lon:  164.0 },
        PolePosition { year: 2022.0, lat: 87.4, lon:  158.0 },
        PolePosition { year: 2023.0, lat: 87.5, lon:  153.0 },
        PolePosition { year: 2024.0, lat: 87.6, lon:  147.0 },
        PolePosition { year: 2025.0, lat: 87.7, lon:  142.0 }, // WMM-2025 estimate
    ]
}

// ── NOAA SWPC endpoints ───────────────────────────────────────────────────────

const SOLAR_URL: &str =
    "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json";

const KP_URL: &str =
    "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json";

// ── Fetch: solar cycle indices ────────────────────────────────────────────────

pub fn fetch_solar(n_months: usize) -> Result<Vec<SolarReading>> {
    #[derive(Deserialize)]
    struct Raw {
        #[serde(rename = "time-tag")]
        time_tag: String,
        smoothed_ssn: Option<f64>,
        #[serde(rename = "f10.7")]
        f107: Option<f64>,
    }

    let resp: Vec<Raw> = ureq::get(SOLAR_URL)
        .timeout(Duration::from_secs(20))
        .call()?
        .into_json()?;

    // Keep last N months where both SSN and F10.7 are present
    let mut filtered: Vec<Raw> = resp.into_iter().filter(|r| r.f107.is_some()).collect();
    filtered.reverse();
    filtered.truncate(n_months);
    filtered.reverse();
    let readings: Vec<SolarReading> = filtered
        .into_iter()
        .map(|r| SolarReading {
            time_tag: r.time_tag,
            ssn:  r.smoothed_ssn.unwrap_or(0.0),
            f107: r.f107.unwrap_or(70.0),
        })
        .collect();

    if readings.is_empty() {
        anyhow::bail!("No valid solar readings returned");
    }
    Ok(readings)
}

// ── Fetch: planetary Kp index ────────────────────────────────────────────────

pub fn fetch_kp() -> Result<Vec<KpReading>> {
    // Returns array of arrays; first row is header:
    //   ["time_tag", "Kp", "a_running", "station_list"]
    // Data rows: ["2025-01-01 00:00:00", "2.00", "15", "station..."]
    // Kp = -1 means no data — skip those.
    let resp: Vec<Vec<serde_json::Value>> = ureq::get(KP_URL)
        .timeout(Duration::from_secs(20))
        .call()?
        .into_json()?;

    let readings: Vec<KpReading> = resp
        .into_iter()
        .skip(1) // skip header
        .filter_map(|row| {
            let time_tag = row.first()?.as_str()?.to_string();
            let kp = row.get(1)?.as_str()?.parse::<f64>().ok()?;
            if kp < 0.0 { return None; } // -1 = missing
            Some(KpReading { time_tag, kp })
        })
        .collect();

    if readings.is_empty() {
        anyhow::bail!("No valid Kp readings returned");
    }
    Ok(readings)
}

// ── Cache ────────────────────────────────────────────────────────────────────

pub fn default_cache_path() -> PathBuf {
    // Place cache next to the binary (e.g. target/debug/ during dev)
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("field_coherence_cache.json")))
        .unwrap_or_else(|| PathBuf::from("field_coherence_cache.json"))
}

pub fn load_cache(path: &PathBuf) -> Result<DataState> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

pub fn save_cache(state: &DataState, path: &PathBuf) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(state)?)?;
    Ok(())
}

// ── Background refresh thread ─────────────────────────────────────────────────

/// Spawns a thread that fetches NOAA data every 15 minutes.
/// Updates `shared` in place; keeps existing data on partial failures.
pub fn start_background_fetch(shared: Arc<RwLock<DataState>>) {
    let cache_path = default_cache_path();

    std::thread::spawn(move || loop {
        let now = chrono::Utc::now()
            .format("%Y-%m-%dT%H:%M:%SZ")
            .to_string();

        let solar_res = fetch_solar(72); // 6 years monthly
        let kp_res    = fetch_kp();

        {
            let mut s = shared.write().unwrap();

            match solar_res {
                Ok(solar) if !solar.is_empty() => {
                    s.solar    = solar;
                    s.fetch_ok = true;
                }
                Err(e) => {
                    s.fetch_ok  = false;
                    s.fetch_msg = format!("Solar fetch failed: {e}");
                }
                _ => {}
            }

            match kp_res {
                Ok(kp) if !kp.is_empty() => {
                    s.kp = kp;
                }
                Err(e) => {
                    s.fetch_ok  = false;
                    s.fetch_msg = format!("Kp fetch failed: {e}");
                }
                _ => {}
            }

            if s.fetch_ok {
                s.fetch_msg = "Live — NOAA SWPC".to_string();
            }
            s.last_fetch = Some(now);

            // Always ensure WMM pole data is present
            if s.pole.is_empty() {
                s.pole = wmm_pole_history();
            }

            let _ = save_cache(&*s, &cache_path);
        }

        // Refresh every 15 minutes
        std::thread::sleep(Duration::from_secs(900));
    });
}
