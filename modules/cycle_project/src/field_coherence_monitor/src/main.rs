//! FIELD_COHERENCE_MONITOR — cycle_project module 4
//!
//! Real-time dashboard connecting the geological past (Laschamp, Younger Dryas)
//! to present-day geomagnetic field state.
//!
//! Data sources (all free, no API key):
//!   Solar:       NOAA SWPC  — observed solar cycle indices (monthly JSON)
//!   Geomagnetic: NOAA SWPC  — planetary Kp-index (3-hourly JSON)
//!   Pole drift:  WMM annual — hardcoded 2000-2025, NOAA NCEI
//!   Cosmic ray:  Derived proxy from F10.7 (anticorrelated)
//!                → Real flux at nmdb.eu / Oulu (cosmicrays.oulu.fi)
//!
//! Usage:
//!   cargo run --release
//!   cargo run --release -- --offline   # skip fetch, use cache only

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::{Arc, RwLock};

mod app;
mod coherence;
mod fetcher;
mod probe_state;

fn main() -> eframe::Result<()> {
    // Shared state: updated by background thread, read by egui
    let state = Arc::new(RwLock::new(fetcher::DataState::default()));

    // Seed with WMM pole history (always available, no network needed)
    {
        let mut s = state.write().unwrap();
        s.pole = fetcher::wmm_pole_history();
        s.fetch_msg = "Starting…".to_string();
    }

    // Load cached data (enables offline use from last session)
    let args: Vec<String> = std::env::args().collect();
    let offline = args.iter().any(|a| a == "--offline");

    if !offline {
        let cache_path = fetcher::default_cache_path();
        if let Ok(cached) = fetcher::load_cache(&cache_path) {
            let mut s = state.write().unwrap();
            if !cached.solar.is_empty() { s.solar = cached.solar; }
            if !cached.kp.is_empty()    { s.kp    = cached.kp;    }
            s.last_fetch = cached.last_fetch;
            s.fetch_ok   = cached.fetch_ok;
            s.fetch_msg  = format!("[cached] {}", cached.fetch_msg);
        }
        fetcher::start_background_fetch(Arc::clone(&state));
    } else {
        let mut s = state.write().unwrap();
        s.fetch_msg = "Offline mode — using cache only".to_string();
        s.fetch_ok  = true;
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("⊕  FIELD COHERENCE MONITOR  |  cycle_project")
            .with_inner_size([1300.0, 840.0])
            .with_min_inner_size([900.0, 640.0]),
        ..Default::default()
    };

    eframe::run_native(
        "field_coherence_monitor",
        native_options,
        Box::new(move |cc| Box::new(app::FieldCoherenceApp::new(cc, Arc::clone(&state)))),
    )
}
