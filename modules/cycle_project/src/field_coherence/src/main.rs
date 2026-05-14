// cycle_project/src/field_coherence/src/main.rs
//
// Real-time egui dashboard:
//   - Geomagnetic north pole acceleration (NOAA WMM)
//   - Solar flux F10.7 (NOAA SWPC)
//   - Cosmic ray flux (Oulu Neutron Monitor)
//
// Build:  cargo build --release
// Run:    ./target/release/field_coherence

use anyhow::Result;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const RING_BUFFER_SIZE: usize = 1440; // 24 h @ 1-min resolution
const POLL_INTERVAL_S: u64 = 3600;   // 1 h for remote feeds

// ─── Shared state ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct FieldState {
    pub f107:       VecDeque<(f64, f64)>, // (timestamp_unix, sfu)
    pub cosmic_ray: VecDeque<(f64, f64)>, // (timestamp_unix, counts/min)
    pub pole_lat:   f64,
    pub pole_lon:   f64,
    pub pole_speed: f64,                  // km/yr
    pub last_update: u64,
}

impl FieldState {
    fn push_f107(&mut self, ts: f64, val: f64) {
        if self.f107.len() >= RING_BUFFER_SIZE {
            self.f107.pop_front();
        }
        self.f107.push_back((ts, val));
    }

    fn push_cosmic(&mut self, ts: f64, val: f64) {
        if self.cosmic_ray.len() >= RING_BUFFER_SIZE {
            self.cosmic_ray.pop_front();
        }
        self.cosmic_ray.push_back((ts, val));
    }
}

// ─── Data fetchers (blocking, run in background thread) ───────────────────────

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Fetch NOAA SWPC F10.7 cm flux (latest 3 days JSON)
fn fetch_f107() -> Result<Vec<(f64, f64)>> {
    let url = "https://services.swpc.noaa.gov/json/f107_cm_flux.json";
    let body = reqwest::blocking::get(url)?.text()?;
    let parsed: serde_json::Value = serde_json::from_str(&body)?;
    let mut out = vec![];
    if let Some(arr) = parsed.as_array() {
        for item in arr {
            // format: {"time_tag": "2024-01-01 00:00:00", "flux": 120.5, ...}
            let flux = item["flux"].as_f64().unwrap_or(f64::NAN);
            // Use sequential index as timestamp proxy (real: parse time_tag)
            out.push((now_unix() as f64, flux));
        }
    }
    Ok(out)
}

/// Fetch Oulu cosmic-ray count rate (simplified — real impl needs HTML parse)
fn fetch_cosmic_rays() -> Result<f64> {
    // Oulu NM provides data at http://cosmicrays.oulu.fi/
    // For a real implementation parse their CGI output.
    // Stub returns NaN until proper parser is implemented.
    Ok(f64::NAN)
}

/// Background polling thread
fn poll_thread(state: Arc<Mutex<FieldState>>) {
    loop {
        let ts = now_unix() as f64;

        // F10.7
        match fetch_f107() {
            Ok(points) => {
                let mut s = state.lock().unwrap();
                for (_, v) in &points {
                    s.push_f107(ts, *v);
                }
                s.last_update = now_unix();
            }
            Err(e) => eprintln!("[poll] F10.7 error: {e}"),
        }

        // Cosmic rays
        match fetch_cosmic_rays() {
            Ok(v) => {
                let mut s = state.lock().unwrap();
                s.push_cosmic(ts, v);
            }
            Err(e) => eprintln!("[poll] Cosmic ray error: {e}"),
        }

        thread::sleep(Duration::from_secs(POLL_INTERVAL_S));
    }
}

// ─── egui App ─────────────────────────────────────────────────────────────────

struct FieldApp {
    state: Arc<Mutex<FieldState>>,
}

impl FieldApp {
    fn new(state: Arc<Mutex<FieldState>>) -> Self {
        Self { state }
    }
}

impl eframe::App for FieldApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Repaint every 5 s
        ctx.request_repaint_after(Duration::from_secs(5));

        let state = self.state.lock().unwrap().clone();

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("🌍 FIELD_COHERENCE MONITOR — cycle_project");

            ui.separator();
            ui.columns(3, |cols| {
                cols[0].group(|ui| {
                    ui.label("Geomagnetic Pole");
                    ui.monospace(format!("Lat  {:.2}°N", state.pole_lat));
                    ui.monospace(format!("Lon  {:.2}°E", state.pole_lon));
                    ui.monospace(format!("Speed {:.1} km/yr", state.pole_speed));
                });
                cols[1].group(|ui| {
                    ui.label("Solar F10.7");
                    let last_f107 = state.f107.back().map(|(_, v)| *v).unwrap_or(f64::NAN);
                    ui.monospace(format!("{:.1} sfu", last_f107));
                });
                cols[2].group(|ui| {
                    ui.label("Cosmic Rays (Oulu NM)");
                    let last_cr = state.cosmic_ray.back().map(|(_, v)| *v).unwrap_or(f64::NAN);
                    ui.monospace(format!("{:.1} cpm", last_cr));
                });
            });

            ui.separator();

            // F10.7 plot
            ui.label("Solar Flux F10.7 (sfu)");
            let f107_points: PlotPoints = state
                .f107
                .iter()
                .enumerate()
                .map(|(i, (_, v))| [i as f64, *v])
                .collect();
            Plot::new("f107_plot")
                .height(150.0)
                .show(ui, |plot| plot.line(Line::new(f107_points).name("F10.7")));

            ui.separator();

            // Cosmic ray plot
            ui.label("Cosmic Ray Flux (counts/min)");
            let cr_points: PlotPoints = state
                .cosmic_ray
                .iter()
                .enumerate()
                .filter(|(_, (_, v))| !v.is_nan())
                .map(|(i, (_, v))| [i as f64, *v])
                .collect();
            Plot::new("cosmic_plot")
                .height(150.0)
                .show(ui, |plot| plot.line(Line::new(cr_points).name("Cosmic rays")));

            ui.separator();
            let last_update = state.last_update;
            ui.label(format!("Last update: unix ts {last_update}"));
        });
    }
}

// ─── Entry point ──────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let state = Arc::new(Mutex::new(FieldState::default()));
    let state_bg = Arc::clone(&state);

    // Background polling thread
    thread::spawn(move || poll_thread(state_bg));

    // egui window
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("FIELD_COHERENCE — cycle_project")
            .with_inner_size([1200.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "field_coherence",
        options,
        Box::new(|_cc| Ok(Box::new(FieldApp::new(state)))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))?;

    Ok(())
}
