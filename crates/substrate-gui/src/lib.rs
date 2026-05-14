//! substrate-gui — SUBSTRATE Quantum Workbench
//! iNFAMØUS OS · White + Red · Glassmorphism + Spectrogram

use eframe::egui;
use eframe::egui_wgpu;
use egui::{
    Align, CentralPanel, Color32, FontFamily, FontId, Layout, Margin,
    Pos2, Rect, RichText, Rounding, SidePanel, Stroke,
    TopBottomPanel, Vec2, Sense,
};
// egui_plot retained for future use
use substrate_core::{SubstrateEngine, TuiMsg};
use substrate_core::layers::LayerStatus as CoreStatus;
use substrate_gpu::{FieldCallback, FieldPipeline, FieldUniforms};
use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

// ══════════════════════ THEME — WHITE + RED ═══════════════════════════════ //

pub mod theme {
    use eframe::egui::Color32;

    pub const BG_BASE:    Color32 = Color32::from_rgb(230, 230, 242);
    pub const BG_GLASS:   Color32 = Color32::from_rgb(250, 250, 255);
    pub const BG_CARD:    Color32 = Color32::WHITE;
    pub const BG_DARK:    Color32 = Color32::from_rgb(8,   8,   16);
    pub const ACCENT:     Color32 = Color32::from_rgb(255, 32,  72);
    pub const BLUE_INFO:  Color32 = Color32::from_rgb(30,  120, 255);
    pub const GREEN_OK:   Color32 = Color32::from_rgb(0,   180, 90);
    pub const AMBER:      Color32 = Color32::from_rgb(220, 110,  0);
    pub const TEXT_DARK:  Color32 = Color32::from_rgb(12,  12,  24);
    pub const TEXT_MID:   Color32 = Color32::from_rgb(70,  70,  100);
    pub const TEXT_DIM:   Color32 = Color32::from_rgb(150, 150, 180);
    pub const BORDER:     Color32 = Color32::from_rgb(208, 208, 226);
    pub const BORDER_RED: Color32 = Color32::from_rgb(255, 180, 192);

    pub fn mesh_color(z: f64) -> Color32 {
        let t = z.clamp(0.0, 1.0) as f32;
        if t < 0.40 {
            let s = t / 0.40;
            Color32::from_rgb(lerp(50, 100, s), lerp(80, 50, s), lerp(210, 255, s))
        } else if t < 0.70 {
            let s = (t - 0.40) / 0.30;
            Color32::from_rgb(lerp(100, 220, s), lerp(50, 40, s), lerp(255, 150, s))
        } else {
            let s = (t - 0.70) / 0.30;
            Color32::from_rgb(lerp(220, 255, s), lerp(40, 32, s), lerp(150, 72, s))
        }
    }

    pub fn spec_color(p: f32) -> Color32 {
        let t = p.clamp(0.0, 1.0);
        if t < 0.15      { Color32::from_rgb((t / 0.15 * 80.0) as u8, 0, 0) }
        else if t < 0.45 { let s=(t-0.15)/0.30; Color32::from_rgb(lerp(80,255,s), 0, 0) }
        else if t < 0.70 { let s=(t-0.45)/0.25; Color32::from_rgb(255, lerp(0,180,s), 0) }
        else if t < 0.88 { let s=(t-0.70)/0.18; Color32::from_rgb(255, lerp(180,255,s), lerp(0,60,s)) }
        else             { let s=(t-0.88)/0.12; Color32::from_rgb(255, 255, lerp(60,220,s)) }
    }

    pub fn lerp(a: u8, b: u8, t: f32) -> u8 {
        (a as f32 + (b as f32 - a as f32) * t.clamp(0.0, 1.0)) as u8
    }
}

// ══════════════════════ DATA MODEL ════════════════════════════════════════ //

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LayerState { Idle, Running, Done, Error }

#[derive(Clone, Debug)]
pub struct LayerMetrics {
    pub name:       String,
    pub id:         String,
    pub state:      LayerState,
    pub score:      f64,
    pub latency_ms: f64,
    pub fields:     Vec<(String, String, Color32)>,
    pub history:    VecDeque<f64>,
    pub history2:   VecDeque<f64>,
    pub spec_data:  VecDeque<Vec<f32>>,
    pub has_real_data: bool,
}

impl LayerMetrics {
    pub fn new(id: &str, name: &str) -> Self {
        Self {
            name: name.into(), id: id.into(),
            state: LayerState::Idle, score: 0.0, latency_ms: 0.0,
            fields: vec![],
            history:   VecDeque::with_capacity(240),
            history2:  VecDeque::with_capacity(240),
            spec_data: VecDeque::with_capacity(120),
            has_real_data: false,
        }
    }
    pub fn push(&mut self, v: f64, v2: f64) {
        if self.history.len()  >= 240 { self.history.pop_front(); }
        if self.history2.len() >= 240 { self.history2.pop_front(); }
        self.history.push_back(v.clamp(0.0, 1.0));
        self.history2.push_back(v2.clamp(-1.0, 1.0));
        self.score = v.clamp(0.0, 1.0);
    }
    pub fn push_spec(&mut self, col: Vec<f32>) {
        if self.spec_data.len() >= 120 { self.spec_data.pop_front(); }
        self.spec_data.push_back(col);
    }
    pub fn state_color(&self) -> Color32 {
        match self.state {
            LayerState::Idle    => theme::TEXT_DIM,
            LayerState::Running => theme::BLUE_INFO,
            LayerState::Done    => theme::GREEN_OK,
            LayerState::Error   => theme::ACCENT,
        }
    }
    pub fn state_dot(&self) -> &'static str {
        match self.state {
            LayerState::Idle    => "○",
            LayerState::Running => "◐",
            LayerState::Done    => "●",
            LayerState::Error   => "◉",
        }
    }
}

#[derive(Clone, Debug)]
pub struct CoherenceEvent {
    pub pairs:    Vec<(String, String, f64)>, // (layer_a, layer_b, pearson_r)
    pub severity: f64,                        // pairs.len() / max_possible
    pub cycle:    u64,
}

#[derive(Clone)]
pub struct TelemetryFrame {
    pub layers:      Vec<LayerMetrics>,
    pub coherence:   f64,
    pub entropy:     f64,
    pub snr:         f64,
    pub latency_ms:  f64,
    pub field_nt:    f64,
    pub status:      String,
    pub has_real_data:  bool,
    pub is_simulating:  bool,
    pub corr_matrix: Vec<Vec<f64>>,   // N×N Pearson matrix (live)
    pub events:      VecDeque<CoherenceEvent>, // ring buffer of detected events
    pub cycle:       u64,
}

impl Default for TelemetryFrame {
    fn default() -> Self {
        let n = 10;
        Self {
            layers: vec![
                LayerMetrics::new("quantum",      "Quantum"),
                LayerMetrics::new("geomagnetic",  "Geomagnetic"),
                LayerMetrics::new("magnon",       "Magnon"),
                LayerMetrics::new("quantum_lab",  "Quantum Lab"),
                LayerMetrics::new("solar",        "Solar"),
                LayerMetrics::new("cosmological", "Cosmological"),
                LayerMetrics::new("eeg",          "EEG Biosensor"),
                LayerMetrics::new("lunar",        "Lunar"),
                LayerMetrics::new("radio",        "Radio / CMB"),
                LayerMetrics::new("seismic",      "Seismic"),
            ],
            coherence: 0.0, entropy: 0.0, snr: 0.0,
            latency_ms: 0.0, field_nt: 42.3,
            status: "IDLE".into(),
            has_real_data:  false,
            is_simulating:  true,
            corr_matrix: vec![vec![0.0f64; n]; n],
            events:      VecDeque::with_capacity(64),
            cycle:       0,
        }
    }
}

// ══════════════════════ CORRELATION ENGINE ════════════════════════════════ //

/// Pearson r between two score histories. Returns None if insufficient data.
fn pearson(a: &VecDeque<f64>, b: &VecDeque<f64>) -> Option<f64> {
    let n = a.len().min(b.len());
    if n < 8 { return None; }
    let av: Vec<f64> = a.iter().rev().take(n).cloned().collect();
    let bv: Vec<f64> = b.iter().rev().take(n).cloned().collect();
    let ma = av.iter().sum::<f64>() / n as f64;
    let mb = bv.iter().sum::<f64>() / n as f64;
    let num = av.iter().zip(bv.iter()).map(|(x,y)| (x-ma)*(y-mb)).sum::<f64>();
    let da  = av.iter().map(|x| (x-ma).powi(2)).sum::<f64>().sqrt();
    let db  = bv.iter().map(|y| (y-mb).powi(2)).sum::<f64>().sqrt();
    if da < 1e-10 || db < 1e-10 { return None; }
    Some((num / (da * db)).clamp(-1.0, 1.0))
}

/// Rebuild the N×N correlation matrix from live layer histories.
fn recompute_corr_matrix(frame: &mut TelemetryFrame) {
    let n = frame.layers.len();
    let mut mat = vec![vec![0.0f64; n]; n];
    for i in 0..n { mat[i][i] = 1.0; }
    for i in 0..n {
        for j in (i+1)..n {
            let r = pearson(&frame.layers[i].history, &frame.layers[j].history)
                .unwrap_or(0.0);
            mat[i][j] = r;
            mat[j][i] = r;
        }
    }
    // Detect coherence event: ≥3 pairs with |r| > 0.78
    let threshold = 0.78f64;
    let mut high: Vec<(String, String, f64)> = Vec::new();
    for i in 0..n {
        for j in (i+1)..n {
            if mat[i][j].abs() >= threshold {
                high.push((
                    frame.layers[i].id.clone(),
                    frame.layers[j].id.clone(),
                    mat[i][j],
                ));
            }
        }
    }
    if high.len() >= 3 {
        let max_pairs = n * (n - 1) / 2;
        let ev = CoherenceEvent {
            severity: high.len() as f64 / max_pairs as f64,
            pairs:    high,
            cycle:    frame.cycle,
        };
        if frame.events.len() >= 64 { frame.events.pop_front(); }
        frame.events.push_back(ev);
    }
    frame.corr_matrix = mat;
    frame.cycle += 1;
}

// ══════════════════════ ENGINE BRIDGE ═════════════════════════════════════ //

pub struct EngineBridge {
    pub frame:   Arc<Mutex<TelemetryFrame>>,
    pub running: Arc<Mutex<bool>>,
}

impl EngineBridge {
    pub fn new() -> Self {
        Self {
            frame:   Arc::new(Mutex::new(TelemetryFrame::default())),
            running: Arc::new(Mutex::new(false)),
        }
    }

    pub fn spawn_simulator(&self) {
        let frame   = Arc::clone(&self.frame);
        let running = Arc::clone(&self.running);
        *running.lock().unwrap() = true;

        std::thread::spawn(move || {
            let params: &[(f64, f64, f64)] = &[
                (0.71, 0.00, 0.38), // quantum
                (0.43, 1.20, 0.42), // geomagnetic
                (0.93, 2.10, 0.32), // magnon
                (0.62, 0.80, 0.35), // quantum_lab
                (0.28, 3.50, 0.28), // solar
                (0.19, 1.70, 0.22), // cosmological
                (1.10, 0.30, 0.45), // eeg
                (0.35, 0.50, 0.30), // lunar
                (0.55, 1.10, 0.25), // radio
                (0.85, 2.50, 0.40), // seismic
            ];
            let mut t = 0.0f64;
            loop {
                if !*running.lock().unwrap() { break; }
                std::thread::sleep(Duration::from_millis(75));
                t += 0.075;
                let mut f = frame.lock().unwrap();
                if !f.is_simulating {
                    // Engine is active, simulator must yield to preserve telemetry integrity
                    continue; 
                }
                if f.has_real_data {
                    break; 
                }
                let ids: Vec<String> = f.layers.iter().map(|l| l.id.clone()).collect();
                for (i, layer) in f.layers.iter_mut().enumerate() {
                    if layer.has_real_data {
                        continue; // Keep the stable real data trace perfectly intact!
                    }
                    let (freq, phase, amp) = params[i];
                    let s  = (0.50 + amp*(freq*t+phase).sin()
                             + 0.07*(freq*2.3*t+phase+1.0).sin()).clamp(0.0, 1.0);
                    let s2 = (freq*1.7*t+phase+0.5).cos() * 0.6;
                    layer.push(s, s2);
                    layer.latency_ms = 2.5 + 10.0*(t*0.7+i as f64).sin().abs();
                    layer.state = if t < 0.4 { LayerState::Running }
                                  else if s > 0.68 { LayerState::Done }
                                  else { LayerState::Running };
                    layer.fields    = make_fields(&ids[i], s, t);
                    layer.push_spec(make_spec_col(&ids[i], s, t));
                }
                let scores: Vec<f64> = f.layers.iter().map(|l| l.score).collect();
                f.coherence  = scores.iter().sum::<f64>() / scores.len() as f64;
                f.entropy    = 0.28 + 0.14*(t*0.38).sin();
                f.snr        = 20.0 + 10.0*(t*0.51).sin();
                f.latency_ms = f.layers.iter().map(|l| l.latency_ms).sum::<f64>()
                               / f.layers.len() as f64;
                f.field_nt   = 42.3 + 1.8*(t*0.19).sin();
                f.status     = if f.coherence > 0.60 { "COHERENT".into() }
                               else { "RUNNING".into() };
                // Recompute Pearson matrix every sim tick (every ~75ms)
                recompute_corr_matrix(&mut f);
            }
        });
    }

    /// Live engine: spawns the real Python FFI pipeline.
    /// `ctx` is cloned so the receiver thread can trigger repaints on LayerDone.
    /// Falls back gracefully — if engine fails the frame keeps the last value.
    pub fn spawn_live_engine(&self, ctx: egui::Context) {
        let frame   = Arc::clone(&self.frame);
        let running = Arc::clone(&self.running);
        *running.lock().unwrap() = true;

        std::thread::spawn(move || {
            let rt = tokio::runtime::Runtime::new().expect("tokio rt");

            loop {
                if !*running.lock().unwrap() { break; }

                // Mark all layers Running and disable simulation mode immediately
                {
                    let mut f = frame.lock().unwrap();
                    f.is_simulating = false; 
                    for lm in &mut f.layers { 
                        lm.state = LayerState::Running;
                        // Push a starting point to the timeline to ensure continuity
                        let s = lm.score;
                        lm.push(s, s * 2.0 - 1.0);
                    }
                    f.status = "RUNNING".into();
                }
                ctx.request_repaint();

                let (tx, rx) = std::sync::mpsc::channel::<TuiMsg>();
                let frame2   = Arc::clone(&frame);
                let ctx2     = ctx.clone();

                // Receiver thread — processes messages as layers complete
                let recv = std::thread::spawn(move || {
                    for msg in rx.iter() {
                        match msg {
                            TuiMsg::LayerStarted(layer) => {
                                let mut f = frame2.lock().unwrap();
                                if let Some(lm) = f.layers.iter_mut()
                                    .find(|l| l.id == layer.name())
                                {
                                    lm.state = LayerState::Running;
                                }
                                ctx2.request_repaint();
                            }
                            TuiMsg::LayerDone(result) => {
                                update_layer_from_result(&frame2, result);
                                ctx2.request_repaint();
                            }
                            TuiMsg::AllDone => break,
                        }
                    }
                });

                // Run full engine cycle (blocks until all layers complete)
                let engine = SubstrateEngine::new();
                let _ = rt.block_on(engine.run_streaming_filtered(tx, "all"));
                let _ = recv.join();

                if !*running.lock().unwrap() { break; }
                // Short pause between cycles before re-running
                std::thread::sleep(Duration::from_secs(3));
            }
        });
    }

    pub fn stop(&self) { *self.running.lock().unwrap() = false; }
}

// ══════════════════════ LIVE ENGINE HELPERS ═══════════════════════════════ //

fn update_layer_from_result(
    frame:  &Arc<Mutex<TelemetryFrame>>,
    result: substrate_core::layers::LayerResult,
) {
    let mut f = frame.lock().unwrap();
    f.has_real_data = true; // Signal simulator thread to gracefully stop colliding
    let id    = result.layer.name();

    if let Some(lm) = f.layers.iter_mut().find(|l| l.id == id) {
        if !lm.has_real_data {
            // First time receiving real data for this specific layer!
            // Let's clear the synthetic simulation history to prevent a sharp visual jump/glitch!
            lm.history.clear();
            lm.history2.clear();
            lm.has_real_data = true;
            
            // Push a "start" point at the current score to ensure egui_plot has 2 points and draws a segment
            lm.push(result.score, result.score * 0.8 - 0.4); 
        }
        let score   = result.score;
        let latency = result.metadata
            .pointer("/data/latency_ms")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);

        // Use a different mapping for secondary to prevent perfect overlap with the blue line
        lm.push(score, (score * 0.9 - 0.45).clamp(-1.0, 1.0)); 
        lm.latency_ms = latency;
        lm.state = match &result.status {
            CoreStatus::Done     => LayerState::Done,
            CoreStatus::Error(_) => LayerState::Error,
            _                    => LayerState::Done,
        };
        lm.fields    = extract_fields_real(id, &result.metadata);
        // Spec col driven by real score + deterministic time seed
        let t_seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);
        lm.push_spec(make_spec_col(id, score, t_seed));
    }

    // Collect done-layer data first (no borrow held after this block)
    let (scores, lats): (Vec<f64>, Vec<f64>) = {
        let done: Vec<_> = f.layers.iter()
            .filter(|l| l.state == LayerState::Done)
            .collect();
        let sc  = done.iter().map(|l| l.score).collect();
        let lat = done.iter().map(|l| l.latency_ms).filter(|&v| v > 0.0).collect();
        (sc, lat)
    };
    // Now mutate freely
    if !scores.is_empty() {
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        f.coherence = mean;

        if !lats.is_empty() {
            f.latency_ms = lats.iter().sum::<f64>() / lats.len() as f64;
        }

        if scores.len() > 1 {
            let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
                / (scores.len() - 1) as f64;
            let noise = variance.sqrt().max(1e-6);
            f.snr = 20.0 * (mean / noise).max(1e-6).log10();
        }

        // Normalized Shannon entropy: H = -Σ p·ln(p) / ln(N), p_i = score_i / Σscores
        let score_sum = scores.iter().sum::<f64>().max(1e-12);
        let ln_n = (scores.len() as f64).ln().max(1e-12);
        f.entropy = -scores.iter()
            .map(|&s| { let p = s / score_sum; if p > 1e-12 { p * p.ln() } else { 0.0 } })
            .sum::<f64>() / ln_n;
    }
    let done  = f.layers.iter().filter(|l| l.state == LayerState::Done).count();
    let total = f.layers.len();
    f.status = if done == total && f.coherence > 0.60 { "COHERENT".into() }
               else if done == total { "COMPLETE".into() }
               else { format!("{done}/{total} DONE") };

    // Recompute Pearson matrix every LayerDone
    recompute_corr_matrix(&mut f);
}

fn extract_fields_real(id: &str, meta: &serde_json::Value) -> Vec<(String, String, Color32)> {
    let data = if let Some(d) = meta.get("data") { d } else {
        // No data section — fall back to synthetic
        return make_fields(id, meta["score"].as_f64().unwrap_or(0.0), 0.0);
    };

    match id {
        "eeg" => vec![
            ("α_rel".into(),
             data.get("alpha_rel").and_then(|v|v.as_f64())
                 .or_else(|| data.pointer("/band_powers/alpha").and_then(|v|v.as_f64()))
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("θ_rel".into(),
             data.pointer("/band_powers/theta").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::TEXT_MID),
            ("Mode".into(),
             data["mode"].as_str().unwrap_or("?").into(),
             theme::AMBER),
            ("Latency".into(),
             data["latency_ms"].as_f64()
                 .map(|v| format!("{v:.1}ms")).unwrap_or_else(|| "?".into()),
             theme::TEXT_DIM),
        ],
        "geomagnetic" => vec![
            ("Kp_idx".into(),
             data.get("kp_index").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.1}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("Dst".into(),
             data.get("dst_nt").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.0} nT")).unwrap_or_else(|| "?".into()),
             theme::AMBER),
            ("Source".into(),
             data["source"].as_str().unwrap_or("cached").into(),
             theme::TEXT_MID),
            ("Latency".into(),
             data["latency_ms"].as_f64()
                 .map(|v| format!("{v:.1}ms")).unwrap_or_else(|| "?".into()),
             theme::TEXT_DIM),
        ],
        "solar" => vec![
            ("SSN".into(),
             data.get("ssn").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.0}")).unwrap_or_else(|| "?".into()),
             theme::AMBER),
            ("F10.7".into(),
             data.get("f10_7").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.1}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("Cycle".into(),
             data["cycle"].as_str().unwrap_or("SC25").into(),
             theme::TEXT_DIM),
            ("Latency".into(),
             data["latency_ms"].as_f64()
                 .map(|v| format!("{v:.1}ms")).unwrap_or_else(|| "?".into()),
             theme::TEXT_DIM),
        ],
        "lunar" => vec![
            ("Phase".into(),
             data.get("phase").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("Dist_km".into(),
             data.get("distance_km").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.0}")).unwrap_or_else(|| "?".into()),
             theme::AMBER),
            ("g_pert".into(),
             data.get("gravity_perturbation").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.2e}")).unwrap_or_else(|| "?".into()),
             theme::TEXT_MID),
            ("Latency".into(),
             data["latency_ms"].as_f64()
                 .map(|v| format!("{v:.1}ms")).unwrap_or_else(|| "?".into()),
             theme::TEXT_DIM),
        ],
        "radio" => vec![
            ("Deficit".into(),
             data.get("quadrupole_deficit").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("Asym".into(),
             data.get("hemispheric_asymmetry").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::AMBER),
            ("SDR_Floor".into(),
             data.get("noise_floor_db").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.1} dB")).unwrap_or_else(|| "?".into()),
             theme::TEXT_MID),
            ("Latency".into(),
             data["latency_ms"].as_f64()
                 .map(|v| format!("{v:.1}ms")).unwrap_or_else(|| "?".into()),
             theme::TEXT_DIM),
        ],
        "seismic" => vec![
            ("Events_24h".into(),
             data.get("events_24h").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.0}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("Max_Mag".into(),
             data.get("max_magnitude").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.1}")).unwrap_or_else(|| "?".into()),
             theme::AMBER),
            ("E_rel".into(),
             data.get("total_energy_relative").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::TEXT_MID),
            ("Latency".into(),
             data["latency_ms"].as_f64()
                 .map(|v| format!("{v:.1}ms")).unwrap_or_else(|| "?".into()),
             theme::TEXT_DIM),
        ],
        "magnon" => vec![
            ("T2_us".into(),
             data.get("t2_effective_us").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.2}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("Purity".into(),
             data.get("purity").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::GREEN_OK),
            ("Singlet".into(),
             data.get("singlet_yield").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::ACCENT),
            ("Entropy".into(),
             data.get("entropy_bits").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::AMBER),
        ],
        "quantum_lab" => vec![
            ("ln_Z".into(),
             data.get("log_Z").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::GREEN_OK),
            ("F/site".into(),
             data.get("free_energy_per_site").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("β".into(),
             data.get("beta").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.3}")).unwrap_or_else(|| "?".into()),
             theme::AMBER),
            ("L".into(),
             data.get("lattice_L").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.0}")).unwrap_or_else(|| "?".into()),
             theme::TEXT_MID),
        ],
        "cosmological" => vec![
            ("C₂".into(),
             data.get("quadrupole_C2").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.6}")).unwrap_or_else(|| "?".into()),
             theme::BLUE_INFO),
            ("Hemi_Δ".into(),
             data.get("hemi_asymmetry").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::AMBER),
            ("C₂/exp".into(),
             data.get("c2_ratio").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.4}")).unwrap_or_else(|| "?".into()),
             theme::GREEN_OK),
            ("lmax".into(),
             data.get("lmax").and_then(|v|v.as_f64())
                 .map(|v| format!("{v:.0}")).unwrap_or_else(|| "?".into()),
             theme::TEXT_MID),
        ],
        _ => {
            // Generic: extract first 4 numeric fields from data
            let mut out = Vec::new();
            if let Some(obj) = data.as_object() {
                let colors = [theme::BLUE_INFO, theme::AMBER, theme::TEXT_MID, theme::TEXT_DIM];
                for (i, (k, v)) in obj.iter()
                    .filter(|(k,v)| k.as_str() != "latency_ms" && v.is_number())
                    .take(3)
                    .enumerate()
                {
                    out.push((k.clone(), format!("{:.4}", v.as_f64().unwrap_or(0.0)), colors[i]));
                }
                if let Some(lat) = obj.get("latency_ms").and_then(|v|v.as_f64()) {
                    out.push(("Latency".into(), format!("{lat:.1}ms"), theme::TEXT_DIM));
                }
            }
            if out.is_empty() {
                make_fields(id, meta["score"].as_f64().unwrap_or(0.0), 0.0)
            } else { out }
        }
    }
}

// ══════════════════════ FIELD DATA ════════════════════════════════════════ //

fn make_fields(id: &str, s: f64, t: f64) -> Vec<(String, String, Color32)> {
    match id {
        "quantum" => vec![
            ("Φ_singlet".into(), format!("{:.4}", s*0.94),          theme::BLUE_INFO),
            ("Trace_f".into(),   format!("{:.4}", 0.98-s*0.1),      theme::TEXT_MID),
            ("n_sites".into(),   "14".into(),                        theme::TEXT_DIM),
            ("ΔB_rms".into(),    format!("{:.2e} T", 4.7e-9_f64),   theme::AMBER),
        ],
        "geomagnetic" => vec![
            ("B_total".into(),    format!("{:.2} nT", 42.0+2.0*(t*0.21).sin()), theme::BLUE_INFO),
            ("P_excursion".into(),format!("{:.4}", s*0.45),          theme::AMBER),
            ("LSTM_1kyr".into(),  format!("{:.4}", 0.50+s*0.30),    theme::TEXT_MID),
            ("Kp_index".into(),   format!("{:.1}", s*9.0),           theme::TEXT_DIM),
        ],
        "magnon" => vec![
            ("Fidelity".into(),  format!("{:.4}", s*0.98),           theme::BLUE_INFO),
            ("T2_eff".into(),    format!("{:.2} μs", 8.4+s*3.0),    theme::TEXT_MID),
            ("Purity".into(),    format!("{:.4}", 0.90+s*0.08),     theme::GREEN_OK),
            ("ω_magnon".into(),  format!("{:.3} GHz", 2.4+s*0.6),   theme::AMBER),
        ],
        "quantum_lab" => vec![
            ("ln Z".into(),   format!("{:.4}", -12.0+s*3.0),         theme::BLUE_INFO),
            ("F/site".into(), format!("{:.4}", -0.40-s*0.2),         theme::TEXT_MID),
            ("L".into(),      "16".into(),                            theme::TEXT_DIM),
            ("β".into(),      format!("{:.2}", 2.5-s*0.5),           theme::AMBER),
        ],
        "solar" => vec![
            ("SC_phase".into(), format!("{:.4}", (t*0.003)%1.0),    theme::AMBER),
            ("F10.7".into(),    format!("{:.1}", 120.0+s*80.0),      theme::BLUE_INFO),
            ("Cycle".into(),    "SC25".into(),                        theme::TEXT_DIM),
            ("SSN".into(),      format!("{:.0}", s*175.0),            theme::TEXT_MID),
        ],
        "cosmological" => vec![
            ("C_2".into(),    format!("{:.6}", 0.0146+s*0.002),      theme::BLUE_INFO),
            ("Hemi_Δ".into(), format!("{:.4}", 1.03+s*0.04),         theme::TEXT_MID),
            ("C2/exp".into(), format!("{:.4}", 0.95+s*0.06),         theme::AMBER),
            ("l_max".into(),  "191".into(),                           theme::TEXT_DIM),
        ],
        "eeg" => vec![
            ("α_rel".into(),    format!("{:.4}", s*0.82),             theme::BLUE_INFO),
            ("θ_rel".into(),    format!("{:.4}", (1.0-s)*0.28),      theme::TEXT_MID),
            ("Mode".into(),     "simulated".into(),                    theme::AMBER),
            ("Channels".into(), "4".into(),                            theme::TEXT_DIM),
        ],
        "lunar" => vec![
            ("Phase".into(),    format!("{:.4}", s),                  theme::BLUE_INFO),
            ("Dist_km".into(),  format!("{:.0}", 384400.0-s*20000.0), theme::AMBER),
            ("g_pert".into(),   format!("{:.2e}", 5.6e-8_f64),       theme::TEXT_MID),
            ("Status".into(),   "nominal".into(),                     theme::TEXT_DIM),
        ],
        "radio" => vec![
            ("T_cmb".into(),    "2.7255 K".into(),                    theme::BLUE_INFO),
            ("Deficit".into(),  format!("{:.4}", 0.85+s*0.05),        theme::TEXT_MID),
            ("Asym".into(),     format!("{:.4}", 0.07+s*0.01),        theme::AMBER),
            ("RTL-SDR".into(),  "active".into(),                      theme::GREEN_OK),
        ],
        "seismic" => vec![
            ("Events_24h".into(),"245".into(),                        theme::BLUE_INFO),
            ("Max_mag".into(),  format!("{:.1}", 5.2+s*0.8),          theme::AMBER),
            ("E_rel".into(),    format!("{:.4}", s*0.25),             theme::TEXT_MID),
            ("Alert".into(),    "low".into(),                         theme::GREEN_OK),
        ],
        _ => vec![],
    }
}

fn make_spec_col(id: &str, s: f64, t: f64) -> Vec<f32> {
    const N: usize = 40;
    let mut col = vec![0.0f32; N];
    for fi in 0..N {
        let freq  = fi as f64 * 1.25; // maps 0..N → 0..50 Hz
        let noise = 0.04 * ((fi as f64 * 1.3 + t * 7.1).sin().abs());
        let sig = match id {
            "eeg" => {
                let alpha = 0.90 * (-((freq-10.0).powi(2)/5.0)).exp();
                let theta = 0.55 * (-((freq- 6.0).powi(2)/3.0)).exp();
                let delta = 0.35 * (-((freq- 2.0).powi(2)/2.0)).exp();
                let one_f = 0.20 / (freq*0.5+1.0);
                alpha*s + theta*(0.8-s*0.4) + delta*0.3 + one_f
            },
            "geomagnetic" => {
                let s1 = 0.95*(-((freq- 7.83).powi(2)/2.0)).exp();
                let s2 = 0.55*(-((freq-14.30).powi(2)/2.0)).exp();
                let s3 = 0.35*(-((freq-20.80).powi(2)/2.5)).exp();
                let s4 = 0.20*(-((freq-27.30).powi(2)/2.5)).exp();
                (s1+s2+s3+s4)*(0.5+s*0.5)
            },
            "quantum" => {
                let q1 = 0.80*(-((freq- 5.0).powi(2)/3.0)).exp();
                let q2 = 0.60*(-((freq-13.0).powi(2)/4.0)).exp();
                let q3 = 0.40*(-((freq-28.0).powi(2)/5.0)).exp();
                (q1+q2+q3)*s
            },
            "magnon" => {
                let m1 = 0.85*(-((freq- 9.0).powi(2)/2.5)).exp();
                let m2 = 0.45*(-((freq-20.0).powi(2)/3.5)).exp();
                (m1+m2)*s
            },
            "solar" => {
                let sol = 0.70*(-(freq/6.0).powi(2)).exp();
                let h2  = 0.40*(-((freq-4.0).powi(2)/2.0)).exp();
                (sol+h2)*(0.4+s*0.6)
            },
            "cosmological" => {
                let c1 = 0.65*(-((freq- 1.5).powi(2)/1.5)).exp();
                let c2 = 0.35*(-((freq- 8.0).powi(2)/3.0)).exp();
                let c3 = 0.20*(-((freq-22.0).powi(2)/5.0)).exp();
                (c1+c2+c3)*s
            },
            "quantum_lab" => {
                let q1 = 0.75*(-((freq- 7.0).powi(2)/2.5)).exp();
                let q2 = 0.50*(-((freq-18.0).powi(2)/3.5)).exp();
                (q1+q2)*s
            },
            "lunar" => {
                let l1 = 0.80*(-((freq- 2.0).powi(2)/2.0)).exp();
                let l2 = 0.40*(-((freq-14.7).powi(2)/3.0)).exp();
                (l1+l2)*s
            },
            "radio" => {
                let r1 = 0.90*(-((freq- 2.7).powi(2)/1.5)).exp();
                let r2 = 0.60*(-((freq-21.0).powi(2)/4.0)).exp();
                let r3 = 0.45*(-((freq-40.0).powi(2)/5.0)).exp();
                (r1+r2+r3)*s
            },
            "seismic" => {
                let s1 = 0.85*(-((freq- 0.5).powi(2)/1.0)).exp();
                let s2 = 0.55*(-((freq- 5.0).powi(2)/2.0)).exp();
                let s3 = 0.35*(-((freq-15.0).powi(2)/3.0)).exp();
                (s1+s2+s3)*s
            },
            _ => 0.15/(freq*0.2+1.0),
        };
        let tm = 1.0 + 0.20*(t*0.4+fi as f64*0.08).sin();
        col[fi] = ((sig*tm+noise) as f32).clamp(0.0, 1.0);
    }
    col
}

// ══════════════════════ APP ═══════════════════════════════════════════════ //

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum CentralView { LayerInstrument, GlobalMesh }

pub struct SubstrateGui {
    bridge:       EngineBridge,
    selected:     usize,
    start_time:   Instant,
    central_view: CentralView,
}

impl SubstrateGui {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        // Override to white+red glassmorphism theme
        let mut vis = egui::Visuals::light();
        vis.override_text_color             = Some(theme::TEXT_DARK);
        vis.panel_fill                      = theme::BG_BASE;
        vis.window_fill                     = theme::BG_BASE;
        vis.widgets.noninteractive.bg_fill  = theme::BG_GLASS;
        vis.widgets.inactive.bg_fill        = theme::BG_GLASS;
        vis.widgets.hovered.bg_fill         = Color32::from_rgb(240, 240, 252);
        vis.widgets.active.bg_fill          = Color32::from_rgb(232, 232, 248);
        vis.selection.bg_fill               = Color32::from_rgba_unmultiplied(255,32,72,35);
        vis.selection.stroke                = Stroke::new(1.0, theme::ACCENT);
        cc.egui_ctx.set_visuals(vis);

        // Register wgpu pipeline into egui callback resources
        if let Some(wgpu_state) = cc.wgpu_render_state.as_ref() {
            let pipeline = FieldPipeline::new(
                &wgpu_state.device,
                wgpu_state.target_format,
            );
            wgpu_state.renderer
                .write()
                .callback_resources
                .insert(pipeline);
        }

        let bridge = EngineBridge::new();
        bridge.spawn_simulator();
        bridge.spawn_live_engine(cc.egui_ctx.clone());
        Self { bridge, selected: 0, start_time: Instant::now(), central_view: CentralView::LayerInstrument }
    }
    fn get_frame(&self) -> TelemetryFrame {
        self.bridge.frame.lock().unwrap().clone()
    }
}

impl eframe::App for SubstrateGui {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let tel   = self.get_frame();
        let t     = self.start_time.elapsed().as_secs_f64();
        if self.selected >= tel.layers.len() { self.selected = 0; }
        let layer = tel.layers[self.selected].clone();

        // ── Status bar ─────────────────────────────────────────
        TopBottomPanel::bottom("status")
            .exact_height(22.0)
            .frame(egui::Frame::none().fill(theme::BG_GLASS)
                .stroke(Stroke::new(1.0, theme::BORDER))
                .inner_margin(Margin::symmetric(14.0, 3.0)))
            .show(ctx, |ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    ui.colored_label(theme::TEXT_DIM,
                        RichText::new("SUBSTRATE SOBERANO · v0.1.0")
                            .font(FontId::new(10.5, FontFamily::Monospace)));
                    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                        let sc = if tel.status == "COHERENT" { theme::GREEN_OK }
                                 else { theme::BLUE_INFO };
                        ui.colored_label(sc,
                            RichText::new(format!("● {}  ·  {}",
                                tel.status,
                                chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC")))
                                .font(FontId::new(10.5, FontFamily::Monospace)));
                    });
                });
            });

        // ── Transport bar ──────────────────────────────────────
        TopBottomPanel::top("transport")
            .exact_height(44.0)
            .frame(egui::Frame::none().fill(theme::BG_CARD)
                .stroke(Stroke::new(1.0, theme::BORDER))
                .inner_margin(Margin::symmetric(16.0, 0.0)))
            .show(ctx, |ui| {
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    egui::Frame::none()
                        .fill(theme::ACCENT)
                        .rounding(Rounding::same(14.0))
                        .inner_margin(Margin::symmetric(9.0, 3.0))
                        .show(ui, |ui| {
                            ui.label(RichText::new("D")
                                .font(FontId::new(13.0, FontFamily::Proportional))
                                .color(Color32::WHITE));
                        });
                    ui.add_space(8.0);
                    let sc = if tel.status == "COHERENT" { theme::GREEN_OK }
                             else { theme::BLUE_INFO };
                    ui.colored_label(sc,
                        RichText::new(format!("● {}", tel.status))
                            .font(FontId::new(12.0, FontFamily::Monospace)));
                    ui.add_space(10.0);
                    ui.colored_label(theme::TEXT_DIM,
                        RichText::new(format!("{:.1}s", t))
                            .font(FontId::new(12.0, FontFamily::Monospace)));
                    ui.add_space(10.0);
                    for icon in &["⏹", "⏸", "▶"] {
                        if ui.add(egui::Button::new(
                            RichText::new(*icon)
                                .font(FontId::new(14.0, FontFamily::Proportional))
                                .color(theme::TEXT_MID)
                        ).frame(false)).clicked() {}
                        ui.add_space(2.0);
                    }
                    ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                        ui.label(RichText::new("■ SUBSTRATE")
                            .font(FontId::new(18.0, FontFamily::Proportional))
                            .color(theme::ACCENT));
                        ui.label(RichText::new("  QUANTUM WORKBENCH")
                            .font(FontId::new(12.0, FontFamily::Proportional))
                            .color(theme::TEXT_DIM));
                    });
                });
            });

        // ── Instrument Data Tape ───────────────────────────────
        TopBottomPanel::top("kpis")
            .exact_height(34.0)
            .frame(egui::Frame::none()
                .fill(theme::BG_CARD)
                .stroke(Stroke::new(1.0, theme::BORDER))
                .inner_margin(Margin::symmetric(14.0, 0.0)))
            .show(ctx, |ui| {
                ui.with_layout(Layout::left_to_right(Align::Center), |ui| {
                    let snr_sign = if tel.snr >= 0.0 { "+" } else { "" };
                    let snr_col  = if tel.snr >= 10.0 { theme::GREEN_OK }
                                   else if tel.snr >= 0.0 { theme::AMBER }
                                   else { theme::ACCENT };
                    let coh_col  = if tel.coherence >= 0.6 { theme::GREEN_OK }
                                   else if tel.coherence >= 0.35 { theme::AMBER }
                                   else { theme::TEXT_MID };
                    let lat_col  = if tel.latency_ms < 100.0 { theme::GREEN_OK }
                                   else if tel.latency_ms < 500.0 { theme::AMBER }
                                   else { theme::ACCENT };

                    // Max off-diagonal Pearson r across the correlation matrix
                    let max_r: f64 = {
                        let mut best = 0.0f64;
                        let n = tel.corr_matrix.len();
                        for i in 0..n {
                            for j in (i+1)..n {
                                if let Some(r) = tel.corr_matrix.get(i).and_then(|row| row.get(j)) {
                                    if r.abs() > best.abs() { best = *r; }
                                }
                            }
                        }
                        best
                    };
                    let psi_col = if max_r.abs() >= 0.7 { theme::GREEN_OK }
                                  else if max_r.abs() >= 0.4 { theme::AMBER }
                                  else { theme::ACCENT };
                    let psi_str  = format!("r={:+.3}", max_r);

                    // Segment-colored render: COHR value red, rest dimmer
                    let segments: &[(&str, Color32)] = &[
                        ("COHR ",          theme::TEXT_DIM),
                        (&format!("{:.4}", tel.coherence), coh_col),
                        ("   FIELD ",      theme::TEXT_DIM),
                        (&format!("{:.2} nT", tel.field_nt), theme::BLUE_INFO),
                        ("   Q_SNR ",      theme::TEXT_DIM),
                        (&format!("{}{:.1} dB", snr_sign, tel.snr), snr_col),
                        ("   LAT ",        theme::TEXT_DIM),
                        (&format!("{:.1} ms", tel.latency_ms), lat_col),
                        ("   ENT ",        theme::TEXT_DIM),
                        (&format!("{:.4}", tel.entropy), theme::AMBER),
                        ("   ψ ",          theme::TEXT_DIM),
                        (&psi_str,         psi_col),
                    ];
                    for (text, col) in segments {
                        ui.colored_label(*col, RichText::new(*text)
                            .font(FontId::new(11.5, FontFamily::Monospace)));
                    }
                });
            });

        // ── Left Sidebar — flat instrument panel ───────────────
        SidePanel::left("navigator")
            .exact_width(210.0)
            .frame(egui::Frame::none().fill(theme::BG_GLASS)
                .stroke(Stroke::new(1.0, theme::BORDER))
                .inner_margin(Margin::symmetric(0.0, 8.0)))
            .show(ctx, |ui| {
                // Section header
                ui.add_space(2.0);
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 16.0),
                    Layout::left_to_right(Align::Center),
                    |ui| {
                        ui.add_space(10.0);
                        ui.colored_label(theme::TEXT_DIM,
                            RichText::new("LAYERS")
                                .font(FontId::new(9.0, FontFamily::Monospace)));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.add_space(8.0);
                            let done_n = tel.layers.iter().filter(|l| l.state == LayerState::Done).count();
                            ui.colored_label(theme::TEXT_DIM,
                                RichText::new(format!("{}/{}", done_n, tel.layers.len()))
                                    .font(FontId::new(9.0, FontFamily::Monospace)));
                        });
                    }
                );
                ui.add_space(4.0);

                // Flat row list — no card boxes
                for (idx, lyr) in tel.layers.iter().enumerate() {
                    let sel = self.selected == idx;
                    let row_h = 26.0f32;
                    let (row_rect, row_resp) = ui.allocate_exact_size(
                        Vec2::new(ui.available_width(), row_h), Sense::click());

                    if ui.is_rect_visible(row_rect) {
                        let painter = ui.painter_at(row_rect);
                        // Selection background
                        if sel {
                            painter.rect_filled(row_rect,
                                Rounding::same(0.0),
                                Color32::from_rgba_unmultiplied(255, 32, 72, 10));
                        }
                        // Left accent bar
                        if sel {
                            painter.rect_filled(
                                Rect::from_min_size(row_rect.min, Vec2::new(3.0, row_h)),
                                Rounding::same(0.0), theme::ACCENT);
                        }
                        // State LED
                        let led_x = row_rect.min.x + 12.0;
                        let led_y = row_rect.center().y;
                        painter.circle_filled(Pos2::new(led_x, led_y), 3.5, lyr.state_color());

                        // Layer name
                        let name_col = if sel { theme::ACCENT } else { theme::TEXT_DARK };
                        painter.text(
                            Pos2::new(led_x + 10.0, led_y),
                            egui::Align2::LEFT_CENTER,
                            &lyr.name,
                            FontId::new(12.0, FontFamily::Proportional),
                            name_col,
                        );

                        // Score (right-aligned, monospace)
                        let score_col = if lyr.score >= 0.7 { theme::GREEN_OK }
                                        else if lyr.score >= 0.4 { theme::AMBER }
                                        else { theme::TEXT_DIM };
                        painter.text(
                            Pos2::new(row_rect.max.x - 8.0, led_y),
                            egui::Align2::RIGHT_CENTER,
                            format!("{:.3}", lyr.score),
                            FontId::new(10.5, FontFamily::Monospace),
                            score_col,
                        );

                        // Score bar (thin, bottom of row)
                        let bar_y  = row_rect.max.y - 2.0;
                        let bar_w  = (row_rect.width() - 20.0) * lyr.score as f32;
                        painter.line_segment(
                            [Pos2::new(row_rect.min.x + 20.0, bar_y),
                             Pos2::new(row_rect.min.x + 20.0 + bar_w, bar_y)],
                            Stroke::new(1.5, Color32::from_rgba_unmultiplied(score_col.r(), score_col.g(), score_col.b(), 120)));

                        // Bottom separator (dimmer)
                        painter.line_segment(
                            [Pos2::new(row_rect.min.x + 10.0, row_rect.max.y - 0.5),
                             Pos2::new(row_rect.max.x - 10.0, row_rect.max.y - 0.5)],
                            Stroke::new(0.5, Color32::from_rgba_unmultiplied(200, 200, 220, 80)));
                    }

                    if row_resp.clicked() { self.selected = idx; }
                }

                ui.add_space(8.0);

                // Control buttons — compact, no rounding
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 28.0),
                    Layout::left_to_right(Align::Center),
                    |ui| {
                        ui.add_space(8.0);
                        if ui.add_sized([ui.available_width() * 0.55, 24.0],
                            egui::Button::new(
                                RichText::new("▶ RUN")
                                    .font(FontId::new(10.5, FontFamily::Monospace))
                                    .color(Color32::WHITE))
                            .fill(theme::ACCENT)
                            .rounding(Rounding::same(2.0)),
                        ).clicked() {
                            let mut running = self.bridge.running.lock().unwrap();
                            if !*running {
                                *running = true;
                                drop(running);
                                self.bridge.spawn_live_engine(ctx.clone());
                            }
                        }
                        ui.add_space(4.0);
                        if ui.add_sized([ui.available_width() - 8.0, 24.0],
                            egui::Button::new(
                                RichText::new("■ STOP")
                                    .font(FontId::new(10.5, FontFamily::Monospace))
                                    .color(theme::ACCENT))
                            .fill(theme::BG_CARD)
                            .stroke(Stroke::new(1.0, theme::BORDER_RED))
                            .rounding(Rounding::same(2.0)),
                        ).clicked() { self.bridge.stop(); }
                    }
                );

                ui.add_space(6.0);
                // Divider
                ui.allocate_ui_with_layout(Vec2::new(ui.available_width(), 1.0),
                    Layout::left_to_right(Align::Center), |ui| {
                        let (r, p) = ui.allocate_painter(Vec2::new(ui.available_width(), 1.0), Sense::hover());
                        p.line_segment([r.rect.left_center(), r.rect.right_center()],
                            Stroke::new(1.0, theme::BORDER));
                    });
                ui.add_space(6.0);

                // Field readout — monospace kv pairs
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 14.0),
                    Layout::left_to_right(Align::Center),
                    |ui| {
                        ui.add_space(10.0);
                        ui.colored_label(theme::TEXT_DIM,
                            RichText::new(format!("FIELDS · {}", layer.id.to_uppercase()))
                                .font(FontId::new(8.5, FontFamily::Monospace)));
                    }
                );
                ui.add_space(3.0);
                for (key, val, color) in &layer.fields {
                    ui.allocate_ui_with_layout(
                        Vec2::new(ui.available_width(), 16.0),
                        Layout::left_to_right(Align::Center),
                        |ui| {
                            ui.add_space(10.0);
                            ui.colored_label(theme::TEXT_DIM,
                                RichText::new(key)
                                    .font(FontId::new(10.5, FontFamily::Monospace)));
                            ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                ui.add_space(8.0);
                                ui.colored_label(*color,
                                    RichText::new(val)
                                        .font(FontId::new(10.5, FontFamily::Monospace)));
                            });
                        }
                    );
                }

                // Entropy + cycle — pinned bottom
                let remaining = ui.available_height();
                if remaining > 44.0 { ui.add_space(remaining - 42.0); }
                ui.allocate_ui_with_layout(Vec2::new(ui.available_width(), 1.0),
                    Layout::left_to_right(Align::Center), |ui| {
                        let (r, p) = ui.allocate_painter(Vec2::new(ui.available_width(), 1.0), Sense::hover());
                        p.line_segment([r.rect.left_center(), r.rect.right_center()],
                            Stroke::new(1.0, theme::BORDER));
                    });
                ui.add_space(3.0);
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 32.0),
                    Layout::left_to_right(Align::Center),
                    |ui| {
                        ui.add_space(10.0);
                        ui.colored_label(theme::TEXT_DIM,
                            RichText::new("ENT ")
                                .font(FontId::new(9.5, FontFamily::Monospace)));
                        ui.colored_label(theme::AMBER,
                            RichText::new(format!("{:.4}", tel.entropy))
                                .font(FontId::new(15.0, FontFamily::Monospace)));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            ui.add_space(8.0);
                            ui.colored_label(theme::TEXT_DIM,
                                RichText::new(format!("cyc {:04}", tel.cycle))
                                    .font(FontId::new(9.5, FontFamily::Monospace)));
                        });
                    }
                );
            });

        // ── Key G: toggle central view ─────────────────────────
        ctx.input(|i| {
            if i.key_pressed(egui::Key::G) {
                self.central_view = match self.central_view {
                    CentralView::LayerInstrument => CentralView::GlobalMesh,
                    CentralView::GlobalMesh      => CentralView::LayerInstrument,
                };
            }
        });

        // ── Central Panel ──────────────────────────────────────
        CentralPanel::default()
            .frame(egui::Frame::none().fill(theme::BG_BASE)
                .inner_margin(Margin::same(10.0)))
            .show(ctx, |ui| {
                // Sub-header
                ui.allocate_ui_with_layout(
                    Vec2::new(ui.available_width(), 20.0),
                    Layout::left_to_right(Align::Center),
                    |ui| {
                        let hdr = match self.central_view {
                            CentralView::LayerInstrument =>
                                format!("■ {}  —  SCIENTIFIC INSTRUMENT", layer.name.to_uppercase()),
                            CentralView::GlobalMesh =>
                                "■ K-FIELD  TOPOLOGÍA 3D Q-ESPACIO  ·  PEARSON CORRELATIONS".into(),
                        };
                        ui.colored_label(theme::TEXT_MID,
                            RichText::new(hdr).font(FontId::new(10.5, FontFamily::Proportional)));
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            let (bg, fg, lbl) = match self.central_view {
                                CentralView::GlobalMesh      => (theme::ACCENT, Color32::WHITE,       "[G] INSTRUMENT"),
                                CentralView::LayerInstrument => (theme::BG_CARD, theme::ACCENT,       "[G] GLOBAL MESH"),
                            };
                            egui::Frame::none()
                                .fill(bg)
                                .rounding(Rounding::same(10.0))
                                .stroke(Stroke::new(1.0, theme::BORDER_RED))
                                .inner_margin(Margin::symmetric(8.0, 2.0))
                                .show(ui, |ui| {
                                    ui.colored_label(fg, RichText::new(lbl)
                                        .font(FontId::new(9.5, FontFamily::Monospace)));
                                });
                        });
                    }
                );
                ui.add_space(6.0);

                let avail   = ui.available_size();
                let events_h = if tel.events.is_empty() { 0.0f32 } else { 68.0 };
                let tl_h    = 90.0f32;
                let main_h  = (avail.y - tl_h - events_h - 20.0).max(80.0);

                // ── Main instrument area ──────────────────────
                match self.central_view {
                    CentralView::GlobalMesh => {
                        let mesh_w = avail.x * 0.60 - 3.0;
                        let spec_w = avail.x * 0.40;
                        ui.allocate_ui_with_layout(
                            Vec2::new(avail.x, main_h),
                            Layout::left_to_right(Align::Min),
                            |ui| {
                                let (rm, pm) = ui.allocate_painter(
                                    Vec2::new(mesh_w, main_h), Sense::hover());
                                pm.rect_filled(rm.rect, Rounding::same(8.0), theme::BG_DARK);
                                pm.rect_stroke(rm.rect, Rounding::same(8.0),
                                    Stroke::new(1.0, theme::BORDER));
                                render_mesh_overlay(&pm, rm.rect, &tel.layers, t,
                                    "global", &tel.corr_matrix);
                                let scores: Vec<f64> = tel.layers.iter().map(|l| l.score).collect();
                                pm.add(egui_wgpu::Callback::new_paint_callback(
                                    rm.rect,
                                    FieldCallback {
                                        uniforms: FieldUniforms::from_telemetry(
                                            t as f32, tel.coherence, &scores, &tel.corr_matrix),
                                    },
                                ));
                                // Coherence badge
                                let badge = Rect::from_min_size(
                                    rm.rect.min + Vec2::new(12.0, 10.0), Vec2::new(160.0, 44.0));
                                pm.rect_filled(badge, Rounding::same(8.0),
                                    Color32::from_rgba_unmultiplied(255,255,255,220));
                                pm.rect_stroke(badge, Rounding::same(8.0),
                                    Stroke::new(1.0, theme::BORDER));
                                pm.text(badge.min + Vec2::new(10.0, 6.0), egui::Align2::LEFT_TOP,
                                    "Global Coherence Score",
                                    FontId::new(9.5, FontFamily::Proportional), theme::TEXT_DIM);
                                pm.text(badge.min + Vec2::new(10.0, 20.0), egui::Align2::LEFT_TOP,
                                    format!("{:.4}  [{} layers]", tel.coherence, tel.layers.len()),
                                    FontId::new(14.0, FontFamily::Monospace), theme::TEXT_DARK);

                                ui.add_space(6.0);
                                let (rs, ps) = ui.allocate_painter(
                                    Vec2::new(spec_w, main_h), Sense::hover());
                                render_corr_matrix(&ps, rs.rect, &tel.layers, &tel.corr_matrix);
                            }
                        );
                    }
                    CentralView::LayerInstrument => {
                        let (r_inst, p_inst) = ui.allocate_painter(
                            Vec2::new(avail.x, main_h), Sense::hover());
                        match layer.id.as_str() {
                            "eeg"          => render_eeg_instrument(&p_inst, r_inst.rect, &layer, t),
                            "quantum"      => render_quantum_instrument(&p_inst, r_inst.rect, &layer, t),
                            "solar"        => render_solar_instrument(&p_inst, r_inst.rect, &layer, t),
                            "geomagnetic"  => render_geomagnetic_instrument(&p_inst, r_inst.rect, &layer, t),
                            "lunar"        => render_lunar_instrument(&p_inst, r_inst.rect, &layer, t),
                            "seismic"      => render_seismic_instrument(&p_inst, r_inst.rect, &layer, t),
                            "radio"        => render_radio_instrument(&p_inst, r_inst.rect, &layer, t),
                            "magnon"       => render_magnon_instrument(&p_inst, r_inst.rect, &layer, t),
                            "quantum_lab"  => render_quantum_lab_instrument(&p_inst, r_inst.rect, &layer, t),
                            "cosmological" => render_cosmological_instrument(&p_inst, r_inst.rect, &layer, t),
                            _              => render_spectrogram(&p_inst, r_inst.rect, &layer.spec_data, &layer.id),
                        }
                    }
                }

                ui.add_space(6.0);

                // ── Timeline — phosphor oscilloscope ──────────
                {
                    let last  = layer.history.back().copied().unwrap_or(layer.score);
                    let prev  = layer.history.iter().nth_back(1).copied().unwrap_or(last);
                    let delta = last - prev;
                    let ds_sign = if delta >= 0.0 { "↑" } else { "↓" };
                    let hdr = format!(
                        "OSC  {}   {:.4} score  ·  {} Δ{:.4}",
                        layer.id.to_uppercase(), layer.score, ds_sign, delta.abs());
                    ui.colored_label(theme::TEXT_DIM,
                        RichText::new(hdr).font(FontId::new(9.5, FontFamily::Monospace)));
                }
                ui.add_space(2.0);

                let plot_h = (tl_h - 16.0).max(40.0);
                let (osc_rect, _) = ui.allocate_exact_size(
                    Vec2::new(ui.available_width(), plot_h), Sense::hover());

                let osc_painter = ui.painter_at(osc_rect);
                // Dark background
                osc_painter.rect_filled(osc_rect, Rounding::same(2.0), theme::BG_DARK);
                // Grid lines
                for frac in [0.25f32, 0.50, 0.75] {
                    let y = osc_rect.min.y + frac * osc_rect.height();
                    osc_painter.line_segment(
                        [Pos2::new(osc_rect.min.x, y), Pos2::new(osc_rect.max.x, y)],
                        Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,18)));
                }

                let n = layer.history.len();
                if n >= 2 {
                    // Auto-scale Y to actual data range so small variations fill the display
                    let h_min = layer.history.iter().cloned().fold(f64::MAX, f64::min);
                    let h_max = layer.history.iter().cloned().fold(f64::MIN, f64::max);
                    let h_range = (h_max - h_min).max(0.04); // min 4% window
                    let h_bot = h_min - h_range * 0.15;
                    let h_top = h_max + h_range * 0.15;
                    let y_of = |v: f64| -> f32 {
                        let t = ((v - h_bot) / (h_top - h_bot)) as f32;
                        osc_rect.max.y - t * osc_rect.height()
                    };

                    // Draw score line — phosphor: glow pass (thick, dim) + sharp pass (thin, bright)
                    for pass in 0..2usize {
                        let mut prev: Option<Pos2> = None;
                        for (i, &v) in layer.history.iter().enumerate() {
                            let x = osc_rect.min.x + (i as f32 / (n-1).max(1) as f32) * osc_rect.width();
                            let y = y_of(v);
                            let pt = Pos2::new(x, y.clamp(osc_rect.min.y, osc_rect.max.y));
                            if let Some(pr) = prev {
                                let (w, col) = if pass == 0 {
                                    (4.0, Color32::from_rgba_unmultiplied(30, 120, 255, 35))
                                } else {
                                    (1.2, Color32::from_rgb(80, 160, 255))
                                };
                                osc_painter.line_segment([pr, pt], Stroke::new(w, col));
                            }
                            prev = Some(pt);
                        }
                    }
                    // Secondary line (red phosphor)
                    let n2 = layer.history2.len();
                    if n2 >= 2 {
                        for pass in 0..2usize {
                            let mut prev: Option<Pos2> = None;
                            for (i, &v) in layer.history2.iter().enumerate() {
                                let vn = (v * 0.5 + 0.5) as f32;
                                let x = osc_rect.min.x + (i as f32 / (n2-1).max(1) as f32) * osc_rect.width();
                                let y = osc_rect.max.y - vn * osc_rect.height();
                                let pt = Pos2::new(x, y.clamp(osc_rect.min.y, osc_rect.max.y));
                                if let Some(pr) = prev {
                                    let (w, col) = if pass == 0 {
                                        (3.0, Color32::from_rgba_unmultiplied(255, 32, 72, 25))
                                    } else {
                                        (1.0, Color32::from_rgba_unmultiplied(255, 80, 100, 200))
                                    };
                                    osc_painter.line_segment([pr, pt], Stroke::new(w, col));
                                }
                                prev = Some(pt);
                            }
                        }
                    }
                    // Current value cursor (rightmost point, bright dot)
                    let last_v = *layer.history.back().unwrap();
                    let dot_x  = osc_rect.max.x - 1.0;
                    let dot_y  = y_of(last_v).clamp(osc_rect.min.y, osc_rect.max.y);
                    osc_painter.circle_filled(Pos2::new(dot_x, dot_y), 3.5,
                        Color32::from_rgb(80, 200, 255));
                    osc_painter.circle_filled(Pos2::new(dot_x, dot_y), 1.5,
                        Color32::WHITE);
                    // Y-axis scale labels (top = h_top, bottom = h_bot)
                    let dim = Color32::from_rgba_unmultiplied(180, 180, 200, 120);
                    osc_painter.text(Pos2::new(osc_rect.min.x + 3.0, osc_rect.min.y + 2.0),
                        egui::Align2::LEFT_TOP, format!("{h_top:.3}"),
                        FontId::new(8.0, FontFamily::Monospace), dim);
                    osc_painter.text(Pos2::new(osc_rect.min.x + 3.0, osc_rect.max.y - 2.0),
                        egui::Align2::LEFT_BOTTOM, format!("{h_bot:.3}"),
                        FontId::new(8.0, FontFamily::Monospace), dim);
                    // Scanlines overlay
                    let mut sy = osc_rect.min.y;
                    while sy < osc_rect.max.y {
                        osc_painter.line_segment(
                            [Pos2::new(osc_rect.min.x, sy), Pos2::new(osc_rect.max.x, sy)],
                            Stroke::new(0.5, Color32::from_rgba_unmultiplied(0,0,0,18)));
                        sy += 3.0;
                    }
                }
                osc_painter.rect_stroke(osc_rect, Rounding::same(2.0),
                    Stroke::new(1.0, theme::BORDER));

                // ── Coherence Events Log ───────────────────────
                if !tel.events.is_empty() {
                    ui.add_space(3.0);
                    egui::Frame::none()
                        .fill(Color32::from_rgba_unmultiplied(255, 32, 72, 8))
                        .rounding(Rounding::same(6.0))
                        .stroke(Stroke::new(1.0, theme::BORDER_RED))
                        .inner_margin(Margin::symmetric(10.0, 4.0))
                        .show(ui, |ui| {
                            ui.allocate_ui_with_layout(
                                Vec2::new(ui.available_width(), 14.0),
                                Layout::left_to_right(Align::Center),
                                |ui| {
                                    ui.colored_label(theme::ACCENT,
                                        RichText::new("■ COHERENCE EVENTS")
                                            .font(FontId::new(9.0, FontFamily::Proportional)));
                                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                        ui.colored_label(theme::TEXT_DIM,
                                            RichText::new(format!("{} detected", tel.events.len()))
                                                .font(FontId::new(9.0, FontFamily::Monospace)));
                                    });
                                }
                            );
                            ui.add_space(1.0);
                            for ev in tel.events.iter().rev().take(2) {
                                let pair_str = ev.pairs.iter().take(3)
                                    .map(|(a, b, r)| format!("{a}↔{b}({r:+.2})"))
                                    .collect::<Vec<_>>().join("  ");
                                ui.allocate_ui_with_layout(
                                    Vec2::new(ui.available_width(), 14.0),
                                    Layout::left_to_right(Align::Center),
                                    |ui| {
                                        ui.colored_label(theme::AMBER,
                                            RichText::new(format!("cycle {:04}", ev.cycle))
                                                .font(FontId::new(9.0, FontFamily::Monospace)));
                                        ui.add_space(8.0);
                                        ui.colored_label(theme::TEXT_MID,
                                            RichText::new(&pair_str)
                                                .font(FontId::new(9.0, FontFamily::Monospace)));
                                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                                            let sc = if ev.severity > 0.3 { theme::ACCENT } else { theme::AMBER };
                                            ui.colored_label(sc,
                                                RichText::new(format!("sev {:.2}", ev.severity))
                                                    .font(FontId::new(9.0, FontFamily::Monospace)));
                                        });
                                    }
                                );
                            }
                        });
                }
            });

        ctx.request_repaint_after(Duration::from_millis(60));
    }
}

// ══════════════════════ MESH RENDERER ═════════════════════════════════════ //

/// Color for correlation value: blue (−1) → white (0) → red (+1)
fn corr_color(r: f64) -> Color32 {
    let t = r.clamp(-1.0, 1.0) as f32;
    if t < 0.0 {
        let s = (-t).clamp(0.0, 1.0);
        Color32::from_rgb(
            theme::lerp(255, 30,  s),
            theme::lerp(255, 80,  s),
            theme::lerp(255, 220, s),
        )
    } else {
        let s = t.clamp(0.0, 1.0);
        Color32::from_rgb(
            255,
            theme::lerp(255, 32,  s),
            theme::lerp(255, 72,  s),
        )
    }
}

/// Renders layer-specific symbolic overlays or the correlation matrix
/// on top of the wgpu dark background. No background/border drawn here.
fn render_mesh_overlay(
    painter:  &egui::Painter,
    rect:     Rect,
    layers:   &[LayerMetrics],
    t:        f64,
    layer_id: &str,
    corr:     &[Vec<f64>],
) {
    let cx = rect.center().x;
    let cy = rect.center().y;

    // ── Layer-specific symbolic views (only in instrument mode, not global) ─
    match layer_id {
        "global" => {} // fall through to corr matrix / wave mesh
        "lunar" => {
            let radius = rect.height() * 0.35;
            painter.circle_filled(Pos2::new(cx, cy), radius, Color32::from_rgb(30, 30, 45));
            painter.circle_stroke(Pos2::new(cx, cy), radius, Stroke::new(1.5, theme::BORDER));
            let phase = layers.iter().find(|l| l.id == "lunar").map(|l| l.score).unwrap_or(0.5) as f32;
            for i in 0..=20usize {
                let f  = i as f32 / 20.0;
                let y  = cy - radius + f * radius * 2.0;
                let w  = (radius * radius - (y - cy) * (y - cy)).sqrt();
                let xt = cx + w * (phase * 2.0 - 1.0);
                painter.line_segment([Pos2::new(cx - w, y), Pos2::new(xt, y)],
                    Stroke::new(1.0, theme::BLUE_INFO));
            }
            return;
        }
        "solar" => {
            let radius = rect.height() * 0.38;
            painter.circle_filled(Pos2::new(cx, cy), radius, Color32::from_rgb(255, 245, 230));
            painter.circle_stroke(Pos2::new(cx, cy), radius, Stroke::new(2.0, theme::AMBER));
            let score  = layers.iter().find(|l| l.id == "solar").map(|l| l.score).unwrap_or(0.5);
            let n_spots = 3 + (score * 12.0) as usize;
            for i in 0..n_spots {
                let angle = i as f64 * 2.3 + t * 0.4;
                let dist  = radius as f64 * 0.6 * (i as f64 * 1.7).sin().abs();
                painter.circle_filled(
                    Pos2::new(cx + (dist * angle.cos()) as f32,
                              cy + (dist * angle.sin() * 0.5) as f32),
                    3.0 + (i % 3) as f32 * 1.5, theme::TEXT_DARK);
            }
            return;
        }
        "geomagnetic" => {
            let radius = rect.height() * 0.18;
            painter.circle_filled(Pos2::new(cx, cy), radius, theme::BLUE_INFO);
            for i in 1..=6usize {
                let r_max = radius + i as f32 * radius * 0.6;
                painter.rect_stroke(
                    Rect::from_center_size(Pos2::new(cx, cy), Vec2::new(r_max * 1.8, r_max)),
                    Rounding::same(r_max),
                    Stroke::new(1.0, Color32::from_rgba_unmultiplied(30, 120, 255, 60)));
            }
            return;
        }
        _ => {}
    }

    // ── If we have real correlation data → render correlation matrix ──────
    let has_corr = !corr.is_empty()
        && corr.len() == layers.len()
        && layers.iter().any(|l| l.has_real_data && l.history.len() >= 8);

    if has_corr {
        render_corr_matrix(painter, rect, layers, corr);
        return;
    }

    // ── Generic wave mesh (sim fallback) ─────────────────────────────────
    let n      = 24usize;
    let tile_w = rect.width() / (n as f32 * 1.08);
    let tile_h = tile_w * 0.48;
    let z_sc   = tile_w * 1.15;
    let coherence = if layers.is_empty() { 0.5 } else {
        layers.iter().map(|l| l.score).sum::<f64>() / layers.len() as f64
    };
    let mut h = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in 0..n {
            let fi = i as f64 / (n-1) as f64;
            let fj = j as f64 / (n-1) as f64;
            let v = 0.30*(fi*3.8*std::f64::consts::PI + t*0.27).sin()
                  + 0.27*(fj*4.2*std::f64::consts::PI - t*0.21).cos()
                  + 0.17*((fi+fj)*5.6*std::f64::consts::PI + t*0.44).sin()
                  + 0.09*(fi*fj*7.8*std::f64::consts::PI - t*0.34).cos()
                  + coherence * 0.12;
            h[i][j] = (v + 0.50).clamp(0.0, 1.0) as f32;
        }
    }
    let project = |i: usize, j: usize| -> Pos2 {
        let z = h[i][j];
        Pos2::new(cx + (i as f32 - j as f32) * tile_w * 0.5,
                  cy + (i as f32 + j as f32) * tile_h * 0.5 - z * z_sc)
    };
    for i in 0..n {
        for j in 0..n-1 {
            let avg = (h[i][j] + h[i][j+1]) * 0.5;
            painter.line_segment([project(i,j), project(i,j+1)],
                Stroke::new(if avg > 0.70 { 1.5 } else { 0.7 }, theme::mesh_color(avg as f64)));
        }
    }
    for j in 0..n {
        for i in 0..n-1 {
            let avg = (h[i][j] + h[i+1][j]) * 0.5;
            painter.line_segment([project(i,j), project(i+1,j)],
                Stroke::new(if avg > 0.70 { 1.5 } else { 0.7 }, theme::mesh_color(avg as f64)));
        }
    }
}

/// Render live N×N Pearson correlation matrix as colored isometric tiles.
fn render_corr_matrix(
    painter: &egui::Painter,
    rect:    Rect,
    layers:  &[LayerMetrics],
    corr:    &[Vec<f64>],
) {
    let n = layers.len();
    if n == 0 { return; }

    let pad   = 40.0f32; // left/bottom margin for labels
    let plot  = Rect::from_min_max(
        Pos2::new(rect.min.x + pad, rect.min.y + 18.0),
        Pos2::new(rect.max.x - 8.0, rect.max.y - pad),
    );
    let cell_w = plot.width()  / n as f32;
    let cell_h = plot.height() / n as f32;

    // Title
    painter.text(
        Pos2::new(rect.min.x + pad, rect.min.y + 3.0),
        egui::Align2::LEFT_TOP,
        "CROSS-LAYER PEARSON CORRELATION MATRIX",
        FontId::new(9.5, FontFamily::Proportional),
        theme::TEXT_MID,
    );

    for i in 0..n {
        for j in 0..n {
            let r = if i < corr.len() && j < corr[i].len() { corr[i][j] } else { 0.0 };
            let x = plot.min.x + j as f32 * cell_w;
            let y = plot.min.y + i as f32 * cell_h;
            let cell = Rect::from_min_size(Pos2::new(x, y), Vec2::new(cell_w - 1.0, cell_h - 1.0));

            // Fill with diverging color
            painter.rect_filled(cell, Rounding::same(2.0), corr_color(r));

            // Diagonal marker
            if i == j {
                painter.rect_stroke(cell, Rounding::same(2.0),
                    Stroke::new(1.5, Color32::from_rgba_unmultiplied(0, 0, 0, 80)));
            }

            // Value text in cell if big enough
            if cell_w > 28.0 {
                let txt_col = if r.abs() > 0.5 { Color32::WHITE } else { theme::TEXT_DARK };
                painter.text(
                    cell.center(),
                    egui::Align2::CENTER_CENTER,
                    if i == j { "1.00".into() } else { format!("{r:+.2}") },
                    FontId::new((cell_w * 0.28).clamp(7.0, 11.0), FontFamily::Monospace),
                    txt_col,
                );
            }
        }

        // Row label (layer name, abbreviated)
        let label = layers[i].id.chars().take(4).collect::<String>();
        painter.text(
            Pos2::new(rect.min.x + 2.0, plot.min.y + i as f32 * cell_h + cell_h * 0.5),
            egui::Align2::LEFT_CENTER,
            &label,
            FontId::new(8.5, FontFamily::Monospace),
            theme::TEXT_DIM,
        );

        // Col label (bottom)
        painter.text(
            Pos2::new(plot.min.x + i as f32 * cell_w + cell_w * 0.5, plot.max.y + 2.0),
            egui::Align2::CENTER_TOP,
            &label,
            FontId::new(8.5, FontFamily::Monospace),
            theme::TEXT_DIM,
        );
    }

    // Color scale bar (right side, blue→white→red)
    let bar = Rect::from_min_size(
        Pos2::new(plot.max.x + 4.0, plot.min.y),
        Vec2::new(6.0, plot.height()),
    );
    let steps = 20usize;
    for k in 0..steps {
        let frac = k as f32 / steps as f32;
        let r    = frac as f64 * 2.0 - 1.0;
        let y0   = bar.min.y + (1.0 - frac) * bar.height();
        let y1   = bar.min.y + (1.0 - (k + 1) as f32 / steps as f32) * bar.height();
        painter.rect_filled(
            Rect::from_min_max(Pos2::new(bar.min.x, y0.min(y1)), Pos2::new(bar.max.x, y0.max(y1))),
            Rounding::same(0.0),
            corr_color(r),
        );
    }
}

// ══════════════════════ SPECTROGRAM RENDERER ══════════════════════════════ //

fn render_spectrogram(
    painter: &egui::Painter,
    rect: Rect,
    spec_data: &VecDeque<Vec<f32>>,
    layer_id: &str,
) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    if spec_data.is_empty() { return; }

    let n_cols  = spec_data.len();
    let n_freqs = spec_data[0].len();
    let lbl_w   = 28.0f32;
    let lbl_h   = 14.0f32;
    let pad     = 6.0f32;

    let plot = Rect::from_min_max(
        Pos2::new(rect.min.x + lbl_w, rect.min.y + pad + 14.0),
        Pos2::new(rect.max.x - pad,   rect.max.y - lbl_h),
    );

    let col_w = plot.width()  / n_cols  as f32;
    let row_h = plot.height() / n_freqs as f32;

    // Draw spectrogram cells
    for (ci, col) in spec_data.iter().enumerate() {
        for (fi, &power) in col.iter().enumerate() {
            let x = plot.min.x + ci as f32 * col_w;
            let y = plot.max.y - (fi as f32 + 1.0) * row_h;
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(x, y), Vec2::new(col_w + 0.6, row_h + 0.6)),
                Rounding::same(0.0),
                theme::spec_color(power),
            );
        }
    }

    // Frequency band markers + labels
    let bands: &[(f32, &str)] = match layer_id {
        "eeg" => &[(0.0,"δ"),(4.0,"θ"),(8.0,"α"),(13.0,"β"),(30.0,"γ")],
        "geomagnetic" => &[(7.83,"S1"),(14.3,"S2"),(20.8,"S3"),(27.3,"S4")],
        "solar" => &[(0.0,"SC"),(11.0,"SSN"),(28.0,"F10")],
        "lunar" => &[(0.0,"SYN"),(14.7,"G_T"),(29.5,"PER")],
        "radio" => &[(2.7,"CMB"),(21.0,"H_I"),(40.0,"SDR")],
        "seismic" => &[(0.5,"P"),(5.0,"S"),(15.0,"L")],
        _ => &[(0.0,"0"),(12.5,"10"),(25.0,"20"),(37.5,"30")],
    };
    let max_hz = 50.0f32;
    for &(hz, lbl) in bands {
        let frac = hz / max_hz;
        let y    = plot.max.y - frac * plot.height();
        painter.line_segment(
            [Pos2::new(plot.min.x, y), Pos2::new(plot.max.x, y)],
            Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,35)),
        );
        painter.text(
            Pos2::new(rect.min.x + 2.0, y - 5.0),
            egui::Align2::LEFT_TOP, lbl,
            FontId::new(9.0, FontFamily::Monospace),
            Color32::from_rgb(170, 170, 200),
        );
    }

    // Title
    painter.text(
        Pos2::new(plot.min.x + 4.0, rect.min.y + 3.0),
        egui::Align2::LEFT_TOP,
        format!("FREQ ANALYSIS — {}", layer_id.to_uppercase()),
        FontId::new(9.5, FontFamily::Proportional),
        Color32::from_rgb(170, 170, 200),
    );

    // Clip indicator (latest column = rightmost)
    let peak = spec_data.back().and_then(|c| c.iter().copied().reduce(f32::max)).unwrap_or(0.0);
    let peak_col = if peak > 0.85 { theme::ACCENT } else { Color32::from_rgb(80,200,80) };
    painter.circle_filled(
        Pos2::new(plot.max.x - 6.0, rect.min.y + 8.0),
        3.5, peak_col,
    );
}

// ══════════════════════ LAYER INSTRUMENTS ═════════════════════════════════ //

/// EEG — scalp topographic map (4-electrode Muse 2 layout) + band power bars
fn render_eeg_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.55;
    let topo_r   = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let bands_r  = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);

    // ── Scalp topographic map ─────────────────────────────────────────
    let cx  = topo_r.center().x;
    let cy  = topo_r.center().y + topo_r.height() * 0.05;
    let hr  = topo_r.height().min(topo_r.width()) * 0.40; // head radius

    // Background interpolated field from 4 electrode alpha powers
    let s   = layer.score as f32;
    let tp9  = s * 0.55 + 0.15 * (t * 0.8  + 0.0).sin()  as f32; // L temporal
    let af7  = s * 0.72 + 0.12 * (t * 0.65 + 1.1).sin()  as f32; // L frontal
    let af8  = s * 0.68 + 0.12 * (t * 0.65 + 2.2).sin()  as f32; // R frontal
    let tp10 = s * 0.50 + 0.15 * (t * 0.8  + 3.3).sin()  as f32; // R temporal

    // Electrode positions (normalized, then mapped to circle)
    let elec: &[(f32, f32, f32, &str)] = &[
        (-0.72, 0.10, tp9,  "TP9"),  // left temporal
        (-0.38, -0.65, af7, "AF7"), // left frontal
        ( 0.38, -0.65, af8, "AF8"), // right frontal
        ( 0.72, 0.10, tp10, "TP10"),// right temporal
    ];

    // Scalp color field — blend 4 gaussian kernels
    let grid_n = 32usize;
    let cell_w = hr * 2.0 / grid_n as f32;
    for gi in 0..grid_n {
        for gj in 0..grid_n {
            let gx = -1.0 + (gi as f32 + 0.5) / grid_n as f32 * 2.0;
            let gy = -1.0 + (gj as f32 + 0.5) / grid_n as f32 * 2.0;
            if gx*gx + gy*gy > 1.02 { continue; } // only inside head circle
            let mut val = 0.0f32;
            let mut wsum = 0.0f32;
            for &(ex, ey, pow, _) in elec {
                let d2 = (gx - ex).powi(2) + (gy - ey).powi(2);
                let w  = (-d2 * 3.0).exp();
                val  += pow * w;
                wsum += w;
            }
            if wsum > 1e-6 { val /= wsum; }
            let val = val.clamp(0.0, 1.0);
            // Blue (low alpha) → red (high alpha)
            let col = if val < 0.5 {
                let t = val / 0.5;
                Color32::from_rgba_unmultiplied(
                    theme::lerp(20, 30, t), theme::lerp(20, 100, t), theme::lerp(180, 220, t), 200)
            } else {
                let t = (val - 0.5) / 0.5;
                Color32::from_rgba_unmultiplied(
                    theme::lerp(30, 255, t), theme::lerp(100, 32, t), theme::lerp(220, 72, t), 200)
            };
            let px = cx + gx * hr;
            let py = cy + gy * hr;
            painter.rect_filled(
                Rect::from_center_size(Pos2::new(px, py), Vec2::splat(cell_w + 0.5)),
                Rounding::same(0.0), col);
        }
    }

    // Head outline + nose + ears
    painter.circle_stroke(Pos2::new(cx, cy), hr, Stroke::new(2.0, Color32::from_rgba_unmultiplied(255,255,255,200)));
    // Nose
    let nose = [Pos2::new(cx - 7.0, cy - hr + 6.0), Pos2::new(cx, cy - hr - 12.0), Pos2::new(cx + 7.0, cy - hr + 6.0)];
    painter.line_segment([nose[0], nose[1]], Stroke::new(1.5, Color32::from_rgba_unmultiplied(255,255,255,200)));
    painter.line_segment([nose[1], nose[2]], Stroke::new(1.5, Color32::from_rgba_unmultiplied(255,255,255,200)));
    // Central cross-hair
    painter.line_segment([Pos2::new(cx - hr, cy), Pos2::new(cx + hr, cy)],
        Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,30)));
    painter.line_segment([Pos2::new(cx, cy - hr), Pos2::new(cx, cy + hr)],
        Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,30)));

    // Electrode markers + labels + alpha halos
    for &(ex, ey, pow, lbl) in elec {
        let px = cx + ex * hr;
        let py = cy + ey * hr;
        let halo_r = 8.0 + pow * 18.0;
        let halo_col = Color32::from_rgba_unmultiplied(255, (32.0 + 200.0*(1.0-pow)) as u8, 72, 60);
        painter.circle_filled(Pos2::new(px, py), halo_r, halo_col);
        painter.circle_filled(Pos2::new(px, py), 4.5, Color32::WHITE);
        painter.circle_stroke(Pos2::new(px, py), 4.5, Stroke::new(1.0, theme::ACCENT));
        painter.text(Pos2::new(px, py - 14.0), egui::Align2::CENTER_BOTTOM,
            lbl, FontId::new(8.5, FontFamily::Monospace),
            Color32::from_rgb(200, 200, 220));
        painter.text(Pos2::new(px, py + 8.0), egui::Align2::CENTER_TOP,
            format!("{:.2}", pow), FontId::new(8.0, FontFamily::Monospace),
            Color32::from_rgba_unmultiplied(255,200,200,200));
    }

    // Title
    painter.text(topo_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "SCALP TOPOGRAPHY — α POWER (Muse 2: TP9 AF7 AF8 TP10)",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(160, 160, 200));

    // ── Band power bars (right panel) ─────────────────────────────────
    let bands: &[(&str, f64, Color32)] = &[
        ("δ  0–4 Hz",  (0.20 + (1.0-layer.score)*0.30).clamp(0.0,1.0), Color32::from_rgb(80, 80, 200)),
        ("θ  4–8 Hz",  (0.25 + (0.8-layer.score)*0.25).clamp(0.0,1.0), Color32::from_rgb(80, 180, 220)),
        ("α  8–13 Hz", layer.score.clamp(0.0, 1.0),                      theme::ACCENT),
        ("β 13–30 Hz", ((1.0-layer.score)*0.6 + 0.15).clamp(0.0,1.0),  theme::AMBER),
        ("γ 30–50 Hz", (layer.score*0.3).clamp(0.0,1.0),                theme::GREEN_OK),
    ];
    let bar_area = Rect::from_min_max(
        Pos2::new(bands_r.min.x + 8.0, bands_r.min.y + 26.0),
        Pos2::new(bands_r.max.x - 8.0, bands_r.max.y - 8.0));
    painter.text(bands_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "EEG BAND POWER", FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(160,160,200));
    let bar_h = (bar_area.height() / bands.len() as f32 - 6.0).max(10.0);
    for (i, (lbl, pwr, col)) in bands.iter().enumerate() {
        let y = bar_area.min.y + i as f32 * (bar_h + 6.0);
        painter.text(Pos2::new(bar_area.min.x, y + bar_h * 0.5), egui::Align2::LEFT_CENTER,
            *lbl, FontId::new(9.0, FontFamily::Monospace), Color32::from_rgb(160,160,200));
        let bx = bar_area.min.x + 80.0;
        let bw = bar_area.width() - 80.0;
        let bg_bar = Rect::from_min_size(Pos2::new(bx, y), Vec2::new(bw, bar_h));
        painter.rect_filled(bg_bar, Rounding::same(3.0),
            Color32::from_rgba_unmultiplied(255,255,255,12));
        let fill_w = bw * (*pwr as f32);
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(bx, y), Vec2::new(fill_w, bar_h)),
            Rounding::same(3.0), *col);
        painter.text(Pos2::new(bx + fill_w + 4.0, y + bar_h * 0.5),
            egui::Align2::LEFT_CENTER, format!("{:.2}", pwr),
            FontId::new(8.5, FontFamily::Monospace),
            Color32::from_rgba_unmultiplied(255,255,255,160));
    }
}

/// Quantum — Bloch sphere + density matrix + purity
fn render_quantum_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.45;
    let bloch_r = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let rho_r   = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);

    // ── Bloch sphere ──────────────────────────────────────────────────
    let cx = bloch_r.center().x;
    let cy = bloch_r.center().y;
    let r  = bloch_r.height().min(bloch_r.width()) * 0.38;

    // Sphere outline (isometric 3D illusion)
    painter.circle_stroke(Pos2::new(cx, cy), r,
        Stroke::new(1.5, Color32::from_rgba_unmultiplied(255,255,255,80)));
    // Equatorial ellipse
    let eq_pts: Vec<Pos2> = (0..=64).map(|i| {
        let a = i as f32 / 64.0 * std::f32::consts::TAU;
        Pos2::new(cx + r * a.cos(), cy + r * 0.38 * a.sin())
    }).collect();
    for i in 0..eq_pts.len()-1 {
        painter.line_segment([eq_pts[i], eq_pts[i+1]],
            Stroke::new(0.8, Color32::from_rgba_unmultiplied(255,255,255,40)));
    }
    // Vertical axis line
    painter.line_segment([Pos2::new(cx, cy - r - 12.0), Pos2::new(cx, cy + r + 4.0)],
        Stroke::new(0.8, Color32::from_rgba_unmultiplied(255,255,255,50)));
    // Axis labels
    painter.text(Pos2::new(cx, cy - r - 14.0), egui::Align2::CENTER_BOTTOM,
        "|0⟩", FontId::new(11.0, FontFamily::Monospace), Color32::from_rgb(100, 180, 255));
    painter.text(Pos2::new(cx, cy + r + 6.0), egui::Align2::CENTER_TOP,
        "|1⟩", FontId::new(11.0, FontFamily::Monospace), Color32::from_rgb(255, 100, 100));
    painter.text(Pos2::new(cx + r + 4.0, cy + 4.0), egui::Align2::LEFT_CENTER,
        "|+⟩", FontId::new(9.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,120));

    // State vector: θ from score (0=north=|0⟩, π=south=|1⟩), φ from time
    let theta = layer.score as f32 * std::f32::consts::PI;
    let phi   = (t as f32 * 0.4) % std::f32::consts::TAU;
    // Isometric projection of Bloch vector
    let bx = r * theta.sin() * phi.cos() * 0.9;  // x component
    let by = r * theta.cos();                       // z → y on screen
    let bz = r * theta.sin() * phi.sin() * 0.4;   // y → depth (compressed)
    let tip = Pos2::new(cx + bx, cy - by + bz);

    // State vector arrow
    painter.line_segment([Pos2::new(cx, cy), tip],
        Stroke::new(2.5, theme::ACCENT));
    painter.circle_filled(tip, 5.5, theme::ACCENT);
    painter.circle_stroke(tip, 5.5, Stroke::new(1.0, Color32::WHITE));

    // Dashed projection lines
    painter.line_segment([Pos2::new(cx, cy), Pos2::new(cx + bx, cy + bz)],
        Stroke::new(0.8, Color32::from_rgba_unmultiplied(255,32,72,80)));
    painter.line_segment([Pos2::new(cx + bx, cy + bz), tip],
        Stroke::new(0.8, Color32::from_rgba_unmultiplied(255,32,72,80)));

    // Title + coords
    painter.text(bloch_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "BLOCH SPHERE — QUBIT STATE |ψ⟩",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(160,160,200));
    painter.text(Pos2::new(bloch_r.min.x + 8.0, bloch_r.max.y - 30.0),
        egui::Align2::LEFT_TOP,
        format!("θ={:.3}π  φ={:.3}π", theta / std::f32::consts::PI, phi / std::f32::consts::PI),
        FontId::new(9.5, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,200,200,200));

    // ── Density matrix ρ (2×2) ────────────────────────────────────────
    painter.text(rho_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "DENSITY MATRIX  ρ = |ψ⟩⟨ψ|",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(160,160,200));

    let s = layer.score as f32;
    let rho = [
        [theta.cos().powi(2) / 2.0 + 0.5 * s, theta.sin() * theta.cos() * 0.5],
        [theta.sin() * theta.cos() * 0.5,       theta.sin().powi(2) / 2.0 + 0.5 * (1.0-s)],
    ];
    let cell_sz = (rho_r.width() * 0.35).min(rho_r.height() * 0.28);
    let mat_x   = rho_r.min.x + 20.0;
    let mat_y   = rho_r.min.y + 36.0;
    let labels  = ["|0⟩", "|1⟩"];

    for i in 0..2 {
        for j in 0..2 {
            let x = mat_x + j as f32 * (cell_sz + 4.0);
            let y = mat_y + i as f32 * (cell_sz + 4.0);
            let v = rho[i][j].abs().clamp(0.0, 1.0);
            let col = Color32::from_rgba_unmultiplied(
                (30.0 + 225.0 * v) as u8,
                (32.0 + 100.0 * v) as u8,
                (72.0 + 80.0  * (1.0-v)) as u8,
                200);
            let cell = Rect::from_min_size(Pos2::new(x, y), Vec2::splat(cell_sz));
            painter.rect_filled(cell, Rounding::same(4.0), col);
            painter.rect_stroke(cell, Rounding::same(4.0),
                Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,40)));
            painter.text(cell.center(), egui::Align2::CENTER_CENTER,
                format!("{:.3}", rho[i][j]),
                FontId::new(10.0, FontFamily::Monospace), Color32::WHITE);
        }
        // Row/col labels
        painter.text(Pos2::new(mat_x - 4.0, mat_y + i as f32*(cell_sz+4.0) + cell_sz*0.5),
            egui::Align2::RIGHT_CENTER, labels[i],
            FontId::new(9.0, FontFamily::Monospace), Color32::from_rgb(160,180,220));
        painter.text(Pos2::new(mat_x + i as f32*(cell_sz+4.0) + cell_sz*0.5, mat_y - 4.0),
            egui::Align2::CENTER_BOTTOM, labels[i],
            FontId::new(9.0, FontFamily::Monospace), Color32::from_rgb(160,180,220));
    }

    // Purity + entropy
    let purity = rho[0][0].powi(2) + 2.0*rho[0][1].powi(2) + rho[1][1].powi(2);
    let entropy = -((purity).max(1e-9).ln()) * 0.5; // approx von Neumann
    let my = mat_y + 2.0*(cell_sz+4.0) + 16.0;
    let metrics = [
        ("Purity   Tr(ρ²)", format!("{:.4}", purity), theme::BLUE_INFO),
        ("Entropy  S(ρ)",   format!("{:.4}", entropy), theme::AMBER),
        ("Fidelity ⟨0|ρ|0⟩", format!("{:.4}", rho[0][0]), theme::GREEN_OK),
    ];
    for (i, (lbl, val, col)) in metrics.iter().enumerate() {
        let y = my + i as f32 * 22.0;
        painter.text(Pos2::new(rho_r.min.x + 12.0, y), egui::Align2::LEFT_TOP,
            *lbl, FontId::new(9.0, FontFamily::Monospace), Color32::from_rgb(160,160,200));
        painter.text(Pos2::new(rho_r.max.x - 12.0, y), egui::Align2::RIGHT_TOP,
            val, FontId::new(11.0, FontFamily::Monospace), *col);
    }
}

/// Solar — full solar disk + animated sunspots + SC25 progress bar
fn render_solar_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.50;
    let disk_r  = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let info_r  = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);

    let cx  = disk_r.center().x;
    let cy  = disk_r.center().y;
    let rad = disk_r.height().min(disk_r.width()) * 0.38;
    let s   = layer.score as f32;

    // Corona rings
    for k in 1..=4usize {
        let alpha = (80 - k * 16) as u8;
        painter.circle_stroke(Pos2::new(cx, cy), rad + k as f32 * 9.0,
            Stroke::new(2.0 - k as f32*0.3,
                Color32::from_rgba_unmultiplied(255, 180, 60, alpha)));
    }

    // Solar disk limb darkening gradient (concentric rings from bright center to darker edge)
    let n_rings = 16;
    for ri in (0..n_rings).rev() {
        let frac = ri as f32 / n_rings as f32;
        let r_ring = rad * (ri as f32 / n_rings as f32 + 1.0 / n_rings as f32);
        let bright = 1.0 - frac * frac * 0.45;
        let col = Color32::from_rgb(
            (255.0 * bright) as u8,
            (220.0 * bright) as u8,
            (100.0 * bright * 0.7) as u8,
        );
        painter.circle_filled(Pos2::new(cx, cy), r_ring, col);
    }

    // Sunspot groups
    let n_spots = 2 + (s * 10.0) as usize;
    for i in 0..n_spots {
        let seed  = i as f64 * 1.618 + 0.5;
        let angle = seed * 2.3 + t * (0.1 + seed * 0.05);
        let dist  = rad as f64 * 0.2 + rad as f64 * 0.6 * (seed * 0.7).fract();
        let sx    = cx + (dist * angle.cos()) as f32;
        let sy    = cy + (dist * angle.sin() * 0.8) as f32; // perspective squash
        // Clip to disk
        if (sx - cx).powi(2) + (sy - cy).powi(2) > (rad * 0.92).powi(2) { continue; }
        let spot_r = 3.0 + (i % 4) as f32 * 2.0;
        painter.circle_filled(Pos2::new(sx, sy), spot_r * 1.4,
            Color32::from_rgb(80, 40, 10));
        painter.circle_filled(Pos2::new(sx, sy), spot_r,
            Color32::from_rgb(40, 20, 5));
    }

    // Solar equator line
    painter.line_segment(
        [Pos2::new(cx - rad, cy), Pos2::new(cx + rad, cy)],
        Stroke::new(0.5, Color32::from_rgba_unmultiplied(255, 255, 255, 25)));

    // Title
    painter.text(disk_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "SOLAR DISK — HMI CONTINUUM (SC25)",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(200, 160, 80));

    // SSN badge
    let ssn_est = (s * 175.0) as u32;
    painter.text(Pos2::new(cx, disk_r.max.y - 12.0), egui::Align2::CENTER_BOTTOM,
        format!("SSN ≈ {ssn_est}   F10.7 ≈ {:.1}", 80.0 + s * 120.0),
        FontId::new(9.5, FontFamily::Monospace), Color32::from_rgb(200, 160, 80));

    // ── SC25 progress (right panel) ───────────────────────────────────
    painter.text(info_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "SOLAR CYCLE 25 — ACTIVITY TIMELINE",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(200, 160, 80));

    // SC25 started Dec 2019, predicted max ~2025, min ~2031
    // approximate phase 0→1 within cycle
    let cycle_progress = (s * 0.85 + 0.10).clamp(0.0, 1.0);
    let bar_y  = info_r.min.y + 36.0;
    let bar_x  = info_r.min.x + 12.0;
    let bar_w  = info_r.width() - 24.0;
    let bar_h  = 14.0f32;
    painter.rect_filled(Rect::from_min_size(Pos2::new(bar_x, bar_y), Vec2::new(bar_w, bar_h)),
        Rounding::same(4.0), Color32::from_rgba_unmultiplied(255,255,255,12));
    painter.rect_filled(Rect::from_min_size(Pos2::new(bar_x, bar_y),
        Vec2::new(bar_w * cycle_progress, bar_h)),
        Rounding::same(4.0), theme::AMBER);
    painter.text(Pos2::new(bar_x, bar_y - 2.0), egui::Align2::LEFT_BOTTOM,
        "Dec 2019", FontId::new(8.5, FontFamily::Monospace), Color32::from_rgb(160,130,80));
    painter.text(Pos2::new(bar_x + bar_w, bar_y - 2.0), egui::Align2::RIGHT_BOTTOM,
        "2031", FontId::new(8.5, FontFamily::Monospace), Color32::from_rgb(160,130,80));
    // Current position marker
    let cx_m = bar_x + bar_w * cycle_progress;
    painter.line_segment([Pos2::new(cx_m, bar_y - 4.0), Pos2::new(cx_m, bar_y + bar_h + 4.0)],
        Stroke::new(2.0, Color32::WHITE));
    painter.text(Pos2::new(cx_m, bar_y + bar_h + 6.0), egui::Align2::CENTER_TOP,
        "NOW", FontId::new(8.5, FontFamily::Monospace), Color32::WHITE);

    // Metrics
    let my = bar_y + bar_h + 32.0;
    let phase_str   = format!("{:.1}%", cycle_progress * 100.0);
    let ssn_str     = format!("~{ssn_est}");
    let f107_str    = format!("{:.1} sfu", 80.0 + s * 120.0);
    let act_str     = if s > 0.7 { "HIGH" } else if s > 0.4 { "MED" } else { "LOW" };
    let items: &[(&str, &str, Color32)] = &[
        ("Cycle",    "SC25",       theme::AMBER),
        ("Phase",    &phase_str,   theme::TEXT_MID),
        ("SSN",      &ssn_str,     theme::AMBER),
        ("F10.7",    &f107_str,    theme::BLUE_INFO),
        ("Activity", act_str,      theme::GREEN_OK),
    ];
    for (i, (lbl, val, col)) in items.iter().enumerate() {
        let y = my + i as f32 * 22.0;
        painter.text(Pos2::new(info_r.min.x + 12.0, y), egui::Align2::LEFT_TOP,
            *lbl, FontId::new(9.5, FontFamily::Monospace), Color32::from_rgb(160,130,80));
        painter.text(Pos2::new(info_r.max.x - 12.0, y), egui::Align2::RIGHT_TOP,
            *val, FontId::new(11.0, FontFamily::Monospace), *col);
    }
}

/// Geomagnetic — dipole field lines + Kp ring gauge + vector compass
fn render_geomagnetic_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.52;
    let field_r = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let kp_r    = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);

    let cx  = field_r.center().x;
    let cy  = field_r.center().y;
    let s   = layer.score as f32;
    let kp  = s * 9.0;

    // Dipole field lines (parametric: r = r0 * cos²(λ), x = r*cos(λ), y = r*sin(λ))
    let r0_list = [0.28, 0.38, 0.50, 0.65, 0.80];
    let scale   = field_r.height() * 0.42;
    for &r0_n in &r0_list {
        let r0 = r0_n * scale;
        let alpha_base = if r0_n < 0.45 { 200u8 } else if r0_n < 0.65 { 140 } else { 80 };
        let col = Color32::from_rgba_unmultiplied(30, 120, 255, alpha_base);
        // Northern hemisphere field line
        let pts: Vec<Pos2> = (-90..=90).map(|deg| {
            let lam = deg as f32 * std::f32::consts::PI / 180.0;
            let r   = r0 * lam.cos().powi(2);
            Pos2::new(cx + r * lam.cos(), cy - r * lam.sin())
        }).collect();
        for i in 0..pts.len()-1 {
            painter.line_segment([pts[i], pts[i+1]], Stroke::new(1.0, col));
        }
        // Southern hemisphere (mirror)
        let pts2: Vec<Pos2> = pts.iter().map(|p| Pos2::new(p.x, 2.0*cy - p.y)).collect();
        for i in 0..pts2.len()-1 {
            painter.line_segment([pts2[i], pts2[i+1]], Stroke::new(1.0, col));
        }
    }

    // Earth core dot
    painter.circle_filled(Pos2::new(cx, cy), 10.0, Color32::from_rgb(30, 120, 60));
    painter.circle_stroke(Pos2::new(cx, cy), 10.0, Stroke::new(1.5, Color32::from_rgb(60, 200, 100)));
    painter.circle_filled(Pos2::new(cx, cy), 4.0, Color32::WHITE);

    // Magnetic equator
    painter.line_segment([Pos2::new(field_r.min.x + 12.0, cy), Pos2::new(field_r.max.x - 12.0, cy)],
        Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,30)));

    // Animated geomagnetic pulse rings
    let pulse_t = (t * 1.4) % 3.0;
    let pulse_r = 15.0 + pulse_t as f32 * 40.0;
    let pulse_a = ((3.0 - pulse_t) / 3.0 * 120.0).clamp(0.0, 120.0) as u8;
    painter.circle_stroke(Pos2::new(cx, cy), pulse_r,
        Stroke::new(1.5, Color32::from_rgba_unmultiplied(30, 120, 255, pulse_a)));

    // Title
    painter.text(field_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "GEOMAGNETIC DIPOLE FIELD  B = ∇(V_dip)",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(100, 160, 255));
    painter.text(Pos2::new(cx, field_r.max.y - 10.0), egui::Align2::CENTER_BOTTOM,
        format!("B_total ≈ {:.1} nT   Kp = {:.1}", 42.3 + s * 8.0, kp),
        FontId::new(9.0, FontFamily::Monospace), Color32::from_rgb(100, 160, 255));

    // ── Kp ring gauge (right panel) ────────────────────────────────────
    painter.text(kp_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "Kp INDEX  (0–9 scale)",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(100, 160, 255));

    let kp_cx  = kp_r.center().x;
    let kp_cy  = kp_r.min.y + kp_r.height() * 0.42;
    let kp_rad = (kp_r.width().min(kp_r.height() * 0.8) * 0.38).min(70.0);

    // Kp arc background
    let n_arc = 9usize;
    for i in 0..n_arc {
        let frac   = i as f32 / n_arc as f32;
        let a_start = std::f32::consts::PI + frac * std::f32::consts::PI;
        let a_end   = std::f32::consts::PI + (i+1) as f32 / n_arc as f32 * std::f32::consts::PI;
        let col = if i < 3 { Color32::from_rgb(30,160,80) }
                  else if i < 6 { theme::AMBER }
                  else { theme::ACCENT };
        let n_seg = 8;
        for si in 0..n_seg {
            let a1 = a_start + si as f32 / n_seg as f32 * (a_end - a_start);
            let a2 = a_start + (si+1) as f32 / n_seg as f32 * (a_end - a_start);
            let p1 = Pos2::new(kp_cx + kp_rad * a1.cos(), kp_cy + kp_rad * a1.sin());
            let p2 = Pos2::new(kp_cx + kp_rad * a2.cos(), kp_cy + kp_rad * a2.sin());
            painter.line_segment([p1, p2], Stroke::new(8.0, Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 40)));
        }
        // Label
        let a_mid = (a_start + a_end) * 0.5;
        let lp = Pos2::new(kp_cx + (kp_rad + 14.0) * a_mid.cos(), kp_cy + (kp_rad + 14.0) * a_mid.sin());
        painter.text(lp, egui::Align2::CENTER_CENTER, format!("{i}"),
            FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,120));
    }

    // Kp needle
    let kp_angle = std::f32::consts::PI + (kp / 9.0) as f32 * std::f32::consts::PI;
    let needle_tip = Pos2::new(kp_cx + kp_rad * 0.85 * kp_angle.cos(),
                               kp_cy + kp_rad * 0.85 * kp_angle.sin());
    painter.line_segment([Pos2::new(kp_cx, kp_cy), needle_tip],
        Stroke::new(2.5, theme::ACCENT));
    painter.circle_filled(Pos2::new(kp_cx, kp_cy), 5.0, theme::ACCENT);

    let kp_col = if kp < 3.0 { theme::GREEN_OK } else if kp < 6.0 { theme::AMBER } else { theme::ACCENT };
    painter.text(Pos2::new(kp_cx, kp_cy + kp_rad + 18.0), egui::Align2::CENTER_TOP,
        format!("Kp = {:.1}", kp),
        FontId::new(16.0, FontFamily::Monospace), kp_col);

    // Schumann resonance indicators
    let my = kp_cy + kp_rad + 50.0;
    let schumann = [7.83f32, 14.3, 20.8, 27.3];
    painter.text(Pos2::new(kp_r.min.x + 8.0, my - 14.0), egui::Align2::LEFT_TOP,
        "Schumann resonances", FontId::new(9.0, FontFamily::Proportional), Color32::from_rgb(100,160,255));
    for (i, &hz) in schumann.iter().enumerate() {
        let x    = kp_r.min.x + 12.0;
        let y    = my + i as f32 * 18.0;
        let amp  = (0.5 + 0.4 * (t * hz as f64 * 0.1 + i as f64).sin()) as f32;
        let bw   = kp_r.width() - 24.0;
        painter.rect_filled(Rect::from_min_size(Pos2::new(x, y), Vec2::new(bw, 10.0)),
            Rounding::same(2.0), Color32::from_rgba_unmultiplied(30,120,255,20));
        painter.rect_filled(Rect::from_min_size(Pos2::new(x, y), Vec2::new(bw * amp, 10.0)),
            Rounding::same(2.0), Color32::from_rgba_unmultiplied(30,120,255,140));
        painter.text(Pos2::new(x + bw + 4.0, y + 5.0), egui::Align2::LEFT_CENTER,
            format!("{hz:.1}Hz"), FontId::new(8.0, FontFamily::Monospace), Color32::from_rgb(100,160,255));
    }
}

/// Lunar — orbital diagram + Moon phase disc + tidal force curve
fn render_lunar_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.52;
    let orb_r   = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let moon_r  = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);

    // ── Orbital diagram ───────────────────────────────────────────────
    let cx  = orb_r.center().x;
    let cy  = orb_r.center().y + 10.0;
    let s   = layer.score as f32;

    // Lunar orbit (slightly elliptical)
    let a = orb_r.width() * 0.38;
    let b = a * 0.97; // small eccentricity
    let n_pts = 64;
    let orbit_pts: Vec<Pos2> = (0..=n_pts).map(|i| {
        let ang = i as f32 / n_pts as f32 * std::f32::consts::TAU;
        Pos2::new(cx + a * ang.cos(), cy + b * ang.sin())
    }).collect();
    for i in 0..orbit_pts.len()-1 {
        painter.line_segment([orbit_pts[i], orbit_pts[i+1]],
            Stroke::new(0.8, Color32::from_rgba_unmultiplied(255,255,255,30)));
    }

    // Perigee/apogee labels
    painter.text(Pos2::new(cx + a + 6.0, cy), egui::Align2::LEFT_CENTER,
        "APO", FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,80));
    painter.text(Pos2::new(cx - a - 6.0, cy), egui::Align2::RIGHT_CENTER,
        "PERI", FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,80));

    // Earth at focus
    painter.circle_filled(Pos2::new(cx, cy), 12.0, Color32::from_rgb(30, 80, 180));
    painter.circle_stroke(Pos2::new(cx, cy), 12.0, Stroke::new(1.5, Color32::from_rgb(80, 160, 255)));
    painter.text(Pos2::new(cx, cy + 16.0), egui::Align2::CENTER_TOP, "⊕",
        FontId::new(9.0, FontFamily::Proportional), Color32::from_rgb(80, 160, 255));

    // Moon position on orbit
    let moon_angle = s * std::f32::consts::TAU + (t as f32 * 0.05);
    let moon_x = cx + a * moon_angle.cos();
    let moon_y = cy + b * moon_angle.sin();
    painter.circle_filled(Pos2::new(moon_x, moon_y), 8.0, Color32::from_rgb(200, 200, 210));
    painter.circle_stroke(Pos2::new(moon_x, moon_y), 8.0, Stroke::new(1.0, Color32::from_rgb(160,160,180)));

    // Earth-Moon line
    painter.line_segment([Pos2::new(cx, cy), Pos2::new(moon_x, moon_y)],
        Stroke::new(0.8, Color32::from_rgba_unmultiplied(255,255,255,40)));

    // Tidal force arrow (points toward Moon from Earth, scaled by 1/d²)
    let dist_km = 384400.0f32 - s * 20000.0;
    let tidal_force = 1.0 / (dist_km / 384400.0).powi(2);
    let arrow_len = 18.0 * tidal_force.clamp(0.5, 2.0);
    let dx = (moon_x - cx) / ((moon_x-cx).powi(2)+(moon_y-cy).powi(2)).sqrt();
    let dy = (moon_y - cy) / ((moon_x-cx).powi(2)+(moon_y-cy).powi(2)).sqrt();
    let tip = Pos2::new(cx + dx * arrow_len, cy + dy * arrow_len);
    painter.line_segment([Pos2::new(cx, cy), tip], Stroke::new(2.0, theme::AMBER));
    painter.circle_filled(tip, 3.0, theme::AMBER);

    // Distance label
    painter.text(Pos2::new(cx, orb_r.max.y - 10.0), egui::Align2::CENTER_BOTTOM,
        format!("Dist: {:.0} km   g_pert: {:.2e} m/s²", dist_km, 5.6e-8 * tidal_force),
        FontId::new(9.0, FontFamily::Monospace), Color32::from_rgb(180,180,210));

    painter.text(orb_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "LUNAR ORBIT — GRAVITATIONAL PERTURBATION",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(180,180,210));

    // ── Moon phase disc (right panel) ─────────────────────────────────
    painter.text(moon_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "LUNAR PHASE  &  TIDAL FORCE",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(180,180,210));

    let pcx  = moon_r.center().x;
    let pcy  = moon_r.min.y + moon_r.height() * 0.35;
    let prad = (moon_r.width() * 0.32).min(70.0);

    // Dark moon (base)
    painter.circle_filled(Pos2::new(pcx, pcy), prad, Color32::from_rgb(20, 20, 30));
    painter.circle_stroke(Pos2::new(pcx, pcy), prad, Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,60)));

    // Illuminated side (phase = 0 → new moon, 0.5 → full, 1 → new again)
    let phase = s; // 0..1
    // Simple: scan horizontal lines
    for li in 0..=32usize {
        let frac = li as f32 / 32.0;
        let y_off = prad * (2.0 * frac - 1.0);
        let half_w = (prad * prad - y_off * y_off).max(0.0).sqrt();
        // terminator x position: at phase 0 = right half, 0.5 = full, phase goes 0→1→0
        let illum = (phase * 2.0 * std::f32::consts::PI).cos() * 0.5 + 0.5;
        let term_x = pcx + half_w * (2.0 * illum - 1.0);
        // draw illuminated arc from left edge to terminator
        let x_start = if phase < 0.5 { pcx + half_w } else { pcx - half_w };
        let x_end   = term_x;
        if (x_end - x_start).abs() > 0.5 {
            painter.line_segment(
                [Pos2::new(x_start.min(x_end), pcy + y_off),
                 Pos2::new(x_start.max(x_end), pcy + y_off)],
                Stroke::new(1.5, Color32::from_rgb(220, 220, 230)));
        }
    }

    let phase_name = match (phase * 8.0) as usize % 8 {
        0 => "New Moon",   1 => "Waxing Crescent", 2 => "First Quarter",
        3 => "Waxing Gibbous", 4 => "Full Moon", 5 => "Waning Gibbous",
        6 => "Last Quarter", _ => "Waning Crescent",
    };
    painter.text(Pos2::new(pcx, pcy + prad + 10.0), egui::Align2::CENTER_TOP,
        phase_name, FontId::new(11.0, FontFamily::Proportional), Color32::WHITE);

    // Tidal force curve (small graph below)
    let gy = pcy + prad + 36.0;
    let gx = moon_r.min.x + 12.0;
    let gw = moon_r.width() - 24.0;
    let gh = 44.0f32;
    painter.rect_filled(Rect::from_min_size(Pos2::new(gx, gy), Vec2::new(gw, gh)),
        Rounding::same(3.0), Color32::from_rgba_unmultiplied(255,255,255,8));
    painter.text(Pos2::new(gx, gy - 2.0), egui::Align2::LEFT_BOTTOM,
        "Tidal force proxy", FontId::new(8.5, FontFamily::Monospace), Color32::from_rgb(140,140,180));
    let mut prev: Option<Pos2> = None;
    for i in 0..=40usize {
        let p  = i as f32 / 40.0;
        let d  = 384400.0f32 - p * 40000.0 * s;
        let f  = 1.0 / (d / 384400.0).powi(2);
        let px = gx + p * gw;
        let py = gy + gh - f.clamp(0.5, 2.0) / 2.0 * gh;
        let pt = Pos2::new(px, py);
        if let Some(pr) = prev { painter.line_segment([pr, pt], Stroke::new(1.2, theme::AMBER)); }
        prev = Some(pt);
    }
}

/// Seismic — waveform display + global event heatmap + magnitude histogram
fn render_seismic_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.55;
    let wave_r  = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let map_r   = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);
    let s       = layer.score as f32;

    // ── Seismogram waveform ───────────────────────────────────────────
    painter.text(wave_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "SEISMOGRAM — BROADBAND P/S/L",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(180,100,100));

    let plot = Rect::from_min_max(
        Pos2::new(wave_r.min.x + 8.0, wave_r.min.y + 26.0),
        Pos2::new(wave_r.max.x - 8.0, wave_r.max.y - 20.0));

    // Zero line
    let mid_y = plot.center().y;
    painter.line_segment([Pos2::new(plot.min.x, mid_y), Pos2::new(plot.max.x, mid_y)],
        Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,30)));

    // P-wave (high freq)
    let mut prev: Option<Pos2> = None;
    for i in 0..=200usize {
        let x_frac = i as f32 / 200.0;
        let tx = t * 4.0 + x_frac as f64 * 3.0;
        let p_wave = (tx * 7.83).sin() * 0.3 * s as f64
            + (tx * 14.3).sin() * 0.15 * s as f64
            + (tx * 0.5).sin() * 0.08;
        let px = plot.min.x + x_frac * plot.width();
        let py = mid_y - (p_wave as f32 * plot.height() * 0.38).clamp(-plot.height()*0.45, plot.height()*0.45);
        let pt = Pos2::new(px, py);
        if let Some(pr) = prev {
            painter.line_segment([pr, pt], Stroke::new(1.0, theme::BLUE_INFO));
        }
        prev = Some(pt);
    }

    // S-wave (lower freq, larger amplitude when score high)
    let mut prev2: Option<Pos2> = None;
    for i in 0..=200usize {
        let x_frac = i as f32 / 200.0;
        let tx = t * 2.0 + x_frac as f64 * 3.0;
        let s_wave = (tx * 3.5).sin() * 0.5 * s as f64
            + (tx * 1.2).cos() * 0.2;
        let px = plot.min.x + x_frac * plot.width();
        let py = mid_y - (s_wave as f32 * plot.height() * 0.35).clamp(-plot.height()*0.42, plot.height()*0.42);
        let pt = Pos2::new(px, py);
        if let Some(pr) = prev2 {
            painter.line_segment([pr, pt], Stroke::new(1.0, theme::ACCENT));
        }
        prev2 = Some(pt);
    }

    // Phase labels
    for (label, x_frac, col) in [("P", 0.18f32, theme::BLUE_INFO), ("S", 0.42, theme::ACCENT), ("L", 0.68, theme::AMBER)] {
        let px = plot.min.x + x_frac * plot.width();
        painter.line_segment([Pos2::new(px, plot.min.y), Pos2::new(px, plot.max.y)],
            Stroke::new(0.8, Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 60)));
        painter.text(Pos2::new(px + 3.0, plot.min.y + 3.0), egui::Align2::LEFT_TOP,
            label, FontId::new(9.5, FontFamily::Monospace), col);
    }

    // Max Mag indicator — prefer real fields over simulated estimate
    let real_max_mag = layer.fields.iter()
        .find(|(k, _, _)| k == "Max_Mag")
        .and_then(|(_, v, _)| v.parse::<f32>().ok());
    let real_events = layer.fields.iter()
        .find(|(k, _, _)| k == "Events_24h")
        .and_then(|(_, v, _)| v.parse::<u32>().ok());
    let real_e_rel = layer.fields.iter()
        .find(|(k, _, _)| k == "E_rel")
        .and_then(|(_, v, _)| v.parse::<f64>().ok());
    let (max_mag_disp, evt_disp) = match (real_max_mag, real_events) {
        (Some(m), Some(e)) => (m, e),
        _ => (4.5 + s * 1.5, (100.0 + s * 200.0) as u32),
    };
    let source_tag = if layer.has_real_data { "USGS" } else { "SIM" };
    let e_str = real_e_rel.map(|e| format!("  E_rel {e:.4}")).unwrap_or_default();
    painter.text(Pos2::new(wave_r.center().x, wave_r.max.y - 8.0), egui::Align2::CENTER_BOTTOM,
        format!("[{source_tag}] Max Mw {max_mag_disp:.1}   Events/24h: {evt_disp}{e_str}"),
        FontId::new(9.0, FontFamily::Monospace), Color32::from_rgb(200, 120, 120));

    // ── Global event heatmap (right panel) ───────────────────────────
    painter.text(map_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "GLOBAL SEISMIC DENSITY",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(180,100,100));

    let grid = Rect::from_min_max(
        Pos2::new(map_r.min.x + 8.0, map_r.min.y + 26.0),
        Pos2::new(map_r.max.x - 8.0, map_r.max.y - 8.0));
    let (cols, rows) = (18usize, 9usize);
    let cw = grid.width() / cols as f32;
    let rh = grid.height() / rows as f32;

    // Ring-of-fire emphasis
    for ri in 0..rows {
        for ci in 0..cols {
            let lon = (ci as f32 / cols as f32 - 0.5) * 2.0; // -1..1
            let lat = (ri as f32 / rows as f32 - 0.5) * 2.0;
            // Pacific ring of fire (high seismicity near Pacific rim)
            let ring = (-((lon.powi(2) + lat.powi(2) - 0.6).powi(2) * 8.0)).exp();
            // Random regional seed + score + time
            let seed = (ci as f64 * 1.3 + ri as f64 * 2.7 + t * 0.2).sin().abs();
            let intensity = (ring * 0.7 + seed as f32 * 0.3 + s * 0.3).clamp(0.0, 1.0);
            let col = if intensity > 0.65 { theme::ACCENT }
                      else if intensity > 0.35 { theme::AMBER }
                      else { Color32::from_rgba_unmultiplied(255, 32, 72, (intensity * 40.0) as u8) };
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(grid.min.x + ci as f32 * cw, grid.min.y + ri as f32 * rh),
                    Vec2::new(cw - 1.0, rh - 1.0)),
                Rounding::same(1.0), col);
        }
    }
    // Map border
    painter.rect_stroke(grid, Rounding::same(2.0), Stroke::new(0.5, theme::BORDER));
}

/// Radio / CMB — RTL-SDR waterfall + CMB angular power spectrum
fn render_radio_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.50;
    let wf_r    = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let cmb_r   = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);
    let s       = layer.score as f32;

    // ── RTL-SDR Waterfall ─────────────────────────────────────────────
    let sdr_noise = layer.fields.iter().find(|(k,_,_)| k == "SDR_Floor")
        .map(|(_, v, _)| v.as_str().to_owned())
        .unwrap_or_else(|| "-110.0 dB".into());
    let sdr_src = layer.fields.iter().find(|(k,_,_)| k == "Deficit")
        .map(|_| if layer.has_real_data { "LIVE" } else { "proxy" })
        .unwrap_or("proxy");
    // has_rtlsdr flag comes through source field name containing "RTL-SDR"
    let has_rtl = layer.fields.iter().any(|(k, v, _)| k == "SDR_Floor" && v != "-110.0 dB");
    let wf_label = if has_rtl {
        format!("RTL-SDR WATERFALL  [{sdr_src}]  floor {sdr_noise}")
    } else {
        format!("RTL-SDR WATERFALL  [proxy Planck]  floor {sdr_noise}")
    };
    painter.text(wf_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        wf_label,
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(120,180,120));

    let wf_plot = Rect::from_min_max(
        Pos2::new(wf_r.min.x + 8.0, wf_r.min.y + 24.0),
        Pos2::new(wf_r.max.x - 8.0, wf_r.max.y - 8.0));
    let (wf_cols, wf_rows) = (60usize, 40usize);
    let wcw = wf_plot.width()  / wf_cols as f32;
    let wrh = wf_plot.height() / wf_rows as f32;

    for wi in 0..wf_cols {
        for fi in 0..wf_rows {
            let freq = fi as f64 * 1.25;
            let time_offset = (wf_cols - wi) as f64 * 0.05;
            let t_local = t - time_offset;
            let cmb_sig = 0.9 * (-(freq - 2.7).powi(2) / 1.5).exp();
            let hi_sig  = 0.7 * (-(freq - 21.1).powi(2) / 0.5).exp();
            let noise   = 0.05 * (t_local * 7.0 + fi as f64 * 1.3 + wi as f64 * 0.8).sin().abs();
            let power   = ((cmb_sig + hi_sig) * s as f64 + noise).clamp(0.0, 1.0) as f32;
            let col = theme::spec_color(power);
            painter.rect_filled(
                Rect::from_min_size(Pos2::new(wf_plot.min.x + wi as f32 * wcw,
                    wf_plot.max.y - (fi + 1) as f32 * wrh),
                    Vec2::new(wcw + 0.5, wrh + 0.5)),
                Rounding::same(0.0), col);
        }
    }
    // Frequency labels
    for (hz, lbl) in [(2.7f32, "CMB"), (21.1, "HI"), (40.0, "OHM")] {
        let frac = hz / 50.0;
        let fy   = wf_plot.max.y - frac * wf_plot.height();
        painter.line_segment([Pos2::new(wf_plot.min.x, fy), Pos2::new(wf_plot.max.x, fy)],
            Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,40)));
        painter.text(Pos2::new(wf_plot.min.x - 2.0, fy), egui::Align2::RIGHT_CENTER,
            lbl, FontId::new(8.0, FontFamily::Monospace), Color32::from_rgb(120,180,120));
    }

    // ── CMB Angular Power Spectrum ────────────────────────────────────
    painter.text(cmb_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "CMB POWER SPECTRUM  C_l",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(120,180,120));

    let cmb_plot = Rect::from_min_max(
        Pos2::new(cmb_r.min.x + 10.0, cmb_r.min.y + 26.0),
        Pos2::new(cmb_r.max.x - 8.0,  cmb_r.max.y - 20.0));

    let mut prev: Option<Pos2> = None;
    let n_pts = 60;
    for i in 0..=n_pts {
        let l = i as f32 / n_pts as f32 * 1200.0;
        let cl = 1.00 * (-(l-220.0).powi(2) / 12000.0).exp()
               + 0.45 * (-(l-540.0).powi(2) /  9000.0).exp()
               + 0.25 * (-(l-810.0).powi(2) / 14000.0).exp()
               + 0.12 * (-(l-1050.0).powi(2)/16000.0).exp();
        let cl = (cl * (0.4 + s * 0.6)).clamp(0.0, 1.0);
        let px = cmb_plot.min.x + (i as f32 / n_pts as f32) * cmb_plot.width();
        let py = cmb_plot.max.y - cl * cmb_plot.height() * 0.88;
        let pt = Pos2::new(px, py);
        if let Some(pr) = prev {
            painter.line_segment([pr, pt], Stroke::new(1.5, theme::BLUE_INFO));
        }
        prev = Some(pt);
    }
    // Acoustic peak labels
    for (l_val, lbl) in [(220.0f32, "1st"), (540.0, "2nd"), (810.0, "3rd")] {
        let px = cmb_plot.min.x + (l_val / 1200.0) * cmb_plot.width();
        painter.line_segment([Pos2::new(px, cmb_plot.min.y), Pos2::new(px, cmb_plot.max.y)],
            Stroke::new(0.6, Color32::from_rgba_unmultiplied(100,180,255,50)));
        painter.text(Pos2::new(px, cmb_plot.min.y + 2.0), egui::Align2::CENTER_TOP,
            lbl, FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(100,180,255,160));
    }
    painter.text(Pos2::new(cmb_plot.min.x, cmb_plot.max.y + 4.0), egui::Align2::LEFT_TOP,
        "l=0", FontId::new(8.0, FontFamily::Monospace), Color32::from_rgb(100,140,100));
    painter.text(Pos2::new(cmb_plot.max.x, cmb_plot.max.y + 4.0), egui::Align2::RIGHT_TOP,
        "l=1200", FontId::new(8.0, FontFamily::Monospace), Color32::from_rgb(100,140,100));
    painter.text(Pos2::new(cmb_plot.min.x, cmb_plot.min.y), egui::Align2::LEFT_TOP,
        format!("T_CMB=2.7255K  Quadrupole:{:.4}", 0.85 + layer.score * 0.05),
        FontId::new(8.5, FontFamily::Monospace), Color32::from_rgb(120,180,120));
}

/// Magnon — spin wave dispersion E(k) + magnon density of states
fn render_magnon_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, _t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.55;
    let disp_r  = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let dos_r   = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);
    let s       = layer.score as f32;

    // ── Spin-wave dispersion relation ─────────────────────────────────
    painter.text(disp_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "MAGNON DISPERSION  E(k) = Δ + Jk²",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(200, 100, 255));

    let plot = Rect::from_min_max(
        Pos2::new(disp_r.min.x + 24.0, disp_r.min.y + 26.0),
        Pos2::new(disp_r.max.x - 8.0,  disp_r.max.y - 24.0));

    // Axes
    painter.line_segment([Pos2::new(plot.min.x, plot.max.y), Pos2::new(plot.max.x, plot.max.y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,80)));
    painter.line_segment([Pos2::new(plot.min.x, plot.min.y), Pos2::new(plot.min.x, plot.max.y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,80)));
    painter.text(Pos2::new(plot.center().x, plot.max.y + 4.0), egui::Align2::CENTER_TOP,
        "k  (BZ boundary → π/a)", FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,120));
    painter.text(Pos2::new(plot.min.x - 4.0, plot.center().y), egui::Align2::RIGHT_CENTER,
        "E", FontId::new(9.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,120));

    // Acoustic magnon branch: E = Δ + J*k² (ferromagnetic, gap Δ at k=0)
    let delta = 0.05 + (1.0 - s) * 0.15;
    let j_ex  = 0.60 + s * 0.35;
    let mut prev: Option<Pos2> = None;
    for i in 0..=80usize {
        let k    = i as f32 / 80.0;
        let e    = (delta + j_ex * k * k * std::f32::consts::PI.powi(2)).clamp(0.0, 1.05);
        let px   = plot.min.x + k * plot.width();
        let py   = plot.max.y - e.min(1.0) * plot.height();
        let pt   = Pos2::new(px, py);
        if let Some(pr) = prev { painter.line_segment([pr, pt], Stroke::new(2.0, theme::ACCENT)); }
        prev = Some(pt);
    }

    // Optical branch (antiferromagnetic, higher energy)
    let mut prev2: Option<Pos2> = None;
    for i in 0..=80usize {
        let k  = i as f32 / 80.0;
        let e  = (0.60 + (1.0 - k) * (j_ex * 0.8)).clamp(0.0, 1.05);
        let px = plot.min.x + k * plot.width();
        let py = plot.max.y - e.min(1.0) * plot.height();
        let pt = Pos2::new(px, py);
        if let Some(pr) = prev2 { painter.line_segment([pr, pt], Stroke::new(1.2, theme::BLUE_INFO)); }
        prev2 = Some(pt);
    }

    // Gap annotation
    let gap_y = plot.max.y - delta * plot.height();
    painter.line_segment([Pos2::new(plot.min.x, gap_y), Pos2::new(plot.min.x + 16.0, gap_y)],
        Stroke::new(0.8, Color32::from_rgba_unmultiplied(255,200,0,120)));
    painter.text(Pos2::new(plot.min.x + 18.0, gap_y), egui::Align2::LEFT_CENTER,
        format!("Δ={:.3}", delta), FontId::new(8.5, FontFamily::Monospace), theme::AMBER);

    painter.text(Pos2::new(plot.max.x - 4.0, plot.min.y + 4.0), egui::Align2::RIGHT_TOP,
        "optical", FontId::new(8.0, FontFamily::Monospace), theme::BLUE_INFO);
    painter.text(Pos2::new(plot.max.x - 4.0, plot.min.y + 18.0), egui::Align2::RIGHT_TOP,
        "acoustic", FontId::new(8.0, FontFamily::Monospace), theme::ACCENT);

    // ── Density of states (right) ─────────────────────────────────────
    painter.text(dos_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "MAGNON  DoS  g(E)",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(200,100,255));

    let dp = Rect::from_min_max(
        Pos2::new(dos_r.min.x + 12.0, dos_r.min.y + 26.0),
        Pos2::new(dos_r.max.x - 8.0,  dos_r.max.y - 24.0));
    let n_bins = 32usize;
    let bw = dp.width() / n_bins as f32;
    for bi in 0..n_bins {
        let e    = bi as f32 / n_bins as f32;
        // 3D magnon DoS: van Hove singularity at band edge
        let dos  = (e + 0.01).sqrt() * (1.0 - e * 0.8).max(0.0).powi(2) * s * 2.0;
        let bh   = (dos * dp.height()).clamp(0.0, dp.height());
        let bx   = dp.min.x + bi as f32 * bw;
        let col  = Color32::from_rgba_unmultiplied(180, 60, 255, 180);
        painter.rect_filled(
            Rect::from_min_size(Pos2::new(bx, dp.max.y - bh), Vec2::new(bw - 1.0, bh)),
            Rounding::same(1.0), col);
    }
    painter.line_segment([Pos2::new(dp.min.x, dp.max.y), Pos2::new(dp.max.x, dp.max.y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,80)));

    // Prefer real Lindblad/Terra fields over score-derived estimates
    let f_t2      = layer.fields.iter().find(|(k,_,_)| k == "T2_us")
                        .map(|(_, v, _)| v.clone());
    let f_purity  = layer.fields.iter().find(|(k,_,_)| k == "Purity")
                        .map(|(_, v, _)| v.clone());
    let f_singlet = layer.fields.iter().find(|(k,_,_)| k == "Singlet")
                        .map(|(_, v, _)| v.clone());
    let f_entropy = layer.fields.iter().find(|(k,_,_)| k == "Entropy")
                        .map(|(_, v, _)| v.clone());
    let src_tag   = if layer.has_real_data { "Terra/Lindblad" } else { "SIM" };
    let metrics = [
        ("T2_eff", f_t2.unwrap_or_else(|| format!("{:.2} μs", 8.4 + s * 3.0)), theme::BLUE_INFO),
        ("Singlet",f_singlet.unwrap_or_else(|| format!("{:.4}", 0.3 + s * 0.4)), theme::ACCENT),
        ("Purity", f_purity.unwrap_or_else(|| format!("{:.4}", 0.90 + s * 0.08)), theme::GREEN_OK),
        ("S_vN",   f_entropy.unwrap_or_else(|| format!("{:.4}", (1.0 - s) * 0.6)), theme::AMBER),
    ];
    let my = dp.max.y + 12.0;
    for (i, (lbl, val, col)) in metrics.iter().enumerate() {
        let y = my + i as f32 * 18.0;
        painter.text(Pos2::new(dos_r.min.x + 12.0, y), egui::Align2::LEFT_TOP,
            *lbl, FontId::new(9.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,140));
        painter.text(Pos2::new(dos_r.max.x - 8.0, y), egui::Align2::RIGHT_TOP,
            val, FontId::new(10.0, FontFamily::Monospace), *col);
    }
    painter.text(Pos2::new(dos_r.center().x, dos_r.max.y - 6.0), egui::Align2::CENTER_BOTTOM,
        format!("[{}]", src_tag),
        FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(200,100,255,120));
}

/// Quantum Lab — partition function Z(β) + free energy landscape
fn render_quantum_lab_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, _t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.55;
    let z_r  = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let fe_r = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);
    let s    = layer.score as f32;

    // ── Partition function ln Z(β) ────────────────────────────────────
    painter.text(z_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "PARTITION FUNCTION  ln Z(β)",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(80, 200, 160));

    let zp = Rect::from_min_max(
        Pos2::new(z_r.min.x + 24.0, z_r.min.y + 26.0),
        Pos2::new(z_r.max.x - 8.0,  z_r.max.y - 24.0));
    painter.line_segment([Pos2::new(zp.min.x, zp.max.y), Pos2::new(zp.max.x, zp.max.y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,60)));
    painter.line_segment([Pos2::new(zp.min.x, zp.min.y), Pos2::new(zp.min.x, zp.max.y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,60)));
    painter.text(Pos2::new(zp.center().x, zp.max.y + 4.0), egui::Align2::CENTER_TOP,
        "β = 1/kT  (inverse temperature)", FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,100));

    // ln Z = N * ln(2 cosh(β * J/2)) — Ising-like
    let n_sites = 16.0f32;
    let j_coupling = 0.4 + s * 0.6;
    let mut prev: Option<Pos2> = None;
    for i in 0..=80usize {
        let b  = 0.1 + i as f32 / 80.0 * 4.9; // β from 0.1 to 5.0
        let lnz = n_sites * (2.0 * (b * j_coupling * 0.5).cosh()).ln();
        let lnz_norm = (lnz / (n_sites * (2.0f32).ln())).clamp(0.0, 1.05);
        let px = zp.min.x + (i as f32 / 80.0) * zp.width();
        let py = zp.max.y - lnz_norm.min(1.0) * zp.height();
        let pt = Pos2::new(px, py);
        if let Some(pr) = prev { painter.line_segment([pr, pt], Stroke::new(2.0, theme::GREEN_OK)); }
        prev = Some(pt);
    }

    // β current marker
    let beta_now = 2.5 - s * 1.0;
    let bx = zp.min.x + ((beta_now - 0.1) / 4.9) * zp.width();
    painter.line_segment([Pos2::new(bx, zp.min.y), Pos2::new(bx, zp.max.y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,200,0,100)));
    painter.text(Pos2::new(bx + 2.0, zp.min.y + 4.0), egui::Align2::LEFT_TOP,
        format!("β={:.2}", beta_now), FontId::new(8.5, FontFamily::Monospace), theme::AMBER);

    // ── Free energy landscape F(order param) ─────────────────────────
    painter.text(fe_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "FREE ENERGY  F(m) — Landau",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(80, 200, 160));

    let fp = Rect::from_min_max(
        Pos2::new(fe_r.min.x + 12.0, fe_r.min.y + 26.0),
        Pos2::new(fe_r.max.x - 8.0,  fe_r.max.y - 24.0));
    painter.line_segment([Pos2::new(fp.center().x, fp.min.y), Pos2::new(fp.center().x, fp.max.y)],
        Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,30)));
    painter.line_segment([Pos2::new(fp.min.x, fp.center().y), Pos2::new(fp.max.x, fp.center().y)],
        Stroke::new(0.5, Color32::from_rgba_unmultiplied(255,255,255,30)));

    // F = a*m² + b*m⁴ — double well when a < 0 (ordered phase)
    let a_coeff = 0.5 - s * 1.2; // negative when score > 0.42
    let b_coeff = 0.8f32;
    let f_max = b_coeff * 0.25f32.powi(2) + a_coeff.abs() * 0.1; // scale
    let mut prev2: Option<Pos2> = None;
    for i in 0..=80usize {
        let m  = (i as f32 / 80.0 - 0.5) * 2.0; // -1..1
        let f  = a_coeff * m * m + b_coeff * m.powi(4);
        let f_norm = (f / (f_max.abs() + 0.01) * 0.4 + 0.5).clamp(0.0, 1.0);
        let px = fp.min.x + (i as f32 / 80.0) * fp.width();
        let py = fp.min.y + f_norm * fp.height();
        let pt = Pos2::new(px, py);
        if let Some(pr) = prev2 { painter.line_segment([pr, pt], Stroke::new(2.0, theme::BLUE_INFO)); }
        prev2 = Some(pt);
    }

    let phase_str = if a_coeff < 0.0 { "ORDERED  (SSB)" } else { "DISORDERED" };
    painter.text(Pos2::new(fp.center().x, fp.max.y + 6.0), egui::Align2::CENTER_TOP,
        format!("Phase: {}   a={:.3}  b={:.3}", phase_str, a_coeff, b_coeff),
        FontId::new(8.5, FontFamily::Monospace), Color32::from_rgb(80, 200, 160));

    // Metrics — prefer real TN-quimb data over score-derived estimates
    let r_lnz   = layer.fields.iter().find(|(k,_,_)| k == "ln_Z")
                      .map(|(_, v, _)| v.clone());
    let r_fsite = layer.fields.iter().find(|(k,_,_)| k == "F/site")
                      .map(|(_, v, _)| v.clone());
    let r_beta  = layer.fields.iter().find(|(k,_,_)| k == "β")
                      .map(|(_, v, _)| v.clone());
    let r_l     = layer.fields.iter().find(|(k,_,_)| k == "L")
                      .map(|(_, v, _)| v.clone());
    let src_tag = if layer.has_real_data { "quimb PEPS U(1)" } else { "SIM Ising" };
    let my = fp.max.y + 28.0;
    let items: &[(&str, String, Color32)] = &[
        ("ln Z",    r_lnz.unwrap_or_else(||   format!("{:.3}", -12.0 + s * 3.0)),  theme::GREEN_OK),
        ("F/site",  r_fsite.unwrap_or_else(|| format!("{:.4}", -0.40 - s * 0.2)),  theme::BLUE_INFO),
        ("β",       r_beta.unwrap_or_else(||  format!("{:.3}", beta_now)),          theme::AMBER),
        ("L",       r_l.unwrap_or_else(||     "16".into()),                         theme::TEXT_MID),
    ];
    for (i, (lbl, val, col)) in items.iter().enumerate() {
        let y = my + i as f32 * 18.0;
        painter.text(Pos2::new(fe_r.min.x + 12.0, y), egui::Align2::LEFT_TOP,
            *lbl, FontId::new(9.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,140));
        painter.text(Pos2::new(fe_r.max.x - 8.0, y), egui::Align2::RIGHT_TOP,
            val, FontId::new(10.0, FontFamily::Monospace), *col);
    }
    painter.text(Pos2::new(fe_r.center().x, fe_r.max.y - 6.0), egui::Align2::CENTER_BOTTOM,
        format!("[{}]", src_tag),
        FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(80,200,160,120));
}

/// Cosmological — CMB quadrupole map + multipole power deficit
fn render_cosmological_instrument(painter: &egui::Painter, rect: Rect, layer: &LayerMetrics, t: f64) {
    painter.rect_filled(rect, Rounding::same(8.0), theme::BG_DARK);
    painter.rect_stroke(rect, Rounding::same(8.0), Stroke::new(1.0, theme::BORDER));

    let split_x = rect.min.x + rect.width() * 0.52;
    let sky_r   = Rect::from_min_max(rect.min, Pos2::new(split_x - 4.0, rect.max.y));
    let pwr_r   = Rect::from_min_max(Pos2::new(split_x + 4.0, rect.min.y), rect.max);
    let s       = layer.score as f32;

    // ── Mollweide-ish CMB sky map (simplified) ────────────────────────
    painter.text(sky_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "CMB TEMPERATURE MAP — QUADRUPOLE",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(150, 150, 220));

    let cx = sky_r.center().x;
    let cy = sky_r.center().y + 8.0;
    let rx = sky_r.width() * 0.44;
    let ry = sky_r.height() * 0.36;

    // Sky pixel grid (clipped to Mollweide ellipse)
    let gx = 52usize;
    let gy = 28usize;
    for gi in 0..gx {
        for gj in 0..gy {
            let u = (gi as f32 / gx as f32 - 0.5) * 2.0;
            let v = (gj as f32 / gy as f32 - 0.5) * 2.0;
            if u*u + v*v > 1.0 { continue; }
            // Quadrupole: Y_2^0 ∝ 3cos²θ - 1
            let cos_theta = v;
            let y20 = 0.5 * (3.0 * cos_theta * cos_theta - 1.0);
            // Octopole anomaly
            let lon = u * std::f32::consts::PI;
            let y31 = (t as f32 * 0.05 + lon).cos() * v * 0.3;
            let temp_anom = (y20 * 0.7 + y31 * 0.3 + 0.02 * (u * 6.0 + t as f32 * 0.1).sin()) * s;
            let t_norm = (temp_anom + 1.0) * 0.5;
            let col = if t_norm < 0.4 {
                Color32::from_rgb(theme::lerp(0, 60, t_norm / 0.4), 0, theme::lerp(120, 200, t_norm / 0.4))
            } else if t_norm < 0.6 {
                let f = (t_norm - 0.4) / 0.2;
                Color32::from_rgb(theme::lerp(60, 255, f), theme::lerp(0, 255, f), theme::lerp(200, 255, f))
            } else {
                let f = (t_norm - 0.6) / 0.4;
                Color32::from_rgb(255, theme::lerp(255, 32, f), theme::lerp(255, 72, f))
            };
            let px = cx + u * rx;
            let py = cy + v * ry;
            let cell_w = rx * 2.0 / gx as f32 + 1.0;
            let cell_h = ry * 2.0 / gy as f32 + 1.0;
            painter.rect_filled(
                Rect::from_center_size(Pos2::new(px, py), Vec2::new(cell_w, cell_h)),
                Rounding::same(0.0), col);
        }
    }
    // Ellipse border
    let n_e = 64usize;
    let e_pts: Vec<Pos2> = (0..=n_e).map(|i| {
        let a = i as f32 / n_e as f32 * std::f32::consts::TAU;
        Pos2::new(cx + rx * a.cos(), cy + ry * a.sin())
    }).collect();
    for i in 0..e_pts.len()-1 {
        painter.line_segment([e_pts[i], e_pts[i+1]],
            Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,80)));
    }

    // ── CMB power deficit (right panel) ──────────────────────────────
    painter.text(pwr_r.min + Vec2::new(8.0, 6.0), egui::Align2::LEFT_TOP,
        "LOW-l POWER DEFICIT",
        FontId::new(9.5, FontFamily::Proportional), Color32::from_rgb(150, 150, 220));

    let pp = Rect::from_min_max(
        Pos2::new(pwr_r.min.x + 12.0, pwr_r.min.y + 26.0),
        Pos2::new(pwr_r.max.x - 8.0,  pwr_r.max.y - 24.0));
    painter.line_segment([Pos2::new(pp.min.x, pp.max.y), Pos2::new(pp.max.x, pp.max.y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,60)));
    painter.line_segment([Pos2::new(pp.min.x, pp.min.y), Pos2::new(pp.min.x, pp.max.y)],
        Stroke::new(1.0, Color32::from_rgba_unmultiplied(255,255,255,60)));

    // Lambda-CDM expected (flat reference)
    painter.line_segment([Pos2::new(pp.min.x, pp.center().y), Pos2::new(pp.max.x, pp.center().y)],
        Stroke::new(0.8, Color32::from_rgba_unmultiplied(255,255,255,40)));
    painter.text(Pos2::new(pp.max.x + 2.0, pp.center().y), egui::Align2::LEFT_CENTER,
        "ΛCDM", FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,100));

    // Observed power (with deficit at low-l)
    let multipoles = [2u32, 3, 4, 5, 6, 8, 10, 15, 20, 30, 50, 80, 120, 191];
    let deficit = 1.0 - layer.score as f32 * 0.15;
    let mut prev: Option<Pos2> = None;
    for (i, &l) in multipoles.iter().enumerate() {
        let x = pp.min.x + (i as f32 / (multipoles.len()-1) as f32) * pp.width();
        // Low-l deficit
        let low_l_supp = if l <= 6 { 0.5 + 0.3 * (l as f32 - 2.0) / 4.0 } else { 1.0 };
        let cl = 0.5 + (1.0 - low_l_supp * deficit) * 0.4 + 0.1 * (i as f32 * 1.3).sin();
        let y = pp.max.y - cl.clamp(0.0, 1.0) * pp.height();
        let pt = Pos2::new(x, y);
        painter.circle_filled(pt, 3.0, theme::ACCENT);
        if let Some(pr) = prev { painter.line_segment([pr, pt], Stroke::new(1.2, theme::ACCENT)); }
        prev = Some(pt);
        if l <= 10 {
            painter.text(Pos2::new(x, pp.max.y + 3.0), egui::Align2::CENTER_TOP,
                format!("{l}"), FontId::new(7.5, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,120));
        }
    }

    // Real scipy/spherical-harmonic values when available
    let r_c2   = layer.fields.iter().find(|(k,_,_)| k == "C₂")
                     .map(|(_, v, _)| v.clone());
    let r_hemi = layer.fields.iter().find(|(k,_,_)| k == "Hemi_Δ")
                     .map(|(_, v, _)| v.clone());
    let r_rat  = layer.fields.iter().find(|(k,_,_)| k == "C₂/exp")
                     .map(|(_, v, _)| v.clone());
    let r_lmax = layer.fields.iter().find(|(k,_,_)| k == "lmax")
                     .map(|(_, v, _)| v.clone());
    let src_tag = if layer.has_real_data { "scipy/sph_harm" } else { "SIM Y₂⁰" };
    let items: &[(&str, String, Color32)] = &[
        ("C₂",     r_c2.unwrap_or_else(||   format!("{:.6}", 0.0146 + s as f64 * 0.002)), theme::BLUE_INFO),
        ("Hemi_Δ", r_hemi.unwrap_or_else(|| format!("{:.4}", 1.03  + s as f64 * 0.04)),   theme::AMBER),
        ("C₂/exp", r_rat.unwrap_or_else(||  format!("{:.4}", 0.95  + s as f64 * 0.06)),   theme::GREEN_OK),
        ("l_max",  r_lmax.unwrap_or_else(|| "191".into()),                                  theme::TEXT_MID),
    ];
    let my = pp.max.y + 14.0;
    for (i, (lbl, val, col)) in items.iter().enumerate() {
        let y = my + i as f32 * 18.0;
        painter.text(Pos2::new(pwr_r.min.x + 12.0, y), egui::Align2::LEFT_TOP,
            *lbl, FontId::new(9.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(255,255,255,140));
        painter.text(Pos2::new(pwr_r.max.x - 8.0, y), egui::Align2::RIGHT_TOP,
            val, FontId::new(10.0, FontFamily::Monospace), *col);
    }
    painter.text(Pos2::new(pwr_r.center().x, pwr_r.max.y - 6.0), egui::Align2::CENTER_BOTTOM,
        format!("[{}]", src_tag),
        FontId::new(8.0, FontFamily::Monospace), Color32::from_rgba_unmultiplied(150,150,220,120));
}
