//! egui application — FIELD_COHERENCE_MONITOR dashboard.
//!
//! Layout (1300×840):
//!   ┌─────────────────────────────────────────────────────────────┐
//!   │  HEADER: title + alert badge + data status                  │
//!   ├──────────────────────┬──────────────────────────────────────┤
//!   │  LEFT (38%)          │  RIGHT (62%)                         │
//!   │  • Coherence gauge   │  • Kp index  (3-day, 3h intervals)  │
//!   │  • Component bars    │  • F10.7 solar flux (6 years monthly)│
//!   │  • Current readings  │  • CR proxy (inverse F10.7)          │
//!   │  • Pole drift chart  │                                      │
//!   ├──────────────────────┴──────────────────────────────────────┤
//!   │  BOTTOM: % toward Laschamp + historical context bar         │
//!   └─────────────────────────────────────────────────────────────┘

use std::sync::{Arc, RwLock};
use egui::{Color32, Margin, RichText, Stroke};
use egui_plot::{Line, LineStyle, Plot, PlotPoints};

use crate::fetcher::DataState;
use crate::coherence::{self, CoherenceResult};
use crate::probe_state::ProbeState;

// ── Color palette ─────────────────────────────────────────────────────────────

const BG_DARK:   Color32 = Color32::from_rgb( 8, 10, 18);
const BG_PANEL:  Color32 = Color32::from_rgb(15, 20, 35);
const BG_HEADER: Color32 = Color32::from_rgb(10, 13, 25);
const BORDER:    Color32 = Color32::from_rgb(45, 55, 90);
const DIM:       Color32 = Color32::from_rgb(120, 135, 170);

// Signal colors
const C_SOLAR:   Color32 = Color32::from_rgb(255, 190,  50);  // amber
const C_KP:      Color32 = Color32::from_rgb( 80, 195, 255);  // cyan
const C_CR:      Color32 = Color32::from_rgb(180, 100, 255);  // violet
const C_POLE:    Color32 = Color32::from_rgb( 80, 255, 180);  // teal
const C_ALARM:   Color32 = Color32::from_rgb(220,  60,  60);  // red (Laschamp)
const C_THRESH:  Color32 = Color32::from_rgb(200, 120,  40);  // storm threshold

// ── App struct ────────────────────────────────────────────────────────────────

pub struct FieldCoherenceApp {
    state: Arc<RwLock<DataState>>,
    initialized: bool,
    probe: Option<ProbeState>,
}

impl FieldCoherenceApp {
    pub fn new(_cc: &eframe::CreationContext<'_>, state: Arc<RwLock<DataState>>) -> Self {
        // Resolve project root: binary lives in target/…/field_coherence_monitor,
        // so walk up until we find data/processed or give up after 6 levels.
        let probe = (0..6)
            .scan(
                std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|p| p.to_path_buf()))
                    .unwrap_or_default(),
                |dir, _| {
                    let candidate = dir.join("data/processed/probe_state.json");
                    let result = if candidate.exists() { Some(candidate) } else { None };
                    *dir = dir.parent().map(|p| p.to_path_buf()).unwrap_or_default();
                    Some(result)
                },
            )
            .flatten()
            .next()
            .and_then(|p| ProbeState::load(&p));
        Self { state, initialized: false, probe }
    }
}

// ── eframe::App ───────────────────────────────────────────────────────────────

impl eframe::App for FieldCoherenceApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Apply dark theme once on first frame
        if !self.initialized {
            ctx.set_visuals(egui::Visuals::dark());
            self.initialized = true;
        }

        // Snapshot shared state (releases lock immediately)
        let data   = self.state.read().unwrap().clone();
        let result = coherence::compute(&data);

        // Global background
        let mut vis = ctx.style().visuals.clone();
        vis.panel_fill       = BG_DARK;
        vis.window_fill      = BG_DARK;
        vis.extreme_bg_color = BG_PANEL;
        ctx.set_visuals(vis);

        // ── Header ───────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("header")
            .exact_height(50.0)
            .frame(egui::Frame::default()
                .fill(BG_HEADER)
                .inner_margin(Margin::symmetric(12.0, 8.0)))
            .show(ctx, |ui| {
                render_header(ui, &data, &result);
            });

        // ── Laschamp context bar ─────────────────────────────────────────────
        egui::TopBottomPanel::bottom("laschamp_bar")
            .exact_height(54.0)
            .frame(egui::Frame::default()
                .fill(BG_HEADER)
                .inner_margin(Margin::symmetric(12.0, 8.0)))
            .show(ctx, |ui| {
                render_laschamp_bar(ui, &result);
            });

        // ── Central ──────────────────────────────────────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::default()
                .fill(BG_DARK)
                .inner_margin(Margin::same(8.0)))
            .show(ctx, |ui| {
                let probe = self.probe.clone();
                ui.columns(2, |cols| {
                    // Left column: ~38% width (narrower)
                    render_left_panel(&mut cols[0], &data, &result, probe.as_ref());
                    // Right column: plots
                    render_right_panel(&mut cols[1], &data);
                });
            });

        // Poll for data updates every 30 s
        ctx.request_repaint_after(std::time::Duration::from_secs(30));
    }
}

// ── Header ────────────────────────────────────────────────────────────────────

fn render_header(ui: &mut egui::Ui, data: &DataState, result: &CoherenceResult) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("⊕  FIELD COHERENCE MONITOR")
            .size(17.0).strong().color(Color32::WHITE));
        ui.add_space(10.0);
        ui.label(RichText::new("cycle_project / module 4")
            .size(11.0).color(DIM));

        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            // Alert badge
            ui.label(RichText::new(result.alert.label())
                .size(14.0).strong().color(result.alert.color()));
            ui.add_space(20.0);

            // Fetch status
            let ts = data.last_fetch.as_deref().unwrap_or("—");
            let ts_short = if ts.len() >= 16 { &ts[..16] } else { ts };
            let status_color = if data.fetch_ok {
                Color32::from_rgb(0, 180, 80)
            } else {
                Color32::from_rgb(180, 160, 0)
            };
            ui.label(RichText::new(
                format!("{}  ·  {}", ts_short, data.fetch_msg)
            ).size(11.0).color(status_color));
        });
    });
}

// ── Laschamp context bar ──────────────────────────────────────────────────────

fn render_laschamp_bar(ui: &mut egui::Ui, result: &CoherenceResult) {
    ui.label(RichText::new(
        "HISTORICAL CONTEXT  ·  vs LASCHAMP GEOMAGNETIC EXCURSION (41,000 BP)"
    ).size(10.0).color(DIM));
    ui.add_space(3.0);
    ui.horizontal(|ui| {
        let pct = (result.laschamp_pct / 100.0).clamp(0.0, 1.0);
        ui.label(RichText::new(
            format!("{:.1}% toward Laschamp conditions", result.laschamp_pct)
        ).size(13.0).strong().color(lerp_color(
            Color32::from_rgb(0, 180, 80),
            C_ALARM,
            pct,
        )));
        ui.add_space(10.0);
        let bar_w = (ui.available_width() - 320.0).max(100.0);
        ui.add(egui::ProgressBar::new(pct)
            .desired_width(bar_w)
            .fill(lerp_color(Color32::from_rgb(0, 180, 80), C_ALARM, pct)));
        ui.add_space(10.0);
        ui.label(RichText::new(
            "Laschamp: VADM≈15%, aurora at equator, Be-10 +300%, cycle_detect Laschamp score=0.013"
        ).size(9.0).color(DIM));
    });
}

// ── Left panel ────────────────────────────────────────────────────────────────

fn render_left_panel(ui: &mut egui::Ui, data: &DataState, result: &CoherenceResult, probe: Option<&ProbeState>) {
    ui.vertical(|ui| {
        // Coherence score gauge
        card(ui, |ui| {
            ui.label(RichText::new("COHERENCE INDEX").size(11.0).color(DIM));
            ui.add_space(6.0);

            let score_color = lerp_color(C_ALARM, Color32::from_rgb(0, 200, 80), result.score);
            ui.label(RichText::new(format!("{:.4}", result.score))
                .size(44.0).strong().color(score_color).monospace());
            ui.add_space(2.0);
            ui.add(egui::ProgressBar::new(result.score)
                .desired_width(ui.available_width())
                .fill(score_color));

            ui.add_space(10.0);
            ui.label(RichText::new("Signal contributions  (0=stable · 1=destabilized)")
                .size(10.0).color(DIM));
            ui.add_space(4.0);
            contrib_bar(ui, "Kp index (geomagnetic)",    result.kp_contrib,    C_KP);
            contrib_bar(ui, "CR proxy (inv F10.7)",       result.cr_contrib,    C_CR);
            contrib_bar(ui, "Pole drift rate",            result.drift_contrib, C_POLE);
        });

        ui.add_space(6.0);

        // Current readings
        card(ui, |ui| {
            ui.label(RichText::new("CURRENT READINGS").size(11.0).color(DIM));
            ui.add_space(6.0);

            reading(ui, "Kp (24h mean)",
                format!("{:.2}  — {}",
                    result.current_kp,
                    coherence::kp_status_label(result.current_kp)),
                C_KP);

            reading(ui, "Solar flux F10.7",
                format!("{:.1} sfu  — {}", result.current_f107, result.solar_cycle_phase),
                C_SOLAR);

            reading(ui, "Pole drift rate",
                format!("{:.0} km/yr", result.pole_drift_km_yr),
                C_POLE);

            if let Some(p) = data.pole.last() {
                ui.add_space(4.0);
                ui.separator();
                ui.add_space(4.0);
                reading(ui, "Mag North (WMM 2025)",
                    format!("{:.1}°N  {:.1}°{}",
                        p.lat, p.lon.abs(),
                        if p.lon < 0.0 { "W" } else { "E" }),
                    C_POLE);
            }

            ui.add_space(8.0);
            ui.label(RichText::new(
                "⚠ CR proxy = inverse F10.7 only.\n\
                 Direct neutron flux → nmdb.eu · cosmicrays.oulu.fi"
            ).size(9.0).color(DIM));
        });

        ui.add_space(6.0);

        // Pole latitude drift chart
        card(ui, |ui| {
            ui.label(RichText::new("MAG NORTH POLE  ·  LATITUDE 2000–2025")
                .size(11.0).color(DIM));
            ui.add_space(4.0);

            let pts: PlotPoints = data.pole.iter()
                .map(|p| [p.year, p.lat])
                .collect();

            Plot::new("pole_lat")
                .height(95.0)
                .include_y(80.0)
                .include_y(90.0)
                .show_axes([true, true])
                .show(ui, |plot_ui| {
                    plot_ui.line(Line::new(pts).color(C_POLE).name("Lat °N (WMM)"));
                });

            ui.label(RichText::new(
                "Drift crossed antimeridian (Canadian→Siberian Arctic) ~2019"
            ).size(9.0).color(DIM));
        });

        ui.add_space(6.0);

        // Forward probe panel
        card(ui, |ui| {
            ui.label(RichText::new("FORWARD PROBE  ·  module 5").size(11.0).color(DIM));
            ui.add_space(6.0);
            match probe {
                None => {
                    ui.label(RichText::new("No probe data")
                        .size(12.0).color(DIM).italics());
                    ui.label(RichText::new("Run src/forward_probe/run_forward_probe.py")
                        .size(9.0).color(DIM));
                }
                Some(p) => {
                    let prob_color = if p.pre_excursion_prob >= 0.7 {
                        C_ALARM
                    } else if p.pre_excursion_prob >= 0.4 {
                        C_THRESH
                    } else {
                        Color32::from_rgb(0, 200, 80)
                    };
                    reading(ui, "Pre-excursion prob",
                        format!("{:.3}", p.pre_excursion_prob), prob_color);
                    reading(ui, "VADM @ +1,000 yr",
                        format!("{:.3}", p.lstm_vadm_1kyr), C_SOLAR);
                    reading(ui, "VADM @ +5,000 yr",
                        format!("{:.3}", p.lstm_vadm_5kyr), C_SOLAR);
                    if p.instrumental_threshold_yr > 0 {
                        reading(ui, "Threshold crossing",
                            format!("~{} yr", p.instrumental_threshold_yr), C_CR);
                    }
                    if !p.generated_at.is_empty() {
                        ui.add_space(4.0);
                        let ts = if p.generated_at.len() >= 16 { &p.generated_at[..16] } else { &p.generated_at };
                        ui.label(RichText::new(format!("Updated: {}", ts))
                            .size(9.0).color(DIM));
                    }
                }
            }
        });
    });
}

// ── Right panel ───────────────────────────────────────────────────────────────

fn render_right_panel(ui: &mut egui::Ui, data: &DataState) {
    ui.vertical(|ui| {
        // Kp plot (last 3 days = 24 × 3h readings)
        card(ui, |ui| {
            ui.label(RichText::new(
                "PLANETARY Kp INDEX  ·  3-hour intervals  ·  last 3 days"
            ).size(11.0).color(DIM));
            ui.add_space(4.0);

            let kp_pts: Vec<[f64; 2]> = data.kp.iter()
                .rev().take(24).rev()
                .enumerate()
                .map(|(i, r)| [i as f64, r.kp])
                .collect();
            let n_kp = kp_pts.len().max(1) as f64;

            Plot::new("kp_plot")
                .height(140.0)
                .include_y(0.0)
                .include_y(9.0)
                .show_axes([false, true])
                .show(ui, |plot_ui| {
                    if !kp_pts.is_empty() {
                        plot_ui.line(Line::new(PlotPoints::new(kp_pts))
                            .color(C_KP)
                            .name("Kp"));
                    }
                    // G1 storm threshold
                    let thresh = PlotPoints::new(vec![[0.0, 5.0], [n_kp, 5.0]]);
                    plot_ui.line(Line::new(thresh)
                        .color(C_THRESH)
                        .style(LineStyle::Dashed { length: 6.0 })
                        .name("G1 threshold (Kp=5)"));
                    // G3 strong storm
                    let g3 = PlotPoints::new(vec![[0.0, 7.0], [n_kp, 7.0]]);
                    plot_ui.line(Line::new(g3)
                        .color(C_ALARM)
                        .style(LineStyle::Dashed { length: 4.0 })
                        .name("G3 threshold (Kp=7)"));
                });

            ui.label(RichText::new(
                "Orange dashed = G1 (Kp≥5) · Red dashed = G3 (Kp≥7)"
            ).size(9.0).color(DIM));
        });

        ui.add_space(6.0);

        // F10.7 solar flux (last 72 months)
        card(ui, |ui| {
            ui.label(RichText::new(
                "SOLAR FLUX F10.7  ·  monthly  ·  last 6 years"
            ).size(11.0).color(DIM));
            ui.add_space(4.0);

            let n = data.solar.len() as f64;
            let f107_pts: PlotPoints = data.solar.iter()
                .enumerate()
                .map(|(i, r)| [i as f64, r.f107])
                .collect();
            let ssn_pts: PlotPoints = data.solar.iter()
                .enumerate()
                .map(|(i, r)| [i as f64, r.ssn])
                .collect();

            Plot::new("f107_plot")
                .height(150.0)
                .include_y(60.0)
                .include_y(280.0)
                .show_axes([false, true])
                .show(ui, |plot_ui| {
                    if n > 0.0 {
                        plot_ui.line(Line::new(f107_pts)
                            .color(C_SOLAR)
                            .name("F10.7 (sfu)"));
                        plot_ui.line(Line::new(ssn_pts)
                            .color(Color32::from_rgb(255, 255, 100))
                            .name("SSN (smoothed)"));
                        // Solar min baseline
                        let base = PlotPoints::new(vec![[0.0, 70.0], [n, 70.0]]);
                        plot_ui.line(Line::new(base)
                            .color(DIM)
                            .style(LineStyle::Dashed { length: 4.0 })
                            .name("Solar min (70 sfu)"));
                    }
                });

            ui.label(RichText::new(
                "Amber = F10.7  ·  Yellow = SSN  ·  Solar Cycle 25 peak ~2024-2025"
            ).size(9.0).color(DIM));
        });

        ui.add_space(6.0);

        // CR proxy (inverse F10.7, normalized 0–1)
        card(ui, |ui| {
            ui.label(RichText::new(
                "COSMIC RAY FLUX PROXY  ·  inverse-normalized F10.7  ·  (0=shielded · 1=max exposure)"
            ).size(11.0).color(DIM));
            ui.add_space(4.0);

            let n = data.solar.len() as f64;
            let cr_pts: PlotPoints = data.solar.iter()
                .enumerate()
                .map(|(i, r)| {
                    let norm = ((r.f107 - 70.0) / (310.0 - 70.0)).clamp(0.0, 1.0);
                    [i as f64, 1.0 - norm]
                })
                .collect();

            Plot::new("cr_plot")
                .height(120.0)
                .include_y(0.0)
                .include_y(1.0)
                .show_axes([false, true])
                .show(ui, |plot_ui| {
                    if n > 0.0 {
                        plot_ui.line(Line::new(cr_pts)
                            .color(C_CR)
                            .name("CR proxy"));
                        // Laschamp reference: CR was ~4× modern swing → off scale here
                        // Show "maximum modern exposure" as top reference
                        let top = PlotPoints::new(vec![[0.0, 1.0], [n, 1.0]]);
                        plot_ui.line(Line::new(top)
                            .color(C_ALARM)
                            .style(LineStyle::Dashed { length: 6.0 })
                            .name("Modern max (Laschamp was ~4× this level)"));
                    }
                });

            ui.label(RichText::new(
                "Real neutron flux: nmdb.eu  ·  Oulu station cosmicrays.oulu.fi"
            ).size(9.0).color(DIM));
        });
    });
}

// ── Widget helpers ────────────────────────────────────────────────────────────

fn card(ui: &mut egui::Ui, content: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::default()
        .fill(BG_PANEL)
        .stroke(Stroke::new(1.0, BORDER))
        .inner_margin(Margin::same(10.0))
        .show(ui, content);
}

fn contrib_bar(ui: &mut egui::Ui, label: &str, value: f32, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{:<28}", label))
            .size(11.0).color(DIM).monospace());
        ui.add(egui::ProgressBar::new(value)
            .desired_width(110.0)
            .fill(color));
        ui.label(RichText::new(format!("{:.3}", value))
            .size(11.0).color(color).monospace());
    });
    ui.add_space(2.0);
}

fn reading(ui: &mut egui::Ui, label: &str, value: String, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(format!("{:<22}", label))
            .size(12.0).color(DIM));
        ui.label(RichText::new(value)
            .size(12.0).strong().color(color));
    });
    ui.add_space(2.0);
}

fn lerp_color(a: Color32, b: Color32, t: f32) -> Color32 {
    let t = t.clamp(0.0, 1.0);
    Color32::from_rgb(
        lerp_u8(a.r(), b.r(), t),
        lerp_u8(a.g(), b.g(), t),
        lerp_u8(a.b(), b.b(), t),
    )
}

fn lerp_u8(a: u8, b: u8, t: f32) -> u8 {
    (a as f32 + (b as f32 - a as f32) * t) as u8
}
