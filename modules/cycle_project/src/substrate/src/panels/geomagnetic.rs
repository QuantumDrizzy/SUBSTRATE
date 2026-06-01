//! Panel: Detección de anomalías geomagnéticas (GNN + PCA graph diffusion)

use egui::{Color32, Margin, RichText, Stroke};
use egui_plot::{Line, Plot, PlotPoints, HLine};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::state::{SharedState, JobStatus, AnalysisResult};
use crate::bridge;
use super::shared::*;

// ── Parámetros configurables ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct GeoParams {
    pub window:      usize,
    pub stride:      usize,
    pub corr_thresh: f64,
    pub latent:      usize,
    pub anom_pct:    f64,
}

impl Default for GeoParams {
    fn default() -> Self {
        Self { window: 50, stride: 5, corr_thresh: 0.35, latent: 2, anom_pct: 95.0 }
    }
}

// ── Panel ─────────────────────────────────────────────────────────────────────

pub struct GeomagneticPanel {
    state:   SharedState,
    params:  GeoParams,
    status:  Arc<Mutex<JobStatus>>,
    result:  Arc<Mutex<Option<AnalysisResult>>>,
}

impl GeomagneticPanel {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            params:  GeoParams::default(),
            status:  Arc::new(Mutex::new(JobStatus::Idle)),
            result:  Arc::new(Mutex::new(None)),
        }
    }

    fn run(&self, data_dir: String) {
        let p       = self.params.clone();
        let status  = Arc::clone(&self.status);
        let result  = Arc::clone(&self.result);

        *status.lock().unwrap() = JobStatus::Running;

        thread::spawn(move || {
            match bridge::run_geomagnetic(&data_dir, p.window, p.stride, p.corr_thresh, p.latent) {
                Ok(r)  => {
                    *result.lock().unwrap() = Some(r);
                    *status.lock().unwrap() = JobStatus::Done;
                }
                Err(e) => {
                    *status.lock().unwrap() = JobStatus::Error(e.to_string());
                }
            }
        });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        let data_dir = self.state.read().unwrap()
            .data_dir.to_str().unwrap_or("data/processed").to_string();

        panel_header(ui, "⊕  GEOMAGNETIC ANOMALY SCANNER",
            "Graph-diffused PCA reconstruction error sobre 5 proxies paleoclimáticos");

        ui.add_space(10.0);

        // ── Controles ──────────────────────────────────────────────────────
        card(ui, |ui| {
            ui.label(label_dim("PARÁMETROS"));
            ui.add_space(6.0);
            ui.columns(4, |cols| {
                cols[0].label(label_dim("Window (steps)"));
                cols[0].add(egui::Slider::new(&mut self.params.window, 10..=200).integer());

                cols[1].label(label_dim("Stride"));
                cols[1].add(egui::Slider::new(&mut self.params.stride, 1..=20).integer());

                cols[2].label(label_dim("Corr threshold"));
                cols[2].add(egui::Slider::new(&mut self.params.corr_thresh, 0.1..=0.9));

                cols[3].label(label_dim("Latent dims"));
                cols[3].add(egui::Slider::new(&mut self.params.latent, 1..=5).integer());
            });

            ui.add_space(8.0);

            let status = self.status.lock().unwrap().clone();
            let running = matches!(status, JobStatus::Running);

            if ui.add_enabled(!running,
                egui::Button::new(RichText::new("▶  Run Anomaly Scan").size(13.0))
                    .min_size(egui::vec2(180.0, 30.0))
            ).clicked() {
                self.run(data_dir);
            }

            ui.add_space(4.0);
            render_job_status(ui, &status);
        });

        ui.add_space(8.0);

        // ── Resultado ──────────────────────────────────────────────────────
        let result = self.result.lock().unwrap().clone();
        if let Some(r) = result {
            // Métricas
            card(ui, |ui| {
                ui.label(label_dim("MÉTRICAS"));
                ui.add_space(4.0);
                ui.horizontal_wrapped(|ui| {
                    for (k, v) in &r.scalars {
                        metric_chip(ui, k, &format!("{v:.4}"));
                    }
                });
            });

            ui.add_space(8.0);

            // Plot anomaly scores
            if !r.values.is_empty() {
                card(ui, |ui| {
                    ui.label(label_dim("ANOMALY SCORES  ·  reconstruction error por ventana"));
                    ui.add_space(4.0);

                    let pts: PlotPoints = r.values.iter().enumerate()
                        .map(|(i, &v)| [i as f64, v])
                        .collect();

                    let threshold = r.scalars.iter()
                        .find(|(k, _)| k.contains("threshold"))
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0);

                    Plot::new("geo_scores")
                        .height(220.0)
                        .include_y(0.0)
                        .show_axes([false, true])
                        .show(ui, |plot_ui| {
                            plot_ui.line(Line::new(pts)
                                .color(Color32::from_rgb(80, 160, 255))
                                .name("Anomaly score"));
                            if threshold > 0.0 {
                                plot_ui.hline(HLine::new(threshold)
                                    .color(Color32::from_rgb(220, 60, 60))
                                    .name("p95 threshold"));
                            }
                        });

                    if !r.text.is_empty() {
                        ui.add_space(4.0);
                        ui.label(RichText::new(&r.text).size(11.0)
                            .color(Color32::from_rgb(110, 128, 165)));
                    }
                });
            }
        }
    }
}
