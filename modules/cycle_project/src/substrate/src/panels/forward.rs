//! Panel: Forward Probe (spectral + decay + fingerprint + LSTM)

use egui::{Color32, RichText};
use egui_plot::{Line, Plot, PlotPoints, VLine};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::state::{SharedState, JobStatus, AnalysisResult};
use crate::bridge;
use super::shared::*;

#[derive(Debug, Clone, PartialEq)]
pub enum ForwardMode { Spectral, Decay, Fingerprint, Forecast }

pub struct ForwardPanel {
    state:      SharedState,
    mode:       ForwardMode,
    n_ensemble: usize,
    status:     Arc<Mutex<JobStatus>>,
    result:     Arc<Mutex<Option<AnalysisResult>>>,
}

impl ForwardPanel {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            mode: ForwardMode::Spectral,
            n_ensemble: 50,
            status: Arc::new(Mutex::new(JobStatus::Idle)),
            result: Arc::new(Mutex::new(None)),
        }
    }

    fn run(&self) {
        let data_dir = self.state.read().unwrap()
            .data_dir.to_str().unwrap_or("data/processed").to_string();
        let mode    = self.mode.clone();
        let n_ens   = self.n_ensemble;
        let status  = Arc::clone(&self.status);
        let result  = Arc::clone(&self.result);

        *status.lock().unwrap() = JobStatus::Running;

        thread::spawn(move || {
            let res = match mode {
                ForwardMode::Spectral     => bridge::run_spectral(&data_dir),
                ForwardMode::Decay        => bridge::run_decay(&data_dir),
                ForwardMode::Fingerprint  => bridge::run_fingerprint(&data_dir),
                ForwardMode::Forecast     => bridge::run_forecast(&data_dir, n_ens),
            };
            match res {
                Ok(r)  => { *result.lock().unwrap() = Some(r); *status.lock().unwrap() = JobStatus::Done; }
                Err(e) => { *status.lock().unwrap() = JobStatus::Error(e.to_string()); }
            }
        });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        panel_header(ui, "→  FORWARD PROBE",
            "Espectral · Modelos de decaimiento VADM · Pre-excursión · LSTM ensemble");

        ui.add_space(10.0);

        card(ui, |ui| {
            ui.label(label_dim("MODO"));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                for (mode, label) in [
                    (ForwardMode::Spectral,    "FFT + CWT"),
                    (ForwardMode::Decay,       "Decay Models"),
                    (ForwardMode::Fingerprint, "Fingerprint"),
                    (ForwardMode::Forecast,    "LSTM Forecast"),
                ] {
                    let active = self.mode == mode;
                    if ui.add(egui::Button::new(
                        RichText::new(label).size(12.0)
                            .color(if active { ACCENT } else { DIM })
                    ).fill(if active { Color32::from_rgb(20,35,60) } else { Color32::TRANSPARENT }))
                    .clicked() {
                        self.mode = mode;
                        *self.result.lock().unwrap() = None;
                        *self.status.lock().unwrap() = JobStatus::Idle;
                    }
                    ui.add_space(4.0);
                }
            });

            if self.mode == ForwardMode::Forecast {
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    ui.label(label_dim("Ensemble size"));
                    ui.add(egui::Slider::new(&mut self.n_ensemble, 10..=200).integer());
                });
            }

            ui.add_space(8.0);
            let running = matches!(*self.status.lock().unwrap(), JobStatus::Running);
            if ui.add_enabled(!running,
                egui::Button::new(RichText::new("▶  Run").size(13.0))
                    .min_size(egui::vec2(120.0, 28.0))
            ).clicked() {
                self.run();
            }
            ui.add_space(4.0);
            render_job_status(ui, &self.status.lock().unwrap());
        });

        ui.add_space(8.0);

        if let Some(r) = self.result.lock().unwrap().clone() {
            // Métricas
            if !r.scalars.is_empty() {
                card(ui, |ui| {
                    ui.label(label_dim("RESULTADOS"));
                    ui.add_space(4.0);
                    ui.horizontal_wrapped(|ui| {
                        for (k, v) in &r.scalars {
                            metric_chip(ui, k, &format!("{v:.4}"));
                        }
                    });
                });
                ui.add_space(8.0);
            }

            // Plot
            if !r.values.is_empty() {
                card(ui, |ui| {
                    ui.label(label_dim("SERIE"));
                    ui.add_space(4.0);
                    let pts: PlotPoints = r.values.iter().enumerate()
                        .map(|(i, &v)| [i as f64, v]).collect();
                    Plot::new("fwd_plot")
                        .height(200.0)
                        .show_axes([false, true])
                        .show(ui, |p| {
                            p.line(Line::new(pts)
                                .color(Color32::from_rgb(255, 160, 60))
                                .name("signal"));
                        });
                });
            }

            if !r.text.is_empty() {
                ui.add_space(6.0);
                card(ui, |ui| {
                    ui.label(label_dim("INFORME"));
                    ui.add_space(4.0);
                    egui::ScrollArea::vertical().max_height(120.0).show(ui, |ui| {
                        ui.label(RichText::new(&r.text).size(11.0).color(DIM));
                    });
                });
            }
        }
    }
}
