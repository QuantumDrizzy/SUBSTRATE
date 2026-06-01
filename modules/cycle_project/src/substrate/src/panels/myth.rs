//! Panel: Myth RAG — correlación mito-geomagnética

use egui::{Color32, RichText};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::state::{SharedState, JobStatus, AnalysisResult};
use crate::bridge;
use super::shared::*;

pub struct MythPanel {
    state:      SharedState,
    window_kyr: f64,
    status:     Arc<Mutex<JobStatus>>,
    result:     Arc<Mutex<Option<AnalysisResult>>>,
}

impl MythPanel {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            window_kyr: 5.0,
            status:     Arc::new(Mutex::new(JobStatus::Idle)),
            result:     Arc::new(Mutex::new(None)),
        }
    }

    fn run(&self) {
        let data_dir = self.state.read().unwrap()
            .data_dir.to_str().unwrap_or("data/processed").to_string();
        let window   = self.window_kyr;
        let status   = Arc::clone(&self.status);
        let result   = Arc::clone(&self.result);

        *status.lock().unwrap() = JobStatus::Running;

        thread::spawn(move || {
            match bridge::run_myth_correlate(&data_dir, window) {
                Ok(r)  => { *result.lock().unwrap() = Some(r); *status.lock().unwrap() = JobStatus::Done; }
                Err(e) => { *status.lock().unwrap() = JobStatus::Error(e.to_string()); }
            }
        });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        panel_header(ui, "☽  MYTH RAG",
            "Correlación temporal entre anomalías geomagnéticas y tradiciones míticas de catástrofe");

        ui.add_space(10.0);

        card(ui, |ui| {
            ui.label(label_dim("VENTANA DE CORRELACIÓN"));
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.label(label_dim("Window (kyr)"));
                ui.add(egui::Slider::new(&mut self.window_kyr, 0.5..=20.0));
            });
            ui.add_space(8.0);
            let running = matches!(*self.status.lock().unwrap(), JobStatus::Running);
            if ui.add_enabled(!running,
                egui::Button::new(RichText::new("▶  Run Correlation").size(13.0))
                    .min_size(egui::vec2(160.0, 28.0))
            ).clicked() { self.run(); }
            ui.add_space(4.0);
            render_job_status(ui, &self.status.lock().unwrap());
        });

        ui.add_space(8.0);

        if let Some(r) = self.result.lock().unwrap().clone() {
            if !r.scalars.is_empty() {
                card(ui, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for (k, v) in &r.scalars {
                            metric_chip(ui, k, &format!("{v:.4}"));
                        }
                    });
                });
                ui.add_space(8.0);
            }
            if !r.text.is_empty() {
                card(ui, |ui| {
                    ui.label(label_dim("CORRELACIONES ENCONTRADAS"));
                    ui.add_space(6.0);
                    egui::ScrollArea::vertical().max_height(320.0).show(ui, |ui| {
                        ui.label(RichText::new(&r.text).size(12.0)
                            .color(Color32::from_rgb(200, 200, 220)));
                    });
                });
            }
        }
    }
}
