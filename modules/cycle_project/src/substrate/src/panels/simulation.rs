//! Panel: Lattice Boltzmann simulación litosférica

use egui::{Color32, RichText};
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::state::{SharedState, JobStatus, AnalysisResult};
use crate::bridge;
use super::shared::*;

pub struct SimulationPanel {
    state:   SharedState,
    n_steps: usize,
    use_gpu: bool,
    status:  Arc<Mutex<JobStatus>>,
    result:  Arc<Mutex<Option<AnalysisResult>>>,
}

impl SimulationPanel {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            n_steps: 1000,
            use_gpu: true,
            status:  Arc::new(Mutex::new(JobStatus::Idle)),
            result:  Arc::new(Mutex::new(None)),
        }
    }

    fn run(&self) {
        let n_steps = self.n_steps;
        let use_gpu = self.use_gpu;
        let status  = Arc::clone(&self.status);
        let result  = Arc::clone(&self.result);

        *status.lock().unwrap() = JobStatus::Running;

        thread::spawn(move || {
            match bridge::run_lbm(n_steps, use_gpu) {
                Ok(r)  => { *result.lock().unwrap() = Some(r); *status.lock().unwrap() = JobStatus::Done; }
                Err(e) => { *status.lock().unwrap() = JobStatus::Error(e.to_string()); }
            }
        });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        panel_header(ui, "◈  LITHOSPHERIC SIMULATION",
            "Lattice Boltzmann D2Q9 — flujo astenosférico bajo desplazamiento litosférico");

        ui.add_space(10.0);

        card(ui, |ui| {
            ui.label(label_dim("PARÁMETROS LBM"));
            ui.add_space(6.0);
            ui.columns(2, |c| {
                c[0].label(label_dim("Steps"));
                c[0].add(egui::Slider::new(&mut self.n_steps, 100..=10000).integer()
                    .logarithmic(true));
                c[1].label(label_dim("GPU (CuPy)"));
                c[1].checkbox(&mut self.use_gpu, "Activar");
            });

            ui.add_space(8.0);
            let running = matches!(*self.status.lock().unwrap(), JobStatus::Running);
            if ui.add_enabled(!running,
                egui::Button::new(RichText::new("▶  Run LBM").size(13.0))
                    .min_size(egui::vec2(140.0, 28.0))
            ).clicked() { self.run(); }

            ui.add_space(6.0);
            ui.label(RichText::new("⚠  Requiere CuPy para GPU. Sin él corre en NumPy (lento).")
                .size(10.0).color(Color32::from_rgb(200, 140, 40)));
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
            if !r.values.is_empty() {
                card(ui, |ui| {
                    ui.label(label_dim("DESPLAZAMIENTO LITOSFÉRICO (km)"));
                    ui.add_space(4.0);
                    let pts: PlotPoints = r.values.iter().enumerate()
                        .map(|(i, &v)| [i as f64, v]).collect();
                    Plot::new("lbm_plot")
                        .height(200.0)
                        .show_axes([false, true])
                        .show(ui, |p| {
                            p.line(Line::new(pts)
                                .color(Color32::from_rgb(80, 220, 180))
                                .name("displacement km"));
                        });
                });
            }
            if !r.text.is_empty() {
                ui.add_space(6.0);
                card(ui, |ui| {
                    ui.label(RichText::new(&r.text).size(11.0).color(DIM));
                });
            }
        }
    }
}
