//! Panel: Quantum Biology (radical pair + Lindblad + RF noise)

use egui::{Color32, RichText};
use egui_plot::{Line, Plot, PlotPoints};
use std::sync::{Arc, Mutex};
use std::thread;

use crate::state::{SharedState, JobStatus, AnalysisResult};
use crate::bridge;
use super::shared::*;

#[derive(Debug, Clone, PartialEq)]
pub enum QuantumMode { RadicalPair, Lindblad, RfNoise }

pub struct QuantumPanel {
    state:      SharedState,
    mode:       QuantumMode,
    b_field_ut: f64,
    rf_freq_mhz: f64,
    time_us:    f64,
    n_qubits:   usize,
    t_max:      f64,
    n_steps:    usize,
    status:     Arc<Mutex<JobStatus>>,
    result:     Arc<Mutex<Option<AnalysisResult>>>,
}

impl QuantumPanel {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            mode:        QuantumMode::RadicalPair,
            b_field_ut:  50.0,
            rf_freq_mhz: 1.4,
            time_us:     10.0,
            n_qubits:    2,
            t_max:       10.0,
            n_steps:     500,
            status:      Arc::new(Mutex::new(JobStatus::Idle)),
            result:      Arc::new(Mutex::new(None)),
        }
    }

    fn run(&self) {
        let mode       = self.mode.clone();
        let b          = self.b_field_ut;
        let rf         = self.rf_freq_mhz;
        let t          = self.time_us;
        let n_q        = self.n_qubits;
        let t_max      = self.t_max;
        let n_steps    = self.n_steps;
        let status     = Arc::clone(&self.status);
        let result     = Arc::clone(&self.result);

        *status.lock().unwrap() = JobStatus::Running;

        thread::spawn(move || {
            let res = match mode {
                QuantumMode::RadicalPair => bridge::run_radical_pair(b, t),
                QuantumMode::Lindblad    => bridge::run_lindblad(n_q, t_max, n_steps),
                QuantumMode::RfNoise     => bridge::run_rf_noise(b, rf),
            };
            match res {
                Ok(r)  => { *result.lock().unwrap() = Some(r); *status.lock().unwrap() = JobStatus::Done; }
                Err(e) => { *status.lock().unwrap() = JobStatus::Error(e.to_string()); }
            }
        });
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        panel_header(ui, "ψ  QUANTUM BIOLOGY",
            "Radical pair · Lindblad master equation · RF noise perturbation");

        ui.add_space(10.0);

        card(ui, |ui| {
            ui.horizontal(|ui| {
                for (mode, label) in [
                    (QuantumMode::RadicalPair, "Radical Pair"),
                    (QuantumMode::Lindblad,    "Lindblad"),
                    (QuantumMode::RfNoise,     "RF Noise"),
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

            ui.add_space(8.0);

            match self.mode {
                QuantumMode::RadicalPair | QuantumMode::RfNoise => {
                    ui.columns(2, |c| {
                        c[0].label(label_dim("B field (μT)"));
                        c[0].add(egui::Slider::new(&mut self.b_field_ut, 0.1..=200.0));
                        if self.mode == QuantumMode::RadicalPair {
                            c[1].label(label_dim("Time (μs)"));
                            c[1].add(egui::Slider::new(&mut self.time_us, 0.1..=100.0));
                        } else {
                            c[1].label(label_dim("RF freq (MHz)"));
                            c[1].add(egui::Slider::new(&mut self.rf_freq_mhz, 0.1..=100.0));
                        }
                    });
                }
                QuantumMode::Lindblad => {
                    ui.columns(3, |c| {
                        c[0].label(label_dim("Qubits"));
                        c[0].add(egui::Slider::new(&mut self.n_qubits, 1..=8).integer());
                        c[1].label(label_dim("t_max (ns)"));
                        c[1].add(egui::Slider::new(&mut self.t_max, 1.0..=100.0));
                        c[2].label(label_dim("Steps"));
                        c[2].add(egui::Slider::new(&mut self.n_steps, 100..=2000).integer());
                    });
                }
            }

            ui.add_space(8.0);
            let running = matches!(*self.status.lock().unwrap(), JobStatus::Running);
            if ui.add_enabled(!running,
                egui::Button::new(RichText::new("▶  Run").size(13.0))
                    .min_size(egui::vec2(120.0, 28.0))
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
                            metric_chip(ui, k, &format!("{v:.5}"));
                        }
                    });
                });
                ui.add_space(8.0);
            }
            if !r.values.is_empty() {
                card(ui, |ui| {
                    ui.label(label_dim("EVOLUCIÓN TEMPORAL"));
                    ui.add_space(4.0);
                    let pts: PlotPoints = r.values.iter().enumerate()
                        .map(|(i, &v)| [i as f64, v]).collect();
                    Plot::new("qnt_plot")
                        .height(200.0)
                        .include_y(0.0)
                        .include_y(1.0)
                        .show_axes([false, true])
                        .show(ui, |p| {
                            p.line(Line::new(pts)
                                .color(Color32::from_rgb(180, 100, 255))
                                .name("P_singlet(t)"));
                        });
                });
            }
        }
    }
}
