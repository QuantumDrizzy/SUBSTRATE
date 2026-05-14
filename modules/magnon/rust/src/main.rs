mod terra_protocol;
mod bloch_widget;
mod fidelity_plot;

use eframe::egui;
use terra_protocol::{TerraQciState, SHM_SIZE};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 900.0])
            .with_title("TERRA-QCI :: Quantum Coherence Grid")
            .with_decorations(true),
        ..Default::default()
    };
    eframe::run_native(
        "TERRA-QCI :: Quantum Coherence Grid",
        options,
        Box::new(|cc| {
            cc.egui_ctx.set_visuals(egui::Visuals::dark());
            Ok(Box::new(TerraApp::new()))
        }),
    )
}

struct TerraApp {
    shm_path: String,
    state: TerraQciState,
    last_seq: u64,
    history: Vec<(f64, f64)>,
    start_time: std::time::Instant,
}

impl TerraApp {
    fn new() -> Self {
        // Ruta relativa al CWD (TERRA-QCI/rust/ → TERRA-QCI/data/)
        let shm_path = "../data/terra_qci.shm".to_string();
        
        Self {
            shm_path,
            state: TerraQciState::default(),
            last_seq: 0,
            history: Vec::with_capacity(3600),
            start_time: std::time::Instant::now(),
        }
    }

    fn try_read_state(&mut self) {
        let mut buf = [0u8; SHM_SIZE];
        if let Ok(mut file) = OpenOptions::new().read(true).open(&self.shm_path) {
            let _ = file.seek(SeekFrom::Start(0));
            if file.read_exact(&mut buf).is_ok() {
                let new_state = TerraQciState::from_bytes(&buf);
                if new_state.seq > self.last_seq {
                    self.state = new_state;
                    self.last_seq = new_state.seq;
                }
            }
        }
    }
}

impl eframe::App for TerraApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Leer estado del motor Python
        self.try_read_state();
        
        let t = self.start_time.elapsed().as_secs_f64();
        
        // Samplear historial a ~10 Hz
        if self.history.is_empty() || (t - self.history.last().unwrap().0) > 0.1 {
            self.history.push((t, self.state.fidelity));
            if self.history.len() > 36000 { self.history.remove(0); }
        }

        let state = self.state;

        egui::CentralPanel::default()
            .frame(egui::Frame::default().fill(egui::Color32::from_rgb(8, 12, 18)))
            .show(ctx, |ui| {
                // Header
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.heading(
                        egui::RichText::new("⚛ TERRA-QCI :: QUANTUM COHERENCE GRID")
                            .color(egui::Color32::from_rgb(0, 255, 136))
                            .size(22.0)
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let status_color = if state.seq > 0 {
                            egui::Color32::from_rgb(0, 255, 100)
                        } else {
                            egui::Color32::from_rgb(255, 80, 80)
                        };
                        let status_text = if state.seq > 0 { "● ONLINE" } else { "○ WAITING" };
                        ui.label(egui::RichText::new(status_text).color(status_color).monospace().size(14.0));
                    });
                });
                
                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                ui.horizontal(|ui| {
                    // Panel Izquierdo: Esfera de Bloch
                    ui.vertical(|ui| {
                        ui.set_width(700.0);
                        ui.label(
                            egui::RichText::new("BLOCH SPHERE — Radical Pair State")
                                .color(egui::Color32::from_rgb(150, 170, 200))
                                .size(14.0)
                        );
                        ui.add_space(4.0);
                        bloch_widget::BlochSphere::new(
                            state.bloch_x as f32,
                            state.bloch_y as f32,
                            state.bloch_z as f32,
                            state.fidelity as f32,
                        ).ui(ui);
                    });

                    ui.separator();

                    // Panel Derecho: Métricas + Gráfico
                    ui.vertical(|ui| {
                        ui.set_width(660.0);
                        ui.add_space(8.0);
                        
                        // Métricas numéricas
                        ui.label(
                            egui::RichText::new("DECOHERENCE MONITOR")
                                .color(egui::Color32::from_rgb(0, 255, 136))
                                .size(16.0)
                        );
                        ui.add_space(12.0);

                        let fid_color = if state.fidelity > 0.9 {
                            egui::Color32::from_rgb(0, 255, 100)
                        } else if state.fidelity > 0.7 {
                            egui::Color32::YELLOW
                        } else {
                            egui::Color32::from_rgb(255, 60, 60)
                        };

                        let status_label = if state.fidelity > 0.9 {
                            "COHERENT"
                        } else if state.fidelity > 0.7 {
                            "DECOHERING"
                        } else {
                            "⚠ COLLAPSED"
                        };

                        ui.label(egui::RichText::new(format!("  STATUS:       {}", status_label))
                            .color(fid_color).monospace().size(16.0));
                        ui.add_space(8.0);

                        ui.label(egui::RichText::new(format!("  Fidelity:     {:.6}", state.fidelity))
                            .color(fid_color).monospace());
                        ui.label(egui::RichText::new(format!("  Bloch X:      {:>+.6}", state.bloch_x))
                            .color(egui::Color32::from_rgb(200, 220, 255)).monospace());
                        ui.label(egui::RichText::new(format!("  Bloch Y:      {:>+.6}", state.bloch_y))
                            .color(egui::Color32::from_rgb(200, 220, 255)).monospace());
                        ui.label(egui::RichText::new(format!("  Bloch Z:      {:>+.6}", state.bloch_z))
                            .color(egui::Color32::from_rgb(200, 220, 255)).monospace());
                        ui.label(egui::RichText::new(format!("  γ Lindblad:   {:.6e} Hz", state.lindblad_gamma))
                            .color(egui::Color32::from_rgb(180, 180, 200)).monospace());
                        ui.label(egui::RichText::new(format!("  Frame:        #{}", state.seq))
                            .color(egui::Color32::from_rgb(120, 120, 140)).monospace());

                        ui.add_space(20.0);
                        
                        // Gráfico de Fidelidad
                        ui.label(
                            egui::RichText::new("FIDELITY TIMELINE (60s window)")
                                .color(egui::Color32::from_rgb(150, 170, 200))
                                .size(14.0)
                        );
                        ui.add_space(4.0);
                        fidelity_plot::FidelityHistory::new(&self.history).ui(ui);
                    });
                });
            });
        
        // Repintar a 60 FPS
        ctx.request_repaint_after(std::time::Duration::from_millis(16));
    }
}
