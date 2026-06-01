//! Panel: Field Coherence — lanza el binario Rust como subproceso

use egui::{Color32, RichText};
use std::process::{Child, Command};
use std::sync::{Arc, Mutex};

use crate::state::SharedState;
use super::shared::*;

pub struct CoherencePanel {
    state:   SharedState,
    process: Arc<Mutex<Option<Child>>>,
    offline: bool,
}

impl CoherencePanel {
    pub fn new(state: SharedState) -> Self {
        Self {
            state,
            process: Arc::new(Mutex::new(None)),
            offline: false,
        }
    }

    fn launch(&self) {
        // Busca el binario compilado
        let binary = find_binary("field_coherence_monitor");
        let mut proc = self.process.lock().unwrap();

        if let Some(ref mut child) = *proc {
            // Ya corriendo
            let _ = child.try_wait();
        } else {
            let mut cmd = Command::new(&binary);
            if self.offline { cmd.arg("--offline"); }
            match cmd.spawn() {
                Ok(child) => { *proc = Some(child); }
                Err(e)    => { eprintln!("[SUBSTRATE] coherence launch error: {e}"); }
            }
        }
    }

    fn kill(&self) {
        if let Some(ref mut child) = *self.process.lock().unwrap() {
            let _ = child.kill();
        }
        *self.process.lock().unwrap() = None;
    }

    fn is_running(&self) -> bool {
        if let Some(ref mut child) = *self.process.lock().unwrap() {
            matches!(child.try_wait(), Ok(None))
        } else {
            false
        }
    }

    pub fn ui(&mut self, ui: &mut egui::Ui) {
        panel_header(ui, "~  FIELD COHERENCE MONITOR",
            "Dashboard real-time: Kp index · F10.7 solar flux · pole drift · CR proxy");

        ui.add_space(10.0);

        card(ui, |ui| {
            ui.label(label_dim("BINARIO: field_coherence_monitor (Rust/egui)"));
            ui.add_space(6.0);

            let running = self.is_running();

            ui.horizontal(|ui| {
                if ui.add_enabled(!running,
                    egui::Button::new(RichText::new("▶  Abrir monitor").size(13.0))
                        .min_size(egui::vec2(160.0, 28.0))
                ).clicked() { self.launch(); }

                ui.add_space(8.0);

                if ui.add_enabled(running,
                    egui::Button::new(RichText::new("■  Cerrar").size(13.0))
                        .fill(Color32::from_rgb(60, 20, 20))
                        .min_size(egui::vec2(100.0, 28.0))
                ).clicked() { self.kill(); }

                ui.add_space(12.0);
                ui.checkbox(&mut self.offline, "Offline (cache only)");
            });

            ui.add_space(8.0);

            let (dot, label, color) = if running {
                ("●", "RUNNING", Color32::from_rgb(0, 200, 80))
            } else {
                ("○", "STOPPED", DIM)
            };
            ui.label(RichText::new(format!("{dot}  {label}")).size(12.0).color(color));
        });

        ui.add_space(12.0);

        card(ui, |ui| {
            ui.label(label_dim("DESCRIPCIÓN"));
            ui.add_space(6.0);
            ui.label(RichText::new(
                "El monitor de coherencia de campo es una ventana independiente (Rust/egui nativo) \
                 que se conecta a NOAA SWPC, WMM y proxies de rayos cósmicos en tiempo real.\n\n\
                 Datos mostrados:\n\
                 • Kp index — últimas 72 horas (intervalos de 3h)\n\
                 • F10.7 solar flux — últimos 6 años (mensual)\n\
                 • CR proxy — F10.7 invertido normalizado\n\
                 • Pole drift — latitud del polo magnético Norte 2000–2025 (WMM)\n\
                 • Coherence Index — score compuesto 0–1 vs condiciones Laschamp\n\n\
                 Para compilarlo: cargo build --release -p field_coherence_monitor"
            ).size(11.0).color(Color32::from_rgb(160, 170, 200)));
        });
    }
}

fn find_binary(name: &str) -> std::path::PathBuf {
    // Busca en target/release relativo al ejecutable actual
    let exe = std::env::current_exe().unwrap_or_default();
    let release_dir = exe.parent().unwrap_or(std::path::Path::new("."));

    let candidate = release_dir.join(name);
    if candidate.exists() { return candidate; }

    // Fallback: buscar en target/release desde el directorio de trabajo
    let fallback = std::path::PathBuf::from("src/field_coherence_monitor/target/release").join(name);
    if fallback.exists() { return fallback; }

    std::path::PathBuf::from(name)
}
