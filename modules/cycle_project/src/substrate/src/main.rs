//! SUBSTRATE — Quantum Research Platform
//! =======================================
//! Punto de entrada. Inicializa Python (PyO3) y lanza la ventana egui.
//!
//! Uso:
//!   cargo run --release
//!   cargo run --release -- --data-dir /ruta/a/data

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod panels;
mod bridge;
mod state;

use anyhow::Result;

fn main() -> Result<()> {
    // Inicializar Python embebido (PyO3)
    pyo3::prepare_freethreaded_python();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("SUBSTRATE  |  Quantum Research Platform")
            .with_inner_size([1600.0, 960.0])
            .with_min_inner_size([1100.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "substrate",
        native_options,
        Box::new(|cc| Box::new(app::SubstrateApp::new(cc))),
    ).map_err(|e| anyhow::anyhow!("{e}"))
}
