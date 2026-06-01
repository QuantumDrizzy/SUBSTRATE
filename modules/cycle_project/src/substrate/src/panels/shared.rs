//! Widgets y helpers compartidos por todos los paneles.

use egui::{Color32, Margin, RichText, Stroke};
use crate::state::JobStatus;

pub const BG_PANEL: Color32 = Color32::from_rgb(14, 18, 32);
pub const BORDER:   Color32 = Color32::from_rgb(40, 52, 88);
pub const DIM:      Color32 = Color32::from_rgb(110, 128, 165);
pub const ACCENT:   Color32 = Color32::from_rgb(80, 160, 255);
pub const OK:       Color32 = Color32::from_rgb(0, 200, 80);
pub const ERR:      Color32 = Color32::from_rgb(220, 60, 60);
pub const WARN:     Color32 = Color32::from_rgb(220, 160, 40);

pub fn card(ui: &mut egui::Ui, content: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::default()
        .fill(BG_PANEL)
        .stroke(Stroke::new(1.0, BORDER))
        .inner_margin(Margin::same(12.0))
        .show(ui, content);
}

pub fn panel_header(ui: &mut egui::Ui, title: &str, subtitle: &str) {
    ui.label(RichText::new(title).size(16.0).strong().color(Color32::WHITE));
    ui.label(RichText::new(subtitle).size(11.0).color(DIM));
    ui.separator();
}

pub fn label_dim(text: &str) -> RichText {
    RichText::new(text).size(10.0).color(DIM)
}

pub fn metric_chip(ui: &mut egui::Ui, label: &str, value: &str) {
    egui::Frame::default()
        .fill(Color32::from_rgb(20, 26, 46))
        .stroke(Stroke::new(1.0, BORDER))
        .inner_margin(Margin::symmetric(10.0, 5.0))
        .show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(label).size(10.0).color(DIM));
                ui.add_space(6.0);
                ui.label(RichText::new(value).size(13.0).strong().color(ACCENT));
            });
        });
    ui.add_space(4.0);
}

pub fn render_job_status(ui: &mut egui::Ui, status: &JobStatus) {
    match status {
        JobStatus::Idle       => {}
        JobStatus::Running    => {
            ui.label(RichText::new("● Ejecutando…").size(11.0).color(WARN));
        }
        JobStatus::Done       => {
            ui.label(RichText::new("● Completado").size(11.0).color(OK));
        }
        JobStatus::Error(e)   => {
            ui.label(RichText::new(format!("● Error: {e}")).size(11.0).color(ERR));
        }
    }
}
