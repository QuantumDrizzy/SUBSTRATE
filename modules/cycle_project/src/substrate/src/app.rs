//! app.rs — Ventana principal de SUBSTRATE

use egui::{Color32, Margin, RichText, Stroke};
use std::path::PathBuf;

use crate::state::{Panel, SharedState, make_shared_state};
use crate::panels::{
    geomagnetic::GeomagneticPanel,
    forward::ForwardPanel,
    quantum::QuantumPanel,
    simulation::SimulationPanel,
    myth::MythPanel,
    coherence::CoherencePanel,
};

// ── Paleta ────────────────────────────────────────────────────────────────────

const BG:       Color32 = Color32::from_rgb(  8, 10, 18);
const BG_PANEL: Color32 = Color32::from_rgb( 14, 18, 32);
const BG_SIDE:  Color32 = Color32::from_rgb( 10, 13, 25);
const BORDER:   Color32 = Color32::from_rgb( 40, 52, 88);
const DIM:      Color32 = Color32::from_rgb(110, 128, 165);
const ACCENT:   Color32 = Color32::from_rgb( 80, 160, 255);
const WHITE:    Color32 = Color32::WHITE;

// ── App ───────────────────────────────────────────────────────────────────────

pub struct SubstrateApp {
    state:      SharedState,
    active:     Panel,

    // Paneles (cada uno mantiene su propio estado)
    geo:  GeomagneticPanel,
    fwd:  ForwardPanel,
    qnt:  QuantumPanel,
    sim:  SimulationPanel,
    myth: MythPanel,
    coh:  CoherencePanel,
}

impl SubstrateApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let state = make_shared_state();

        // Bootstrap Python path
        let engines_dir = PathBuf::from("python/engines");
        if let Err(e) = crate::bridge::bootstrap_python(&engines_dir) {
            eprintln!("[SUBSTRATE] Python bootstrap error: {e}");
        }

        // Tema oscuro
        cc.egui_ctx.set_visuals(egui::Visuals::dark());

        Self {
            state:  state.clone(),
            active: Panel::Geomagnetic,
            geo:    GeomagneticPanel::new(state.clone()),
            fwd:    ForwardPanel::new(state.clone()),
            qnt:    QuantumPanel::new(state.clone()),
            sim:    SimulationPanel::new(state.clone()),
            myth:   MythPanel::new(state.clone()),
            coh:    CoherencePanel::new(state.clone()),
        }
    }
}

impl eframe::App for SubstrateApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Fondo global
        let mut vis = ctx.style().visuals.clone();
        vis.panel_fill       = BG;
        vis.window_fill      = BG;
        vis.extreme_bg_color = BG_PANEL;
        ctx.set_visuals(vis);

        // ── Header ────────────────────────────────────────────────────────────
        egui::TopBottomPanel::top("header")
            .exact_height(44.0)
            .frame(egui::Frame::default()
                .fill(BG_SIDE)
                .inner_margin(Margin::symmetric(16.0, 8.0))
                .stroke(Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new("SUBSTRATE")
                        .size(18.0).strong().color(WHITE));
                    ui.add_space(6.0);
                    ui.label(RichText::new("Quantum Research Platform")
                        .size(11.0).color(DIM));

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let data_ok = self.state.read().unwrap().data_loaded;
                        let (dot, label, color) = if data_ok {
                            ("●", "DATA OK", Color32::from_rgb(0, 200, 80))
                        } else {
                            ("●", "NO DATA", Color32::from_rgb(200, 80, 40))
                        };
                        ui.label(RichText::new(format!("{dot}  {label}"))
                            .size(11.0).color(color));
                    });
                });
            });

        // ── Sidebar ───────────────────────────────────────────────────────────
        egui::SidePanel::left("sidebar")
            .exact_width(180.0)
            .frame(egui::Frame::default()
                .fill(BG_SIDE)
                .inner_margin(Margin::symmetric(8.0, 12.0))
                .stroke(Stroke::new(1.0, BORDER)))
            .show(ctx, |ui| {
                ui.label(RichText::new("INSTRUMENTS").size(10.0).color(DIM));
                ui.add_space(8.0);

                for panel in Panel::all() {
                    let active = *panel == self.active;
                    let label = format!("{}  {}", panel.icon(), panel.label());
                    let text = if active {
                        RichText::new(label).size(13.0).strong().color(ACCENT)
                    } else {
                        RichText::new(label).size(13.0).color(DIM)
                    };

                    let btn = egui::Button::new(text)
                        .fill(if active { Color32::from_rgb(20, 35, 60) } else { Color32::TRANSPARENT })
                        .stroke(Stroke::new(
                            if active { 1.0 } else { 0.0 },
                            if active { BORDER } else { Color32::TRANSPARENT },
                        ))
                        .min_size(egui::vec2(164.0, 32.0));

                    if ui.add(btn).clicked() {
                        self.active = panel.clone();
                    }
                    ui.add_space(2.0);
                }
            });

        // ── Panel central ─────────────────────────────────────────────────────
        egui::CentralPanel::default()
            .frame(egui::Frame::default()
                .fill(BG)
                .inner_margin(Margin::same(12.0)))
            .show(ctx, |ui| {
                match self.active {
                    Panel::Geomagnetic => self.geo.ui(ui),
                    Panel::Forward     => self.fwd.ui(ui),
                    Panel::Quantum     => self.qnt.ui(ui),
                    Panel::Simulation  => self.sim.ui(ui),
                    Panel::Myth        => self.myth.ui(ui),
                    Panel::Coherence   => self.coh.ui(ui),
                }
            });
    }
}
