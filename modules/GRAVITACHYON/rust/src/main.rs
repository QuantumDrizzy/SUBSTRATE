// CHRONOS // SOVEREIGN V25.1 // THE_SISMOGRAPH_PROTOCOL
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use rodio::{OutputStream, Sink, source::Source};
use std::time::Duration;
use std::fs::File;
use std::io::BufReader;
use std::sync::mpsc::{self, Receiver};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

const COLOR_BG: egui::Color32 = egui::Color32::from_rgb(4, 4, 6);
const COLOR_GLASS: egui::Color32 = egui::Color32::from_rgba_premultiplied(10, 10, 15, 240);
const COLOR_CYAN: egui::Color32 = egui::Color32::from_rgb(0, 200, 255);
const COLOR_ORANGE: egui::Color32 = egui::Color32::from_rgb(255, 120, 0);
const COLOR_RED: egui::Color32 = egui::Color32::from_rgb(255, 30, 50);
const COLOR_GOLD: egui::Color32 = egui::Color32::from_rgb(255, 215, 0);
const COLOR_GREEN: egui::Color32 = egui::Color32::from_rgb(0, 255, 150);
const COLOR_LINE: egui::Color32 = egui::Color32::from_rgba_premultiplied(255, 255, 255, 8);

#[derive(Deserialize, Serialize, Clone)]
struct PythonPayload {
    time: f64,
    strain: f64,
    status: String,
    snr: f32,
}

fn reproducir_anomalia_gravitacional() {
    std::thread::spawn(|| {
        let (_stream, stream_handle) = match OutputStream::try_default() { Ok(s) => s, Err(_) => return };
        let sink = match Sink::try_new(&stream_handle) { Ok(s) => s, Err(_) => return };
        if let Ok(file) = File::open("assets/colapso.ogg") {
            if let Ok(source) = rodio::Decoder::new(BufReader::new(file)) {
                sink.append(source); sink.sleep_until_end(); return;
            }
        }
        let h1 = rodio::source::SineWave::new(150.0).amplify(0.2).take_duration(Duration::from_secs(6));
        let h2 = rodio::source::SineWave::new(154.0).amplify(0.2).take_duration(Duration::from_secs(6));
        let h3 = rodio::source::SineWave::new(148.0).amplify(0.2).take_duration(Duration::from_secs(6));
        sink.append(h1.mix(h2).mix(h3));
        sink.sleep_until_end();
    });
}

struct ChronosApp {
    boot_progress: f32,
    strain: f32,
    snr: f32,
    phase: f32,
    channels: Vec<(&'static str, f32, egui::Color32)>,
    receiver: Receiver<PythonPayload>,
    ringdown_buffer: VecDeque<[f64; 2]>,
    status_msg: String,
}

impl ChronosApp {
    fn new(receiver: Receiver<PythonPayload>) -> Self {
        let channels = vec![
            ("H1_STRAIN", 0.0, COLOR_CYAN), ("SNR", 26.2, COLOR_ORANGE),
            ("FINAL_SPIN", 0.68, COLOR_CYAN), ("ENTROPY_EXP", 0.02, COLOR_ORANGE),
            ("MASS_SOLAR", 65.4, COLOR_ORANGE), ("DIST_MPC", 410.0, COLOR_GREEN),
            ("CTC_PHASE", 0.18, COLOR_ORANGE),
        ];
        Self { boot_progress: 0.0, strain: 0.0, snr: 0.0, phase: 1.0, channels, receiver, ringdown_buffer: VecDeque::with_capacity(300), status_msg: "WAITING...".into() }
    }
}

fn draw_panel(ui: &mut egui::Ui, title: &str, subtitle: &str, w: f32, h: f32, content: impl FnOnce(&mut egui::Ui, egui::Rect)) {
    let (rect, _) = ui.allocate_at_least(egui::vec2(w, h), egui::Sense::hover());
    ui.painter().rect_filled(rect, 0.0, COLOR_GLASS);
    ui.painter().rect_stroke(rect, 0.0, egui::Stroke::new(1.0, COLOR_LINE));
    ui.painter().text(rect.min + egui::vec2(12.0, 14.0), egui::Align2::LEFT_TOP, title, egui::FontId::monospace(10.0), egui::Color32::WHITE);
    ui.painter().text(rect.max - egui::vec2(12.0, rect.height() - 14.0), egui::Align2::RIGHT_TOP, subtitle, egui::FontId::monospace(7.0), egui::Color32::from_gray(100));
    let body_rect = rect.shrink2(egui::vec2(1.0, 22.0));
    let mut child_ui = ui.new_child(egui::UiBuilder::new().max_rect(body_rect));
    content(&mut child_ui, body_rect);
}

fn draw_gauge(painter: &egui::Painter, center: egui::Pos2, radius: f32, value: f32, color: egui::Color32) {
    painter.circle_stroke(center, radius, egui::Stroke::new(1.0, egui::Color32::from_gray(40)));
    let angle = value.clamp(0.0, 1.0) * std::f32::consts::TAU * 0.75;
    let mut pts = Vec::new();
    for i in 0..20 {
        let a = (i as f32 / 20.0) * angle - std::f32::consts::FRAC_PI_2 - std::f32::consts::FRAC_PI_4;
        pts.push(center + egui::vec2(a.cos() * radius, a.sin() * radius));
    }
    painter.add(egui::Shape::line(pts, egui::Stroke::new(2.0, color)));
}

impl ChronosApp {
    fn setup_custom_styles(&self, ctx: &egui::Context) {
        let mut visuals = egui::Visuals::dark();
        visuals.window_fill = COLOR_BG; visuals.panel_fill = COLOR_BG;
        ctx.set_visuals(visuals);
    }

    fn update_physics(&mut self, _time: f64) {
        if self.boot_progress < 1.0 { self.boot_progress += 0.05; return; }
        while let Ok(payload) = self.receiver.try_recv() {
            self.status_msg = payload.status;
            self.snr = payload.snr;
            self.strain = payload.strain as f32;
            self.channels[1].1 = payload.snr;
            self.channels[0].1 = (payload.strain as f32).abs() * 5.0; 
            self.ringdown_buffer.push_back([payload.time, payload.strain]);
            if self.ringdown_buffer.len() > 300 { self.ringdown_buffer.pop_front(); }
        }
    }
}

impl eframe::App for ChronosApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.setup_custom_styles(ctx);
        self.update_physics(ctx.input(|i| i.time));

        if self.boot_progress < 1.0 {
            egui::CentralPanel::default().frame(egui::Frame::none().fill(COLOR_BG)).show(ctx, |ui| {
                ui.centered_and_justified(|ui| { ui.label(egui::RichText::new("CHRONOS // THE_SISMOGRAPH_PROTOCOL_ACTIVE").size(14.0).monospace().color(COLOR_CYAN)); });
            });
            ctx.request_repaint(); return;
        }

        egui::TopBottomPanel::top("t").frame(egui::Frame::none().fill(egui::Color32::from_rgb(10, 15, 20))).exact_height(24.0).show(ctx, |ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.add_space(12.0);
                let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                ui.label(egui::RichText::new(format!("MISSION CLOCK {} ⏱", ts)).color(egui::Color32::from_rgb(0, 255, 200)).monospace().size(10.0));
                ui.add_space(20.0);
                ui.label(egui::RichText::new("CHRONOS // DEFINITIVE_EDITION_V25.1").color(egui::Color32::from_gray(100)).size(8.0).monospace());
            });
        });

        egui::CentralPanel::default().frame(egui::Frame::none().fill(COLOR_BG).inner_margin(8.0)).show(ctx, |ui| {
            let (fw, fh) = (ui.available_width(), ui.available_height());
            let (cw, rh) = ((fw - 8.0) * 0.5, (fh - 8.0) * 0.5);

            egui::Grid::new("m").spacing(egui::vec2(8.0, 8.0)).show(ui, |ui| {
                // [1] TRISOLARIS TOPOLOGY
                draw_panel(ui, "TRISOLARIS // 3-BODY GRAVITY WELL", "CAOTIC_SINGULARITY_LATTICE", cw, rh, |ui, rect| {
                    Plot::new("top").show_axes([false, false]).show_grid(false).allow_drag(false).height(rh - 40.0).show(ui, |plot_ui| {
                        let t = ctx.input(|i| i.time);
                        let grid_size = 65; let scale = 0.6;
                        let cx1 = (t * 1.5).sin() * 8.0; let cy1 = (t * 1.5).cos() * 8.0;
                        let cx2 = (t * 1.1 + 2.0).sin() * 10.0; let cy2 = (t * 1.3 + 2.0).cos() * 6.0;
                        let cx3 = (t * 0.8 + 4.0).sin() * 5.0; let cy3 = (t * 0.9 + 4.0).cos() * 9.0;
                        for row in 0..grid_size {
                            let mut points = Vec::new();
                            for col in 0..grid_size {
                                let x = (col as f64 - 32.5) * scale; let y = (row as f64 - 32.5) * scale;
                                let d1 = ((x - cx1).powi(2) + (y - cy1).powi(2)).sqrt() + 2.0;
                                let d2 = ((x - cx2).powi(2) + (y - cy2).powi(2)).sqrt() + 2.0;
                                let d3 = ((x - cx3).powi(2) + (y - cy3).powi(2)).sqrt() + 2.0;
                                let z = -(40.0/d1) -(30.0/d2) -(30.0/d3) + (x*0.5 + t*5.0).sin()*0.5 + 10.0;
                                points.push([x - y * 0.6, (y * 0.25) + z]);
                            }
                            let fp = row as f32 / grid_size as f32;
                            plot_ui.line(Line::new(PlotPoints::from(points)).color(egui::Color32::from_rgb((255.0*fp) as u8, 20, (255.0*(1.0-fp)) as u8)).width(0.5));
                        }
                    });
                    let gauge_rect = egui::Rect::from_min_size(rect.right_top() - egui::vec2(130.0, -10.0), egui::vec2(120.0, 180.0));
                    for (i, (n, v, c)) in self.channels.iter().take(5).enumerate() {
                        let ty = gauge_rect.min.y + 10.0 + (i as f32 * 28.0);
                        ui.painter().text(egui::pos2(gauge_rect.min.x, ty), egui::Align2::LEFT_CENTER, n, egui::FontId::monospace(7.0), egui::Color32::from_gray(150));
                        draw_gauge(ui.painter(), egui::pos2(gauge_rect.max.x - 15.0, ty), 10.0, (*v / 100.0).clamp(0.0, 1.0), *c);
                    }
                });

                // [2] DEEP CTC SPECTROMETER
                draw_panel(ui, "DEEP CTC SPECTROMETER & ANALYSIS", "SIGNAL_EMERGENCE_CURVE", cw, rh, |ui, rect| {
                    let base_data: Vec<[f64; 2]> = (0..150).map(|i| { let x = i as f64 * 0.1; let y = (x * 0.25).exp(); [x, y] }).collect();
                    Plot::new("spec").show_axes([false, true]).height(rh - 40.0).show(ui, |plot_ui| {
                        for offset in 1..8 {
                            let off_f = offset as f64;
                            let echo_pts: PlotPoints = base_data.iter().map(|&[x, y]| [x + off_f * 0.1, y * (1.0 - off_f * 0.05)]).collect();
                            plot_ui.line(Line::new(echo_pts).color(egui::Color32::from_rgba_unmultiplied(255, 120, 0, (120 - offset*15).max(10) as u8)).width(3.0 + off_f as f32));
                        }
                        plot_ui.line(Line::new(PlotPoints::from(base_data)).color(COLOR_RED).width(3.0));
                    });
                    egui::Window::new("HUD").fixed_pos(rect.right_top() - egui::vec2(140.0, -20.0)).title_bar(false).frame(egui::Frame::none().fill(egui::Color32::from_black_alpha(200)).inner_margin(4.0).stroke(egui::Stroke::new(0.5, COLOR_LINE))).show(ui.ctx(), |ui| {
                        ui.label(egui::RichText::new("ANALYSIS HUD").size(7.0).color(COLOR_GOLD));
                        Plot::new("m1").width(120.0).height(40.0).show_axes([false, false]).show(ui, |plot_ui| {
                            let pts: PlotPoints = (0..60).map(|i| [i as f64, (i as f64 * 0.2 + ctx.input(|i| i.time) * self.phase as f64).cos() * 0.5 + 0.5]).collect();
                            plot_ui.line(Line::new(pts).color(COLOR_GOLD));
                        });
                    });
                });
                ui.end_row();

                // [3] RINGDOWN (LIVE SISMOGRAPH)
                draw_panel(ui, "RINGDOWN (LIVE STREAM)", "LIGO_BRIDGE_DATA_FLOW", cw, rh, |ui, rect| {
                    if self.ringdown_buffer.is_empty() {
                        ui.centered_and_justified(|ui| { ui.label(egui::RichText::new("WAITING FOR PYTHON BRIDGE...\n(Run python bridge_python.py)").color(egui::Color32::GRAY).italics()); });
                    } else {
                        let last_t = self.ringdown_buffer.back().map(|p| p[0]).unwrap_or(0.0);
                        Plot::new("ring")
                            .show_axes([false, true])
                            .show_grid(true)
                            .height(rh - 45.0)
                            .include_x(last_t)
                            .include_x(last_t - 10.0) // Scrolling window of 10 units
                            .show(ui, |plot_ui| {
                                let pts: Vec<[f64; 2]> = self.ringdown_buffer.iter().cloned().collect();
                                plot_ui.line(Line::new(PlotPoints::from(pts)).color(COLOR_CYAN).width(2.0));
                            });
                        ui.painter().text(rect.left_top() + egui::vec2(10.0, 30.0), egui::Align2::LEFT_TOP, format!("STATUS: {}", self.status_msg), egui::FontId::monospace(7.0), COLOR_GREEN);
                    }
                });

                // [4] QUANTUM-TACHYON PIPELINE
                draw_panel(ui, "QUANTUM-TACHYON PIPELINE", "PIPELINE_FLOW_LOGS", cw, rh, |ui, _rect| {
                    ui.vertical_centered(|ui| {
                        ui.add_space(35.0);
                        ui.columns(4, |cols| {
                            let box_size = egui::vec2(110.0, 50.0);
                            let estilo = egui::Frame::none().stroke(egui::Stroke::new(1.0, COLOR_GREEN.gamma_multiply(0.8))).rounding(4.0).inner_margin(5.0).fill(egui::Color32::from_black_alpha(150));
                            let draw_box = |ui: &mut egui::Ui, title: &str, sub: &str, active: bool| {
                                let s = if active { estilo.stroke(egui::Stroke::new(2.0, COLOR_ORANGE)) } else { estilo };
                                s.show(ui, |ui| { ui.set_min_size(box_size); ui.vertical_centered(|ui| { ui.add_space(8.0); ui.label(egui::RichText::new(title).color(egui::Color32::WHITE).strong().size(9.0)); ui.label(egui::RichText::new(sub).color(egui::Color32::GRAY).size(7.0)); }); });
                            };
                            cols[0].vertical_centered(|ui| draw_box(ui, "LIGO", "DATA IN", true));
                            cols[1].vertical_centered(|ui| { ui.horizontal(|ui| { ui.label(egui::RichText::new(" >> ").size(18.0).color(COLOR_GREEN)); draw_box(ui, "Q-FOAM", "GENERATOR", false); }); });
                            cols[2].vertical_centered(|ui| { ui.horizontal(|ui| { ui.label(egui::RichText::new(" >> ").size(18.0).color(COLOR_GREEN)); draw_box(ui, "MERA", "TENSOR", false); }); });
                            cols[3].vertical_centered(|ui| { ui.horizontal(|ui| { ui.label(egui::RichText::new(" >> ").size(18.0).color(COLOR_GREEN)); draw_box(ui, "CTC", "DECODER", false); }); });
                        });
                        ui.add_space(15.0);
                        if ui.button(egui::RichText::new("⚠ INICIAR SECUENCIA TRISOLARIS").color(egui::Color32::BLACK).background_color(COLOR_RED)).clicked() {
                            reproducir_anomalia_gravitacional();
                        }
                    });
                });
            });
        });
        ctx.request_repaint();
    }
}

fn main() -> eframe::Result<()> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let context = zmq::Context::new();
        let subscriber = context.socket(zmq::SUB).expect("Failed to create ZMQ socket");
        subscriber.connect("tcp://localhost:5555").expect("Failed to connect to Python Bridge");
        subscriber.set_subscribe(b"RINGDOWN").expect("Failed to subscribe");
        loop {
            if let Ok(msg) = subscriber.recv_string(0) {
                if let Ok(data_str) = msg {
                    let json_part = data_str.trim_start_matches("RINGDOWN ").trim();
                    if let Ok(payload) = serde_json::from_str::<PythonPayload>(json_part) {
                        let _ = tx.send(payload);
                    }
                }
            }
        }
    });
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]).with_title("CHRONOS // THE_SISMOGRAPH_PROTOCOL"),
        ..Default::default()
    };
    eframe::run_native("CHRONOS", native_options, Box::new(|_| Ok(Box::new(ChronosApp::new(rx)))))
}
