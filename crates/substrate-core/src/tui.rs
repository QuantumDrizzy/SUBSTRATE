use crate::{Layer, LayerResult, LayerStatus, SubstrateState, TuiMsg};
use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph},
    Frame, Terminal,
};
use std::{
    collections::HashMap,
    io,
    sync::mpsc::Receiver,
    time::{Duration, Instant},
};

pub struct Dashboard;

impl Dashboard {
    /// Live TUI: receives TuiMsg from engine thread, redraws at ~10 Hz.
    pub fn run_live(rx: Receiver<TuiMsg>) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        let start     = Instant::now();
        let mut state = LiveState::new();

        loop {
            // Drain all pending messages
            loop {
                match rx.try_recv() {
                    Ok(TuiMsg::LayerStarted(l)) => state.mark_running(l),
                    Ok(TuiMsg::LayerDone(r))    => state.mark_done(r),
                    Ok(TuiMsg::AllDone)         => state.all_done = true,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        state.all_done = true;
                        break;
                    }
                    Err(_) => break,
                }
            }

            let elapsed  = start.elapsed();
            let results  = state.sorted_results();
            let coherence = state.partial_coherence();
            terminal.draw(|f| render_frame(f, &results, coherence, elapsed, state.all_done))?;

            if event::poll(Duration::from_millis(100))? {
                if let Event::Key(k) = event::read()? {
                    if matches!(k.code, KeyCode::Char('q') | KeyCode::Esc) {
                        break;
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }

    /// Render one frame to a 120×35 TestBackend and write it as text.
    pub fn dump_frame(state: &SubstrateState, path: &std::path::Path) -> Result<()> {
        use ratatui::backend::TestBackend;

        let backend  = TestBackend::new(120, 35);
        let mut term = Terminal::new(backend)?;

        let mut results = state.results.clone();
        for l in Layer::all() {
            if !results.iter().any(|r| r.layer == l) {
                results.push(LayerResult::idle(l, 1.0));
            }
        }
        results.sort_by_key(|r| r.layer);

        term.draw(|f| {
            render_frame(f, &results, state.coherence_score, Duration::ZERO, true);
        })?;

        let buf   = term.backend().buffer().clone();
        let area  = buf.area;
        let mut text = String::with_capacity((area.width as usize + 1) * area.height as usize);
        for y in 0..area.height {
            for x in 0..area.width {
                text.push_str(buf.get(x, y).symbol());
            }
            text.push('\n');
        }

        std::fs::create_dir_all(
            path.parent().unwrap_or_else(|| std::path::Path::new(".")),
        )?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Legacy one-shot render: draw final state, wait for Q/Esc.
    pub fn run_once(state: SubstrateState) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;

        terminal.draw(|f| {
            render_frame(f, &state.results, state.coherence_score, Duration::ZERO, true);
        })?;

        loop {
            if event::poll(Duration::from_millis(200))? {
                if let Event::Key(k) = event::read()? {
                    if matches!(k.code, KeyCode::Char('q') | KeyCode::Esc) {
                        break;
                    }
                }
            }
        }

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        Ok(())
    }
}

// ── live state tracker ────────────────────────────────────────────────────────

struct LiveState {
    layers:   HashMap<Layer, LayerResult>,
    pub all_done: bool,
}

impl LiveState {
    fn new() -> Self {
        let weights: &[(Layer, f64)] = &[
            (Layer::Quantum,      1.5),
            (Layer::Geomagnetic,  2.0),
            (Layer::Magnon,       1.2),
            (Layer::QuantumLab,   1.0),
            (Layer::Solar,        1.3),
            (Layer::Cosmological, 0.8),
            (Layer::Eeg,          1.8),   // high weight — direct biosensor
            (Layer::Lunar,        1.4),
            (Layer::Radio,        1.6),
            (Layer::Seismic,      1.1),
        ];
        let layers = weights
            .iter()
            .map(|&(l, w)| (l, LayerResult::idle(l, w)))
            .collect();
        Self { layers, all_done: false }
    }

    fn mark_running(&mut self, layer: Layer) {
        if let Some(r) = self.layers.get_mut(&layer) {
            r.status = LayerStatus::Running;
        }
    }

    fn mark_done(&mut self, result: LayerResult) {
        self.layers.insert(result.layer, result);
    }

    fn sorted_results(&self) -> Vec<LayerResult> {
        let mut v: Vec<LayerResult> = self.layers.values().cloned().collect();
        v.sort_by_key(|r| r.layer);
        v
    }

    fn partial_coherence(&self) -> f64 {
        let done: Vec<&LayerResult> = self
            .layers
            .values()
            .filter(|r| matches!(r.status, LayerStatus::Done))
            .collect();
        if done.is_empty() {
            return 0.0;
        }
        let tw: f64 = done.iter().map(|r| r.weight).sum();
        done.iter().map(|r| r.score * r.weight).sum::<f64>() / tw
    }
}

// ── rendering ─────────────────────────────────────────────────────────────────

fn render_frame(
    f: &mut Frame,
    results: &[LayerResult],
    coherence: f64,
    elapsed: Duration,
    done: bool,
) {
    let area = f.size();

    let outer = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(3),
        ])
        .split(area);

    render_title(f, outer[0]);

    // 3-row grid  (row1: 3 panels | row2: 3 panels | row3: 1 panel — EEG biosensor)
    let grid_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(34),
            Constraint::Percentage(33),
            Constraint::Percentage(33),
        ])
        .split(outer[1]);

    let row_constraints_4 = [
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
        Constraint::Percentage(25),
    ];
    let row_constraints_3 = [
        Constraint::Percentage(33),
        Constraint::Percentage(33),
        Constraint::Percentage(34),
    ];
    let row1 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(row_constraints_4)
        .split(grid_rows[0]);
    let row2 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(row_constraints_3)
        .split(grid_rows[1]);
    let row3 = Layout::default()
        .direction(Direction::Horizontal)
        .constraints(row_constraints_3)
        .split(grid_rows[2]);

    let grid: &[(Layer, Rect)] = &[
        (Layer::Geomagnetic,   row1[0]),
        (Layer::Quantum,       row1[1]),
        (Layer::Magnon,        row1[2]),
        (Layer::QuantumLab,    row1[3]),
        (Layer::Solar,         row2[0]),
        (Layer::Cosmological,  row2[1]),
        (Layer::Eeg,           row2[2]),
        (Layer::Lunar,         row3[0]),
        (Layer::Radio,         row3[1]),
        (Layer::Seismic,       row3[2]),
    ];

    for &(layer, rect) in grid {
        let result = results
            .iter()
            .find(|r| r.layer == layer)
            .cloned()
            .unwrap_or_else(|| LayerResult::idle(layer, 1.0));
        render_panel(f, &result, rect);
    }

    render_status(f, outer[2], coherence, elapsed, done);
}

fn render_title(f: &mut Frame, area: Rect) {
    let ts   = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC");
    let left = " SUBSTRATE v0.1.0 — Unified Field Analysis System";
    let text = format!("{left:<70}{ts:>45}");
    let p = Paragraph::new(text)
        .block(Block::default().borders(Borders::ALL))
        .style(Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD));
    f.render_widget(p, area);
}

fn render_status(
    f: &mut Frame,
    area: Rect,
    coherence: f64,
    elapsed: Duration,
    done: bool,
) {
    let status   = if done { "COMPLETE" } else { "RUNNING " };
    let elapsed_s = format!("{:.1}s", elapsed.as_secs_f64());
    let title    = format!(" {status}  Elapsed: {elapsed_s}  [Q] quit ");
    let pct      = (coherence * 100.0).clamp(0.0, 100.0) as u16;
    let gauge = Gauge::default()
        .block(Block::default().borders(Borders::ALL).title(title))
        .gauge_style(Style::default().fg(score_color(coherence)))
        .percent(pct)
        .label(format!("Coherence  {coherence:.4}"));
    f.render_widget(gauge, area);
}

fn render_panel(f: &mut Frame, result: &LayerResult, area: Rect) {
    let border_color = match &result.status {
        LayerStatus::Done     => Color::Green,
        LayerStatus::Running  => Color::Yellow,
        LayerStatus::Error(_) => Color::Red,
        LayerStatus::Idle     => Color::DarkGray,
    };

    let status_badge = match &result.status {
        LayerStatus::Done     => " \u{2713} ", // ✓
        LayerStatus::Running  => " \u{2026} ", // …
        LayerStatus::Error(_) => " \u{2717} ", // ✗
        LayerStatus::Idle     => "   ",
    };

    let title = Span::styled(
        format!(" {}{}", result.layer.name().to_uppercase(), status_badge),
        Style::default()
            .fg(border_color)
            .add_modifier(Modifier::BOLD),
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(border_color))
        .title(title);

    let inner = block.inner(area);
    f.render_widget(block, area);

    let content: Vec<Line> = match &result.status {
        LayerStatus::Idle => {
            vec![Line::from(Span::styled(
                "  waiting…",
                Style::default().fg(Color::DarkGray),
            ))]
        }
        LayerStatus::Running => {
            vec![Line::from(Span::styled(
                "  running…",
                Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ))]
        }
        LayerStatus::Error(e) => {
            let msg = e.lines().next().unwrap_or("unknown error");
            let snippet = msg.chars().take(inner.width.saturating_sub(4) as usize).collect::<String>();
            vec![Line::from(Span::styled(
                format!("  ERR: {snippet}"),
                Style::default().fg(Color::Red),
            ))]
        }
        LayerStatus::Done => {
            let sc  = result.score;
            let bar = score_bar(sc);
            let [detail0, detail1] = layer_details(result);
            vec![
                Line::from(vec![
                    Span::raw("  "),
                    Span::styled(bar, Style::default().fg(score_color(sc))),
                    Span::styled(
                        format!("  {sc:.4}"),
                        Style::default().fg(score_color(sc)).add_modifier(Modifier::BOLD),
                    ),
                ]),
                Line::from(Span::raw(format!("  {detail0}"))),
                Line::from(Span::styled(
                    format!("  {detail1}"),
                    Style::default().fg(Color::DarkGray),
                )),
            ]
        }
    };

    f.render_widget(Paragraph::new(content), inner);
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn score_bar(score: f64) -> String {
    let filled = ((score * 20.0).round() as usize).min(20);
    format!("{}{}", "\u{2588}".repeat(filled), "\u{2591}".repeat(20 - filled))
}

fn score_color(score: f64) -> Color {
    if score >= 0.7 {
        Color::Green
    } else if score >= 0.4 {
        Color::Yellow
    } else {
        Color::Red
    }
}

fn layer_details(r: &LayerResult) -> [String; 2] {
    let d = &r.metadata["data"];
    let g = |key: &str| d[key].as_f64().unwrap_or(0.0);
    let i = |key: &str| d[key].as_i64().unwrap_or(0);

    match r.layer {
        Layer::Geomagnetic => [
            format!("P(excursion): {:.4}", g("pre_excursion_prob")),
            format!("LSTM-1kyr: {:.4}  5kyr: {:.4}", g("lstm_vadm_1kyr"), g("lstm_vadm_5kyr")),
        ],
        Layer::Quantum => [
            format!("Phi_S: {:.4}  Trace_f: {:.4}", g("singlet_yield"), g("final_trace")),
            format!("FAD/W  n_sites={}", i("n_sites")),
        ],
        Layer::Magnon => [
            format!("Fidelity: {:.4}  T2: {:.2}us", g("fidelity"), g("t2_effective_us")),
            format!("Noise: {:.2e}T  Purity: {:.4}", g("noise_b_rms_T"), g("purity")),
        ],
        Layer::QuantumLab => [
            format!("Z: {:.3e}  lnZ: {:.4}", g("partition_function_Z"), g("log_Z")),
            format!("F/site: {:.4}  L={}  b={:.1}", g("free_energy_per_site"), i("lattice_L"), g("beta")),
        ],
        Layer::Solar => [
            format!("Phase: {:.4}  Score: {:.4}", g("cycle_phase"), r.score),
            "Method: 11-yr sinusoidal SC25".to_owned(),
        ],
        Layer::Cosmological => [
            format!("C2: {:.6}  Hemi: {:.4}", g("quadrupole_C2"), g("hemi_asymmetry")),
            format!("C2/exp: {:.4}  nside={}  lmax={}", g("c2_ratio"), i("nside"), i("lmax")),
        ],
        Layer::Eeg => [
            format!("alpha: {:.4}  theta: {:.4}", g("alpha_rel"), g("band_powers.theta")),
            format!("mode: {}  ch={}  {}", d["mode"].as_str().unwrap_or("?"), i("n_channels"),
                    if d["synthetic"].as_bool().unwrap_or(true) { "[SIM]" } else { "[LIVE]" }),
        ],
        Layer::Lunar => [
            format!("Phase: {:.4} ({})", g("phase"), d["phase_name"].as_str().unwrap_or("?")),
            format!("Dist: {:.0}km  g_pert: {:.2e}", g("distance_km"), g("gravity_perturbation")),
        ],
        Layer::Radio => [
            format!("Deficit: {:.4}  Asym: {:.4}", g("quadrupole_deficit"), g("hemispheric_asymmetry")),
            format!("SDR: {}  Floor: {:.1}dB", if d["has_rtlsdr"].as_bool().unwrap_or(false) { "YES" } else { "NO" }, g("noise_floor_db")),
        ],
        Layer::Seismic => [
            format!("Events/24h: {}  MaxMag: {:.1}", i("events_24h"), g("max_magnitude")),
            format!("EnergyRel: {:.4}  Sig={}", g("total_energy_relative"), i("significant_events")),
        ],
    }
}
