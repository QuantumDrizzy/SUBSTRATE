//! Geomagnetic field coherence score computation.
//!
//! The COHERENCE INDEX (0.0–1.0) quantifies how stable the current
//! geomagnetic environment is relative to two reference states:
//!
//!   1.0 = modern interglacial baseline (quiet, stable field)
//!   0.0 = Laschamp event conditions (VADM≈15%, aurora at equator, Be-10 +300%)
//!
//! Three contributing signals:
//!   • Kp index        — direct geomagnetic disturbance measurement
//!   • CR proxy        — cosmic ray flux proxy via inverse F10.7
//!   • Pole drift rate — computed from WMM annual positions
//!
//! Laschamp reference values (41,000 BP):
//!   Kp_effective ≈ 9  (aurora visible at equator — field almost gone)
//!   CR flux      ≈ +300% above interglacial baseline
//!   Pole drift   ≈ effectively undefined/chaotic (500 km/yr upper bound)

use egui::Color32;
use crate::fetcher::DataState;

// ── Alert levels ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum AlertLevel {
    Nominal,   // score >= 0.70 — modern quiet-time baseline
    Watch,     // score  0.50–0.70 — elevated, monitor
    Warning,   // score  0.30–0.50 — significant departure
    Critical,  // score < 0.30  — Laschamp-like territory
}

impl AlertLevel {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Nominal  => "● NOMINAL",
            Self::Watch    => "◉ WATCH",
            Self::Warning  => "⚠ WARNING",
            Self::Critical => "⛔ CRITICAL",
        }
    }

    pub fn color(&self) -> Color32 {
        match self {
            Self::Nominal  => Color32::from_rgb(  0, 200,  80),
            Self::Watch    => Color32::from_rgb(255, 200,   0),
            Self::Warning  => Color32::from_rgb(255, 120,   0),
            Self::Critical => Color32::from_rgb(220,  50,  50),
        }
    }
}

// ── Result type ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// 0.0 = fully destabilized (Laschamp-like), 1.0 = fully stable
    pub score:            f32,
    /// Each component: 0.0 = calm/stable, 1.0 = extreme/destabilized
    pub kp_contrib:       f32,
    pub cr_contrib:       f32,
    pub drift_contrib:    f32,
    pub alert:            AlertLevel,
    /// Percentage toward Laschamp event conditions (0–100%)
    pub laschamp_pct:     f32,
    /// Raw current readings for display
    pub current_kp:       f64,
    pub current_f107:     f64,
    pub pole_drift_km_yr: f64,
    pub solar_cycle_phase: String,
}

// ── Physical constants ────────────────────────────────────────────────────────

const KP_MAX:            f64 = 9.0;
const F107_SOLAR_MIN:    f64 = 70.0;
const F107_SOLAR_MAX:    f64 = 310.0;
/// Estimated Laschamp-era CR flux anomaly as a fraction of modern solar-cycle swing.
/// Modern solar cycle: CR varies ~15% between min and max.
/// Laschamp: VADM=15% → CR ≈ 300–400% above quiet-time.
/// This maps our proxy (0-1 based on F10.7) to the Laschamp fraction.
const LASCHAMP_CR_SCALE: f64 = 4.0;
/// Upper bound on polar wander rate used for normalization (km/yr).
const LASCHAMP_DRIFT:    f64 = 500.0;

// ── Main computation ──────────────────────────────────────────────────────────

pub fn compute(data: &DataState) -> CoherenceResult {
    // ─ 1. Kp: mean of last 8 readings (~24h) ────────────────────────────────
    let recent_kp: Vec<f64> = data.kp.iter().rev().take(8).map(|r| r.kp).collect();
    let current_kp = if recent_kp.is_empty() {
        2.0 // quiet-time fallback when no data yet
    } else {
        recent_kp.iter().sum::<f64>() / recent_kp.len() as f64
    };
    let kp_contrib = ((current_kp / KP_MAX) as f32).clamp(0.0, 1.0);

    // ─ 2. CR proxy: inverse normalized F10.7 ────────────────────────────────
    // High F10.7 → strong solar wind → fewer CRs reach Earth → stable (low contrib)
    // Low F10.7  → weak modulation   → more CRs → destabilizing (high contrib)
    //
    // During Laschamp: CR increase was driven by VADM collapse, not solar.
    // This proxy captures only the modern solar-cycle component (~25% of full signal).
    let current_f107 = data.solar.last().map(|r| r.f107).unwrap_or(150.0);
    let f107_norm = ((current_f107 - F107_SOLAR_MIN)
        / (F107_SOLAR_MAX - F107_SOLAR_MIN))
        .clamp(0.0, 1.0);
    let cr_contrib = (1.0 - f107_norm) as f32;

    // ─ 3. Pole drift rate ────────────────────────────────────────────────────
    let pole_drift_km_yr = haversine_drift_rate(&data.pole);
    let drift_contrib = ((pole_drift_km_yr / LASCHAMP_DRIFT) as f32).clamp(0.0, 1.0);

    // ─ 4. Composite coherence ────────────────────────────────────────────────
    // Weights: Kp=0.40, CR=0.40, Drift=0.20
    let disturbance = 0.40 * kp_contrib + 0.40 * cr_contrib + 0.20 * drift_contrib;
    let score = (1.0 - disturbance).clamp(0.0, 1.0);

    // ─ 5. % toward Laschamp ─────────────────────────────────────────────────
    let kp_pct    = (current_kp / KP_MAX).clamp(0.0, 1.0);
    let cr_pct    = (cr_contrib as f64 / LASCHAMP_CR_SCALE).clamp(0.0, 1.0);
    let drift_pct = (pole_drift_km_yr / LASCHAMP_DRIFT).clamp(0.0, 1.0);
    let laschamp_pct = ((kp_pct * 0.40 + cr_pct * 0.40 + drift_pct * 0.20) * 100.0)
        .clamp(0.0, 100.0) as f32;

    // ─ 6. Solar cycle phase label ────────────────────────────────────────────
    let solar_cycle_phase = classify_solar_phase(current_f107);

    let alert = match score {
        s if s >= 0.70 => AlertLevel::Nominal,
        s if s >= 0.50 => AlertLevel::Watch,
        s if s >= 0.30 => AlertLevel::Warning,
        _              => AlertLevel::Critical,
    };

    CoherenceResult {
        score,
        kp_contrib,
        cr_contrib,
        drift_contrib,
        alert,
        laschamp_pct,
        current_kp,
        current_f107,
        pole_drift_km_yr,
        solar_cycle_phase,
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Haversine drift rate in km/yr between the last two WMM pole positions.
fn haversine_drift_rate(pole: &[crate::fetcher::PolePosition]) -> f64 {
    if pole.len() < 2 {
        return 35.0; // NOAA-published modern average as fallback
    }
    let a = &pole[pole.len() - 2];
    let b = &pole[pole.len() - 1];
    let dt = b.year - a.year;
    if dt <= 0.0 {
        return 35.0;
    }
    let r    = 6_371.0_f64; // Earth radius km
    let phi1 = a.lat.to_radians();
    let phi2 = b.lat.to_radians();
    let dphi = (b.lat - a.lat).to_radians();
    let dlam = (b.lon - a.lon).to_radians();
    let h = (dphi / 2.0).sin().powi(2)
        + phi1.cos() * phi2.cos() * (dlam / 2.0).sin().powi(2);
    let dist = 2.0 * r * h.sqrt().asin();
    dist / dt
}

fn classify_solar_phase(f107: f64) -> String {
    match f107 as u32 {
        0..=85    => "Solar Minimum".to_string(),
        86..=130  => "Rising Phase".to_string(),
        131..=200 => "Solar Maximum".to_string(),
        _         => "Solar Maximum (intense)".to_string(),
    }
}

/// Return last 8 Kp values for sparkline display.
pub fn kp_sparkline(data: &DataState) -> Vec<f64> {
    let mut v: Vec<f64> = data.kp.iter().rev().take(8).map(|r| r.kp).collect();
    v.reverse();
    v
}

pub fn kp_status_label(kp: f64) -> &'static str {
    match kp as u32 {
        0 | 1 => "QUIET",
        2     => "QUIET",
        3     => "UNSETTLED",
        4     => "ACTIVE",
        5     => "G1 MINOR STORM",
        6     => "G2 MODERATE",
        7     => "G3 STRONG",
        8     => "G4 SEVERE",
        _     => "G5 EXTREME",
    }
}
