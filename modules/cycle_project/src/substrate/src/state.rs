//! Estado compartido de la app (thread-safe).

use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};

// ── Panel activo ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Panel {
    Geomagnetic,
    Forward,
    Quantum,
    Simulation,
    Myth,
    Coherence,
}

impl Panel {
    pub fn label(&self) -> &'static str {
        match self {
            Panel::Geomagnetic => "Geomagnetic",
            Panel::Forward     => "Forward Probe",
            Panel::Quantum     => "Quantum Bio",
            Panel::Simulation  => "Simulation",
            Panel::Myth        => "Myth RAG",
            Panel::Coherence   => "Field Coherence",
        }
    }

    pub fn icon(&self) -> &'static str {
        match self {
            Panel::Geomagnetic => "⊕",
            Panel::Forward     => "→",
            Panel::Quantum     => "ψ",
            Panel::Simulation  => "◈",
            Panel::Myth        => "☽",
            Panel::Coherence   => "~",
        }
    }

    pub fn all() -> &'static [Panel] {
        &[
            Panel::Geomagnetic,
            Panel::Forward,
            Panel::Quantum,
            Panel::Simulation,
            Panel::Myth,
            Panel::Coherence,
        ]
    }
}

// ── Estado de un job (análisis en curso / completado) ─────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum JobStatus {
    Idle,
    Running,
    Done,
    Error(String),
}

// ── Resultado de análisis genérico ────────────────────────────────────────────

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub values: Vec<f64>,        // serie temporal o puntuaciones
    pub labels: Vec<String>,     // etiquetas eje X
    pub scalars: Vec<(String, f64)>, // métricas clave
    pub text: String,            // resumen / informe
}

// ── Estado global compartido ──────────────────────────────────────────────────

#[derive(Debug)]
pub struct AppState {
    pub data_dir:    std::path::PathBuf,
    pub data_loaded: bool,
}

impl AppState {
    pub fn new() -> Self {
        let data_dir = std::path::PathBuf::from("data/processed");
        Self { data_dir, data_loaded: false }
    }
}

pub type SharedState = Arc<RwLock<AppState>>;

pub fn make_shared_state() -> SharedState {
    Arc::new(RwLock::new(AppState::new()))
}
