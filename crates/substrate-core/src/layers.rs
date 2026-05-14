use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, PartialOrd, Ord)]
pub enum Layer {
    Cosmological,
    Eeg,
    Geomagnetic,
    Lunar,
    Magnon,
    Quantum,
    QuantumLab,
    Radio,
    Seismic,
    Solar,
}

impl Layer {
    pub fn name(self) -> &'static str {
        match self {
            Layer::Quantum       => "quantum",
            Layer::Geomagnetic   => "geomagnetic",
            Layer::Magnon        => "magnon",
            Layer::QuantumLab    => "quantum_lab",
            Layer::Solar         => "solar",
            Layer::Cosmological  => "cosmological",
            Layer::Eeg           => "eeg",
            Layer::Lunar         => "lunar",
            Layer::Radio         => "radio",
            Layer::Seismic       => "seismic",
        }
    }

    pub fn all() -> Vec<Layer> {
        vec![
            Layer::Quantum,
            Layer::Geomagnetic,
            Layer::Magnon,
            Layer::QuantumLab,
            Layer::Solar,
            Layer::Cosmological,
            Layer::Eeg,
            Layer::Lunar,
            Layer::Radio,
            Layer::Seismic,
        ]
    }
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerStatus {
    Idle,
    Running,
    Done,
    Error(String),
}

impl std::fmt::Display for LayerStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerStatus::Idle        => write!(f, "IDLE"),
            LayerStatus::Running     => write!(f, "RUNNING"),
            LayerStatus::Done        => write!(f, "DONE"),
            LayerStatus::Error(e)    => write!(f, "ERROR: {e}"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerResult {
    pub layer:    Layer,
    pub status:   LayerStatus,
    pub score:    f64,
    pub weight:   f64,
    pub metadata: serde_json::Value,
}

impl LayerResult {
    pub fn idle(layer: Layer, weight: f64) -> Self {
        Self {
            layer,
            status:   LayerStatus::Idle,
            score:    0.0,
            weight,
            metadata: serde_json::Value::Null,
        }
    }
}
