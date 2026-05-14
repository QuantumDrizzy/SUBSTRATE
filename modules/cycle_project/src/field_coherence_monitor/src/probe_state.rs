//! Loads forward-probe results written by run_forward_probe.py.

use std::path::Path;
use serde::Deserialize;

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProbeState {
    pub pre_excursion_prob: f64,
    pub lstm_vadm_1kyr: f64,
    pub lstm_vadm_5kyr: f64,
    pub instrumental_threshold_yr: u32,
    pub generated_at: String,
}

impl ProbeState {
    pub fn load(path: &Path) -> Option<ProbeState> {
        let text = std::fs::read_to_string(path).ok()?;
        serde_json::from_str(&text).ok()
    }
}
