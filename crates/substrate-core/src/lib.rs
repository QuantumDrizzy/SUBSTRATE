pub mod config;
pub mod engine;
pub mod layers;
pub mod tui;

pub use engine::SubstrateEngine;
pub use layers::{Layer, LayerResult, LayerStatus};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Messages sent from the computation engine to the live TUI dashboard.
#[derive(Debug)]
pub enum TuiMsg {
    LayerStarted(Layer),
    LayerDone(LayerResult),
    AllDone,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubstrateState {
    pub run_id:          u64,
    pub timestamp:       DateTime<Utc>,
    pub results:         Vec<LayerResult>,
    pub coherence_score: f64,
}

impl SubstrateState {
    pub fn new(results: Vec<LayerResult>) -> Self {
        let coherence_score = network_influence_score(&results);
        Self {
            run_id: 0,
            timestamp: Utc::now(),
            results,
            coherence_score,
        }
    }
}

/// Computes the network synchronization / cross-correlation across all physical domains.
/// Models the platform as an influence network where layers couple to one another.
fn network_influence_score(results: &[LayerResult]) -> f64 {
    let n = results.len();
    if n <= 1 {
        let total_w: f64 = results.iter().map(|r| r.weight).sum();
        if total_w == 0.0 { return 0.0; }
        return results.iter().map(|r| r.score * r.weight).sum::<f64>() / total_w;
    }

    let mut sum_coupling = 0.0;
    let mut sum_weights = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let ri = &results[i];
            let rj = &results[j];
            let w_pair = ri.weight * rj.weight;
            // Cross-correlation proxy: phase/score mutual alignment
            let coupling = 1.0 - (ri.score - rj.score).abs();
            sum_coupling += coupling * w_pair;
            sum_weights += w_pair;
        }
    }

    if sum_weights == 0.0 {
        return 0.0;
    }
    sum_coupling / sum_weights
}
