use crate::layers::{Layer, LayerResult, LayerStatus};
use crate::config::SubstrateConfig;
use crate::{SubstrateState, TuiMsg};
use anyhow::Result;
use std::path::PathBuf;
use tokio::task::JoinSet;

pub struct SubstrateEngine {
    config:       SubstrateConfig,
    project_root: PathBuf,
}

impl SubstrateEngine {
    pub fn new() -> Self {
        let config = SubstrateConfig::load().unwrap_or_else(|e| {
            tracing::warn!("Failed to load substrate.toml: {e}. Using defaults.");
            SubstrateConfig::default()
        });

        // Binary lives at <root>/target/{release,debug}/substrate[.exe].
        // Walk up: exe -> release/ -> target/ -> project root.
        let project_root = std::env::current_exe()
            .ok()
            .and_then(|exe| {
                exe.parent()
                    .and_then(|p| p.parent())
                    .and_then(|p| p.parent())
                    .map(|p| p.to_path_buf())
            })
            .filter(|p| p.join("engine").is_dir())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        tracing::debug!("project_root={}", project_root.display());

        Self {
            config,
            project_root,
        }
    }

    /// Run all enabled layers in parallel; return aggregated state.
    pub async fn run(&self) -> Result<SubstrateState> {
        self.run_streaming_filtered(std::sync::mpsc::channel().0, "all").await
    }

    /// Run all enabled layers, streaming TuiMsg updates.
    pub async fn run_streaming(
        &self,
        tx: std::sync::mpsc::Sender<TuiMsg>,
    ) -> Result<SubstrateState> {
        self.run_streaming_filtered(tx, "all").await
    }

    /// Like `run_streaming` but only spawns layers whose names appear in
    /// `filter` (comma-separated, case-insensitive). `"all"` or `""` runs
    /// every enabled layer from config.
    pub async fn run_streaming_filtered(
        &self,
        tx:     std::sync::mpsc::Sender<TuiMsg>,
        filter: &str,
    ) -> Result<SubstrateState> {
        let want: Option<Vec<String>> = if filter == "all" || filter.is_empty() {
            None
        } else {
            Some(filter.split(',').map(|s| s.trim().to_lowercase()).collect())
        };

        tracing::info!("SUBSTRATE streaming run  filter={filter}");
        let mut set: JoinSet<Result<LayerResult>> = JoinSet::new();

        // Resolve the absolute engine/ path once before spawning tasks.
        // project_root is already derived from current_exe() in ::new(), so
        // this works regardless of the working directory the binary is launched from.
        let engine_dir = self.project_root
            .join("engine")
            .to_string_lossy()
            .into_owned();
        tracing::debug!("engine_dir={engine_dir}");

        // Get layers from config
        for (name, cfg) in &self.config.layers {
            if !cfg.enabled { continue; }
            
            // Map name string to Layer enum
            let layer = match name.as_str() {
                "quantum"      => Layer::Quantum,
                "geomagnetic"  => Layer::Geomagnetic,
                "magnon"       => Layer::Magnon,
                "quantum_lab"  => Layer::QuantumLab,
                "solar"        => Layer::Solar,
                "cosmological" => Layer::Cosmological,
                "eeg"          => Layer::Eeg,
                "lunar"        => Layer::Lunar,
                "radio"        => Layer::Radio,
                "seismic"      => Layer::Seismic,
                _ => {
                    tracing::warn!("Unknown layer in config: {name}");
                    continue;
                }
            };

            if let Some(ref names) = want {
                if !names.contains(&name.to_lowercase()) { continue; }
            }

            let weight     = cfg.weight;
            let tx2        = tx.clone();
            let params     = cfg.params.clone();
            let engine_dir = engine_dir.clone();

            set.spawn(async move {
                let _ = tx2.send(TuiMsg::LayerStarted(layer));

                // Use spawn_blocking for FFI calls as they are not async
                let result = tokio::task::spawn_blocking(move || {
                    invoke_layer_ffi(layer, weight, params, engine_dir)
                }).await?;
                
                if let Ok(ref r) = result {
                    let _ = tx2.send(TuiMsg::LayerDone(r.clone()));
                }
                result
            });
        }

        let mut results = drain(&mut set).await;
        let _ = tx.send(TuiMsg::AllDone);
        results.sort_by_key(|r| r.layer);
        Ok(SubstrateState::new(results))
    }

    pub fn idle_status(&self) -> Vec<LayerResult> {
        self.config.layers
            .iter()
            .map(|(name, cfg)| {
                let layer = match name.as_str() {
                    "quantum"      => Layer::Quantum,
                    "geomagnetic"  => Layer::Geomagnetic,
                    "magnon"       => Layer::Magnon,
                    "quantum_lab"  => Layer::QuantumLab,
                    "solar"        => Layer::Solar,
                    "cosmological" => Layer::Cosmological,
                    "eeg"          => Layer::Eeg,
                    "lunar"        => Layer::Lunar,
                    "radio"        => Layer::Radio,
                    "seismic"      => Layer::Seismic,
                    _ => Layer::Quantum, // Should not happen
                };
                LayerResult::idle(layer, cfg.weight)
            })
            .collect()
    }
}

impl Default for SubstrateEngine {
    fn default() -> Self { Self::new() }
}

async fn drain(set: &mut JoinSet<Result<LayerResult>>) -> Vec<LayerResult> {
    let mut results = Vec::new();
    while let Some(join_res) = set.join_next().await {
        match join_res {
            Ok(Ok(r))  => results.push(r),
            Ok(Err(e)) => tracing::error!("layer error: {e}"),
            Err(e)     => tracing::error!("task panic: {e}"),
        }
    }
    results
}

/// Invokes a layer using the embedded Python FFI.
fn invoke_layer_ffi(
    layer:      Layer,
    weight:     f64,
    params:     std::collections::HashMap<String, serde_json::Value>,
    engine_dir: String,
) -> Result<LayerResult> {
    let module_name = layer.name();

    // Convert HashMap to serde_json::Value Object
    let params_val = serde_json::to_value(params)?;

    match substrate_ffi::call_python_layer(module_name, params_val, &engine_dir) {
        Ok(data) => {
            let score = data["score"].as_f64().unwrap_or(0.0);
            Ok(LayerResult {
                layer,
                status: LayerStatus::Done,
                score,
                weight,
                metadata: data,
            })
        }
        Err(e) => {
            Ok(LayerResult {
                layer,
                status: LayerStatus::Error(e.to_string()),
                score: 0.0,
                weight,
                metadata: serde_json::Value::Null,
            })
        }
    }
}
