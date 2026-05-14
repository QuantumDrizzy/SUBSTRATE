use anyhow::Result;
use serde::Deserialize;
use std::collections::HashMap;

#[derive(Debug, Deserialize, Clone)]
pub struct SubstrateConfig {
    pub substrate: SubstrateMeta,
    pub layers:    HashMap<String, LayerConfig>,
    pub output:    OutputConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SubstrateMeta {
    pub name:    String,
    pub version: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LayerConfig {
    pub enabled: bool,
    pub weight:  f64,
    #[serde(default)]
    pub params:  HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OutputConfig {
    pub db_path:     String,
    pub report_path: String,
}

impl Default for SubstrateConfig {
    fn default() -> Self {
        let mut layers = HashMap::new();
        layers.insert("quantum".to_string(),      LayerConfig { enabled: true, weight: 1.5, params: HashMap::new() });
        layers.insert("geomagnetic".to_string(),  LayerConfig { enabled: true, weight: 2.0, params: HashMap::new() });
        layers.insert("magnon".to_string(),       LayerConfig { enabled: true, weight: 1.2, params: HashMap::new() });
        layers.insert("quantum_lab".to_string(),  LayerConfig { enabled: true, weight: 1.0, params: HashMap::new() });
        layers.insert("solar".to_string(),        LayerConfig { enabled: true, weight: 1.3, params: HashMap::new() });
        layers.insert("cosmological".to_string(), LayerConfig { enabled: true, weight: 0.8, params: HashMap::new() });
        let mut eeg_params = HashMap::new();
        eeg_params.insert("mode".to_string(), serde_json::Value::String("auto".to_string()));
        layers.insert("eeg".to_string(), LayerConfig { enabled: true, weight: 1.8, params: eeg_params });
        layers.insert("lunar".to_string(),   LayerConfig { enabled: true, weight: 1.4, params: HashMap::new() });
        layers.insert("radio".to_string(),   LayerConfig { enabled: true, weight: 1.6, params: HashMap::new() });
        layers.insert("seismic".to_string(), LayerConfig { enabled: true, weight: 1.1, params: HashMap::new() });

        Self {
            substrate: SubstrateMeta {
                name: "SUBSTRATE".to_string(),
                version: "0.1.0".to_string(),
            },
            layers,
            output: OutputConfig {
                db_path: "data/substrate.db".to_string(),
                report_path: "data/processed/substrate_report.json".to_string(),
            },
        }
    }
}

impl SubstrateConfig {
    /// Search for substrate.toml in the current directory and up to the project root.
    pub fn load() -> Result<Self> {
        // 1. Try current directory and parents
        let mut path = std::env::current_dir()?;
        loop {
            let config_path = path.join("substrate.toml");
            if config_path.exists() {
                let content = std::fs::read_to_string(config_path)?;
                let config: SubstrateConfig = toml::from_str(&content)?;
                return Ok(config);
            }
            if !path.pop() { break; }
        }

        // 2. Try executable directory
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let config_path = exe_dir.join("substrate.toml");
                if config_path.exists() {
                    let content = std::fs::read_to_string(config_path)?;
                    let config: SubstrateConfig = toml::from_str(&content)?;
                    return Ok(config);
                }
                
                // Also try one level up from target/debug/ if we are in dev
                if let Some(target_dir) = exe_dir.parent() {
                     let config_path = target_dir.parent().map(|p| p.join("substrate.toml"));
                     if let Some(cp) = config_path {
                         if cp.exists() {
                             let content = std::fs::read_to_string(cp)?;
                             let config: SubstrateConfig = toml::from_str(&content)?;
                             return Ok(config);
                         }
                     }
                }
            }
        }
        
        // Fallback or error? Let's provide a default if missing, but log a warning.
        Err(anyhow::anyhow!("substrate.toml not found in any parent directories"))
    }

    pub fn get_layer_config(&self, name: &str) -> Option<&LayerConfig> {
        self.layers.get(name)
    }
}
