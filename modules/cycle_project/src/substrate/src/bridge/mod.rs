//! bridge — PyO3 bridge a los engines Python
//!
//! Cada función acepta parámetros básicos y devuelve AnalysisResult.
//! Los engines Python viven en python/engines/.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use anyhow::{Result, Context};
use std::path::Path;

use crate::state::AnalysisResult;

// ── Bootstrap ─────────────────────────────────────────────────────────────────

pub fn bootstrap_python(engines_dir: &Path) -> Result<()> {
    Python::with_gil(|py| {
        let sys = py.import_bound("sys")?;
        let path = sys.getattr("path")?;
        if let Some(parent) = engines_dir.parent().and_then(|p| p.to_str()) {
            path.call_method1("insert", (0, parent))?;
        }
        if let Some(dir) = engines_dir.to_str() {
            path.call_method1("insert", (0, dir))?;
        }
        Ok(())
    }).context("bootstrap_python")
}

// ── Helper ────────────────────────────────────────────────────────────────────

fn call_engine(module: &str, func: &str, kwargs_builder: impl FnOnce(Python, &Bound<PyDict>) -> PyResult<()>) -> Result<AnalysisResult> {
    Python::with_gil(|py| {
        let m = py.import_bound(module)
            .with_context(|| format!("import {module} failed"))?;
        let kw = PyDict::new_bound(py);
        kwargs_builder(py, &kw)?;
        let raw = m.call_method(func, (), Some(&kw))
            .with_context(|| format!("{module}.{func}() failed"))?;
        let json_mod = py.import_bound("json")?;
        let json_str: String = json_mod.call_method1("dumps", (raw,))?.extract()?;
        serde_json::from_str(&json_str).context("deserialize result")
    })
}

// ── API ───────────────────────────────────────────────────────────────────────

pub fn run_data_load(data_dir: &str) -> Result<AnalysisResult> {
    let d = data_dir.to_string();
    call_engine("engines.data", "load_aligned", move |py, kw| {
        kw.set_item("data_dir", d.into_py(py))
    })
}

pub fn run_geomagnetic(data_dir: &str, window: usize, stride: usize, corr_thresh: f64, latent: usize) -> Result<AnalysisResult> {
    let d = data_dir.to_string();
    call_engine("engines.geomagnetic", "run_anomaly_scan", move |py, kw| {
        kw.set_item("data_dir",    d.into_py(py))?;
        kw.set_item("window",      window.into_py(py))?;
        kw.set_item("stride",      stride.into_py(py))?;
        kw.set_item("corr_thresh", corr_thresh.into_py(py))?;
        kw.set_item("latent",      latent.into_py(py))
    })
}

pub fn run_spectral(data_dir: &str) -> Result<AnalysisResult> {
    let d = data_dir.to_string();
    call_engine("engines.spectral", "run_spectral", move |py, kw| {
        kw.set_item("data_dir", d.into_py(py))
    })
}

pub fn run_decay(data_dir: &str) -> Result<AnalysisResult> {
    let d = data_dir.to_string();
    call_engine("engines.decay", "run_decay_model", move |py, kw| {
        kw.set_item("data_dir", d.into_py(py))
    })
}

pub fn run_fingerprint(data_dir: &str) -> Result<AnalysisResult> {
    let d = data_dir.to_string();
    call_engine("engines.fingerprint", "run_fingerprint", move |py, kw| {
        kw.set_item("data_dir", d.into_py(py))
    })
}

pub fn run_forecast(data_dir: &str, n_ensemble: usize) -> Result<AnalysisResult> {
    let d = data_dir.to_string();
    call_engine("engines.forecast", "run_lstm_ensemble", move |py, kw| {
        kw.set_item("data_dir",   d.into_py(py))?;
        kw.set_item("n_ensemble", n_ensemble.into_py(py))
    })
}

pub fn run_radical_pair(b_field_ut: f64, time_us: f64) -> Result<AnalysisResult> {
    call_engine("engines.radical_pair", "run_radical_pair", move |py, kw| {
        kw.set_item("B_field_uT", b_field_ut.into_py(py))?;
        kw.set_item("time_us",    time_us.into_py(py))
    })
}

pub fn run_lindblad(n_qubits: usize, t_max: f64, n_steps: usize) -> Result<AnalysisResult> {
    call_engine("engines.lindblad", "run_lindblad", move |py, kw| {
        kw.set_item("n_qubits", n_qubits.into_py(py))?;
        kw.set_item("t_max",    t_max.into_py(py))?;
        kw.set_item("n_steps",  n_steps.into_py(py))
    })
}

pub fn run_rf_noise(b_field_ut: f64, rf_freq_mhz: f64) -> Result<AnalysisResult> {
    call_engine("engines.rf_noise", "run_rf_noise", move |py, kw| {
        kw.set_item("B_field_uT",  b_field_ut.into_py(py))?;
        kw.set_item("rf_freq_MHz", rf_freq_mhz.into_py(py))
    })
}

pub fn run_lbm(n_steps: usize, use_gpu: bool) -> Result<AnalysisResult> {
    call_engine("engines.lbm", "run_lbm_simulation", move |py, kw| {
        kw.set_item("n_steps",  n_steps.into_py(py))?;
        kw.set_item("use_gpu",  use_gpu.into_py(py))
    })
}

pub fn run_myth_correlate(data_dir: &str, window_kyr: f64) -> Result<AnalysisResult> {
    let d = data_dir.to_string();
    call_engine("engines.myth_correlate", "run_correlation", move |py, kw| {
        kw.set_item("data_dir",   d.into_py(py))?;
        kw.set_item("window_kyr", window_kyr.into_py(py))
    })
}
