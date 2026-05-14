use pyo3::prelude::*;

/// Derive the absolute engine/ directory from the running executable.
///
/// Resolution order:
///   1. `<exe>/../../..` if that directory contains an `engine/` subdir
///      (covers the normal `target/{release,debug}/substrate[.exe]` layout).
///   2. `<cwd>/engine/` as a fallback (running from project root in dev).
///   3. Bare `"engine"` string as last resort (relative, works only from root).
pub fn resolve_engine_dir() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|exe| {
            exe.parent()                    // target/release/
                .and_then(|p| p.parent())   // target/
                .and_then(|p| p.parent())   // project root
                .map(|p| p.join("engine"))
        })
        .filter(|p| p.is_dir())
        .or_else(|| std::env::current_dir().ok().map(|d| d.join("engine")))
        .filter(|p| p.is_dir())
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|| "engine".to_owned())
}

/// Call `engine/<name>_layer.py::run(params)` via embedded Python and return its dict as JSON.
///
/// `engine_dir` must be an **absolute** path to the `engine/` directory.
/// Use [`resolve_engine_dir`] when no explicit path is available.
pub fn call_python_layer(
    name:       &str,
    params:     serde_json::Value,
    engine_dir: &str,
) -> anyhow::Result<serde_json::Value> {
    Python::with_gil(|py| -> PyResult<serde_json::Value> {
        // Insert engine_dir at position 0 only if not already present, to
        // avoid unbounded growth across repeated calls in the same interpreter.
        let sys       = py.import_bound("sys")?;
        let path      = sys.getattr("path")?;
        let path_list: Vec<String> = path.extract()?;
        if !path_list.contains(&engine_dir.to_owned()) {
            path.call_method1("insert", (0, engine_dir))?;
        }

        let module_name = format!("{name}_layer");
        let module      = py.import_bound(module_name.as_str())?;

        let json_mod = py.import_bound("json")?;

        // Convert params to Python dict via JSON round-trip.
        let params_json = serde_json::to_string(&params)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let py_params = json_mod.call_method1("loads", (params_json,))?;

        let result = module.call_method1("run", (py_params,))?;

        let json_str: String = json_mod
            .call_method1("dumps", (result,))?
            .extract()?;

        serde_json::from_str(&json_str)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    })
    .map_err(|e| anyhow::anyhow!("{}", e))
}

/// Python-callable convenience wrapper.
/// Accepts an optional `engine_dir`; falls back to [`resolve_engine_dir`].
#[pyfunction]
#[pyo3(signature = (name, engine_dir = None))]
fn py_call_layer(name: &str, engine_dir: Option<&str>) -> PyResult<String> {
    let dir = engine_dir
        .map(|s| s.to_owned())
        .unwrap_or_else(resolve_engine_dir);
    call_python_layer(name, serde_json::Value::Null, &dir)
        .map(|v| v.to_string())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pymodule]
fn substrate_ffi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_call_layer, m)?)?;
    Ok(())
}
