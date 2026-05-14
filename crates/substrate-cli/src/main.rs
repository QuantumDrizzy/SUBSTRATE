use anyhow::Result;
use clap::{Parser, Subcommand};
use substrate_core::{SubstrateEngine, TuiMsg};
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name    = "substrate",
    about   = "SUBSTRATE — unified quantum/geomagnetic/cosmological measurement system",
    version = "0.1.0"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run all (or selected) layers and display the live TUI dashboard
    Run {
        #[arg(long, default_value = "all", help = "Comma-separated layer names or 'all'")]
        layers: String,
    },
    /// Print current layer status (no run)
    Status,
    /// Run all (or selected) layers, write JSON report, and dump TUI frame
    Report {
        #[arg(long, default_value = "all", help = "Comma-separated layer names or 'all'")]
        layers: String,
        #[arg(
            long,
            default_value = "data/processed/substrate_report.json",
            help = "Output path for the JSON report"
        )]
        output: String,
    },
    /// Verify environment (GPU, CUDA, Dependencies)
    Check,
    /// Launch the native SUBSTRATE Workbench (GUI)
    Gui,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    let cli    = Cli::parse();
    let engine = SubstrateEngine::new();

    match cli.command {
        Commands::Run { layers } => {
            eprintln!("SUBSTRATE  running layers: {layers}");

            let (tx, rx) = std::sync::mpsc::channel::<TuiMsg>();
            let handle   = std::thread::spawn(move || {
                let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
                rt.block_on(engine.run_streaming_filtered(tx, &layers))
            });

            substrate_core::tui::Dashboard::run_live(rx)?;

            match handle.join() {
                Ok(Ok(state)) => {
                    persist_to_db(&state);
                    print_summary(&state.results, state.coherence_score);
                }
                Ok(Err(e)) => eprintln!("Engine error: {e}"),
                Err(_)     => eprintln!("Computation thread panicked"),
            }
        }

        Commands::Status => {
            for r in engine.idle_status() {
                println!("{:12} {:?}", r.layer.name(), r.status);
            }
        }

        Commands::Report { output, layers } => {
            eprintln!("SUBSTRATE  running layers for report: {layers}");
            let (tx, _rx) = std::sync::mpsc::channel::<TuiMsg>();
            let state = engine.run_streaming_filtered(tx, &layers).await?;

            // JSON report
            let json     = serde_json::to_string_pretty(&state)?;
            let out_path = std::path::Path::new(&output);
            std::fs::create_dir_all(
                out_path.parent().unwrap_or_else(|| std::path::Path::new(".")),
            )?;
            std::fs::write(&output, json)?;
            eprintln!("  JSON → {output}");

            // TUI frame dump
            let frame_path = out_path
                .parent()
                .unwrap_or_else(|| std::path::Path::new("."))
                .join("tui_frame.txt");
            match substrate_core::tui::Dashboard::dump_frame(&state, &frame_path) {
                Ok(()) => eprintln!("  TUI frame → {}", frame_path.display()),
                Err(e) => eprintln!("  TUI frame error: {e}"),
            }

            persist_to_db(&state);
            print_summary(&state.results, state.coherence_score);
        }

        Commands::Check => {
            println!("SUBSTRATE Environment Check");
            println!("═══════════════════════════");
            
            // 1. Rust Binary
            println!("Rust Core:    OK (v0.1.0)");
            
            // 2. & 3. Python Bridge & GPU
            let engine_dir = substrate_ffi::resolve_engine_dir();
            match substrate_ffi::call_python_layer("status_check", serde_json::Value::Null, &engine_dir) {
                Ok(res) => {
                    println!("Python FFI:   OK");
                    if let Some(data) = res.get("data") {
                        println!("\n--- Diagnostic Details ---");
                        println!("CUDA Support: {}", data.get("cuda_available").and_then(|v| v.as_bool()).unwrap_or(false));
                        if let Some(gpu) = data.get("gpu_name") {
                            println!("Active GPU:   {}", gpu.as_str().unwrap_or("Unknown"));
                        }
                        println!("CuPy Status:  {}", if data.get("cupy_ok").and_then(|v| v.as_bool()).unwrap_or(false) { "READY" } else { "NOT FOUND" });
                        println!("Quimb Status: {}", if data.get("quimb_ok").and_then(|v| v.as_bool()).unwrap_or(false) { "READY" } else { "NOT FOUND" });
                    }
                }
                Err(e) => println!("Python FFI:   FAILED ({e})"),
            }
            
            println!("\nCheck Complete.");
        }

        Commands::Gui => {
            println!("Launching SUBSTRATE Native Workbench — iNFAMØUS OS");
            let options = eframe::NativeOptions {
                viewport: eframe::egui::ViewportBuilder::default()
                    .with_inner_size([1280.0, 800.0])
                    .with_min_inner_size([900.0, 600.0])
                    .with_title("SUBSTRATE — iNFAMØUS OS"),
                renderer: eframe::Renderer::Wgpu,
                ..Default::default()
            };

            eframe::run_native(
                "substrate_workbench",
                options,
                Box::new(|cc| {
                    // Apply iNFAMØUS dark theme globally
                    // Theme is fully set in SubstrateGui::new() via set_visuals().

                    Box::new(substrate_gui::SubstrateGui::new(cc))
                }),
            )
            .map_err(|e| anyhow::anyhow!("GUI error: {}", e))?;
        }
    }

    Ok(())
}

/// Write every layer result from this run into data/substrate.db.
fn persist_to_db(state: &substrate_core::SubstrateState) {
    use substrate_db::SubstrateDb;
    if let Err(e) = std::fs::create_dir_all("data") {
        eprintln!("  DB error: cannot create data/ dir: {e}");
        return;
    }
    match SubstrateDb::open("data/substrate.db") {
        Ok(db) => {
            let mut written = 0usize;
            for result in &state.results {
                if db
                    .store_run(result.layer.name(), result.score, &result.metadata)
                    .is_ok()
                {
                    written += 1;
                }
            }
            eprintln!("  DB → data/substrate.db ({written} rows written)");
        }
        Err(e) => eprintln!("  DB error (non-fatal): {e}"),
    }
}

fn print_summary(results: &[substrate_core::LayerResult], coherence: f64) {
    println!();
    println!("╔══════════════════════════════════════════════════════════╗");
    println!("║            SUBSTRATE — Layer Summary                    ║");
    println!("╠══════════════════════════════════════════════════════════╣");
    for r in results {
        let bar  = score_bar_ascii(r.score);
        let flag = if r.score >= 0.7 { "HIGH" } else if r.score >= 0.4 { "MED " } else { "LOW " };
        println!("║  {:12} [{bar}] {:.4}  wt={:.1}  {flag}  ║",
            r.layer.name(), r.score, r.weight);
    }
    println!("╠══════════════════════════════════════════════════════════╣");
    let cbar = score_bar_ascii(coherence);
    println!("║  COHERENCE   [{cbar}] {coherence:.4}              ║");
    println!("╚══════════════════════════════════════════════════════════╝");
}

fn score_bar_ascii(score: f64) -> String {
    let filled = ((score * 16.0).round() as usize).min(16);
    format!("{}{}", "#".repeat(filled), ".".repeat(16 - filled))
}
