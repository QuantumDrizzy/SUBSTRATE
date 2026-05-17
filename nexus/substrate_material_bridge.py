"""
nexus/substrate_material_bridge.py — SUBSTRATE → HELIOS material_states bridge
================================================================================
Runs CryptoTN-GPU fast benchmarks (ErCry4a + FMO) and writes quantum material
outputs into HELIOS's material_states SQLite table.

Mapping:
  quantum_efficiency  ← ErCry4a Φ_S(B=0)   — singlet yield at zero field
  electron_mobility   ← ErCry4a ΔΦ_S       — compass sensitivity (earth field)
  lattice_stability   ← 1 - FMO RMSE       — FMO fidelity vs TENSO reference
  temp_celsius        ← last HELIOS telemetry ambient (default 25.0)

Usage (standalone):
  python nexus/substrate_material_bridge.py

Usage (from pipeline.py):
  from nexus.substrate_material_bridge import run_material_bridge
  run_material_bridge()
"""

from __future__ import annotations

import json
import os
import sys
import sqlite3
import time
import logging
from pathlib import Path

import zmq

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [SUBSTRATE.MATERIAL] %(message)s",
)
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).parent                           # SUBSTRATE/nexus/
_ROOT    = _HERE.parent                                    # SUBSTRATE/
_CRYPTOTN = _ROOT / "modules" / "cryptotn_gpu"
_HELIOS_DB = Path(os.environ.get(
    "HELIOS_DB",
    r"C:\Users\Drizzy\Desktop\HELIOS\data\energy_bus.sqlite"
))

# add cryptotn_gpu to path so its imports resolve
if str(_CRYPTOTN) not in sys.path:
    sys.path.insert(0, str(_CRYPTOTN))

# ── benchmark imports (deferred — only fail if module missing) ─────────────────
def _import_benchmarks():
    """Import benchmark runners. Returns (run_ercry4a, run_fmo) or raises."""
    benchmarks = _CRYPTOTN / "benchmarks"
    if str(benchmarks) not in sys.path:
        sys.path.insert(0, str(benchmarks))
    from bench_ercry4a import run_ercry4a   # noqa: PLC0415
    from bench_fmo      import run_fmo       # noqa: PLC0415
    return run_ercry4a, run_fmo


# ── HELIOS DB helpers ──────────────────────────────────────────────────────────
def _get_ambient_temp(conn: sqlite3.Connection) -> float:
    """Read latest ambient proxy from power_telemetry (placeholder: 25.0°C)."""
    row = conn.execute(
        "SELECT voltage FROM power_telemetry ORDER BY id DESC LIMIT 1"
    ).fetchone()
    # No real thermistor yet — return 25.0 as lab ambient
    return 25.0


def _write_material_state(
    conn: sqlite3.Connection,
    quantum_efficiency: float,
    electron_mobility: float,
    lattice_stability: float,
    temp_celsius: float,
) -> int:
    """Insert one row into material_states. Returns new row id."""
    cursor = conn.execute(
        """
        INSERT INTO material_states
            (quantum_efficiency, electron_mobility, lattice_stability, temp_celsius)
        VALUES (?, ?, ?, ?)
        """,
        (
            round(quantum_efficiency, 6),
            round(electron_mobility, 6),
            round(lattice_stability, 6),
            round(temp_celsius, 2),
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ── main bridge logic ──────────────────────────────────────────────────────────
def run_material_bridge(
    n_nuc: int = 10,
    fast: bool = True,
) -> dict | None:
    """
    Run CryptoTN-GPU benchmarks and write results to HELIOS material_states.

    Parameters
    ----------
    n_nuc : int
        Nuclear spins for ErCry4a (10 = fast ExactSolver, ~1s).
    fast  : bool
        If True: ErCry4a n_steps=100, FMO n_steps=100. ~2-3s total.

    Returns
    -------
    dict with keys: quantum_efficiency, electron_mobility, lattice_stability,
                    temp_celsius, row_id, wall_time_s
    """
    if not _HELIOS_DB.exists():
        log.error(f"HELIOS DB not found: {_HELIOS_DB}")
        return None

    log.info(f"HELIOS DB: {_HELIOS_DB}")
    log.info(f"CryptoTN-GPU path: {_CRYPTOTN}")

    try:
        run_ercry4a, run_fmo = _import_benchmarks()
    except ImportError as exc:
        log.error(f"CryptoTN-GPU import failed: {exc}")
        return None

    t0 = time.perf_counter()

    # ── 1. ErCry4a — singlet yield + compass sensitivity ──────────────────────
    n_steps_cry = 100 if fast else 300
    log.info(f"Running ErCry4a ({n_nuc} nuclei, n_steps={n_steps_cry})...")
    try:
        ercry = run_ercry4a(
            n_nuc=n_nuc,
            B_field_values=[0.0, 0.05],    # only need B=0 and earth-field
            t_max_us=5.0,
            n_steps=n_steps_cry,
            chi=32,
            use_mps=False,
        )
        phi_s_zero   = float(ercry["phi_s"][0])       # Φ_S at B=0
        delta_phi_s  = float(ercry["delta_phi_s_earth"])
    except Exception as exc:
        log.error(f"ErCry4a failed: {exc}")
        phi_s_zero  = 0.5      # neutral fallback (max entangled)
        delta_phi_s = 0.0

    # ── 2. FMO 77K — lattice stability via TENSO fidelity ────────────────────
    n_steps_fmo = 100 if fast else 200
    log.info(f"Running FMO 77K (n_steps={n_steps_fmo})...")
    try:
        fmo = run_fmo(T_K=77.0, t_max_fs=500.0, n_steps=n_steps_fmo)
        rmse_tenso = float(fmo["rmse_vs_tenso"])
    except Exception as exc:
        log.error(f"FMO failed: {exc}")
        rmse_tenso = 0.0

    # ── 3. Map to material_states schema ──────────────────────────────────────
    quantum_efficiency = phi_s_zero               # range ~[0.45, 0.55]
    electron_mobility  = delta_phi_s              # range ~[0.001, 0.010]
    lattice_stability  = max(0.0, 1.0 - rmse_tenso * 10.0)  # invert + scale

    # ── 4. Write to HELIOS DB ─────────────────────────────────────────────────
    conn = sqlite3.connect(str(_HELIOS_DB))
    try:
        temp_c = _get_ambient_temp(conn)
        row_id = _write_material_state(
            conn,
            quantum_efficiency=quantum_efficiency,
            electron_mobility=electron_mobility,
            lattice_stability=lattice_stability,
            temp_celsius=temp_c,
        )
    finally:
        conn.close()

    wall = time.perf_counter() - t0

    result = {
        "quantum_efficiency": round(quantum_efficiency, 6),
        "electron_mobility":  round(electron_mobility,  6),
        "lattice_stability":  round(lattice_stability,  6),
        "temp_celsius":       temp_c,
        "row_id":             row_id,
        "wall_time_s":        round(wall, 2),
        "source": {
            "phi_s_zero":    round(phi_s_zero, 6),
            "delta_phi_s":   round(delta_phi_s, 6),
            "rmse_vs_tenso": round(rmse_tenso,  6),
        },
    }

    log.info(
        f"material_states row {row_id} written — "
        f"Φ_S={phi_s_zero:.5f}  ΔΦ_S={delta_phi_s:.5f}  "
        f"RMSE={rmse_tenso:.5f}  wall={wall:.2f}s"
    )

    # ── 5. Publish to nexus.quantum (port 5562) ───────────────────────────────
    try:
        ctx = zmq.Context()
        pub = ctx.socket(zmq.PUB)
        pub.bind("tcp://*:5562")
        time.sleep(0.1)  # warmup
        frame = json.dumps({
            "ts":    time.time(),
            "event": "substrate_result",
            "payload": result,
        })
        pub.send_string(f"nexus.quantum {frame}", flags=zmq.NOBLOCK)
        time.sleep(0.05)  # let subscribers drain
        pub.close()
        ctx.term()
        log.info("[NEXUS.QUANTUM] substrate_result published on :5562")
    except Exception as exc:
        log.warning(f"[NEXUS.QUANTUM] publish failed (non-fatal): {exc}")

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="SUBSTRATE → HELIOS material bridge")
    parser.add_argument("--n-nuc",  type=int,  default=10,   help="nuclear spins (default 10)")
    parser.add_argument("--full",   action="store_true",      help="full benchmark (slower)")
    args = parser.parse_args()

    result = run_material_bridge(n_nuc=args.n_nuc, fast=not args.full)
    if result:
        print(json.dumps(result, indent=2))
    else:
        sys.exit(1)
