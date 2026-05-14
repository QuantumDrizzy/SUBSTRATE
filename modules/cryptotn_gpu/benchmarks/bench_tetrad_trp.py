"""
benchmarks/bench_tetrad_trp.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
benchmark 3: Tetrad-Trp superradiance in AtCry1.

comparison target: Babcock et al., JPCB 128, 4035 (2024).
metric: P_S(t) oscillation frequency and decay rate.
        singlet yield Φ_S at earth's field.

the tetrad-Trp system (W308-W369 radical pair) shows
fast quantum beating at early times → sensitive to exchange coupling J.
"""
from __future__ import annotations

import time
import json
import logging
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from cryptotn.radical_pair import tetrad_trp_config
from cryptotn.tdvp import ExactSolver, MpsSolver
from cryptotn.observables import singlet_yield

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# reference from Babcock 2024 (Table 3, approximate)
BABCOCK_PHI_S_EARTH  = 0.518   # Φ_S at B=50 μT
BABCOCK_BEAT_FREQ_MHZ = 0.42   # quantum beating frequency (MHz)


def run_tetrad(
    t_max_us: float = 20.0,
    n_steps: int = 800,
    B_mT: float = 0.05,
    n_nuc_per_radical: int = 3,  # full=8; reduce for CPU (ExactSolver limit ~12 sites)
) -> dict:
    """
    run Tetrad-Trp dynamics and extract quantum beating signature.
    """
    cfg = tetrad_trp_config()
    cfg.B_mT = B_mT
    # trim to affordable size for CPU ExactSolver (max ~12 sites, dim ≤ 4096)
    cfg.radical_1.nuclei = cfg.radical_1.nuclei[:n_nuc_per_radical]
    cfg.radical_2.nuclei = cfg.radical_2.nuclei[:n_nuc_per_radical]

    logger.info(
        f"Tetrad-Trp: {cfg.n_sites} sites ({n_nuc_per_radical}×2 nuc), "
        f"B={B_mT} mT  [full benchmark needs MPS, {n_nuc_per_radical}=3 for CPU]"
    )
    t0_wall = time.perf_counter()

    if cfg.n_sites > 14:
        solver = MpsSolver(cfg, chi=64)
    else:
        solver = ExactSolver(cfg)
    t, P_S, trace = solver.run(t_max_us=t_max_us, n_steps=n_steps)
    wall_time = time.perf_counter() - t0_wall

    phi_s = singlet_yield(t, P_S, trace, cfg.k_S_us)

    # detect quantum beating frequency via FFT
    dt = t[1] - t[0]
    P_S_detrended = P_S - np.mean(P_S)
    spectrum = np.abs(np.fft.rfft(P_S_detrended))
    freqs_MHz = np.fft.rfftfreq(len(P_S), d=dt)

    # find dominant peak
    peaks, props = find_peaks(spectrum, height=spectrum.max() * 0.1)
    beat_freq_MHz = float(freqs_MHz[peaks[spectrum[peaks].argmax()]]) if len(peaks) > 0 else 0.0

    result = {
        "system": "Tetrad_Trp",
        "n_sites": cfg.n_sites,
        "B_mT": B_mT,
        "t_max_us": t_max_us,
        "phi_s": round(float(phi_s), 6),
        "phi_s_babcock_ref": BABCOCK_PHI_S_EARTH,
        "delta_phi_s": round(float(phi_s - BABCOCK_PHI_S_EARTH), 6),
        "beat_freq_MHz": round(beat_freq_MHz, 4),
        "beat_freq_ref_MHz": BABCOCK_BEAT_FREQ_MHZ,
        "wall_time_s": round(wall_time, 3),
    }

    out_file = RESULTS_DIR / "tetrad_trp.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    log_file = RESULTS_DIR / "tetrad_benchmark.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    logger.info(
        f"Tetrad-Trp: Φ_S={phi_s:.5f} (ref {BABCOCK_PHI_S_EARTH}), "
        f"beat={beat_freq_MHz:.4f} MHz (ref {BABCOCK_BEAT_FREQ_MHZ}), "
        f"wall={wall_time:.2f}s"
    )
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("benchmark 3: Tetrad-Trp superradiance (AtCry1)")
    print("=" * 60)
    r = run_tetrad(t_max_us=20.0, n_steps=800)
    print(f"Φ_S = {r['phi_s']:.5f}  (Babcock ref: {BABCOCK_PHI_S_EARTH})")
    print(f"beat frequency = {r['beat_freq_MHz']:.4f} MHz  (ref: {BABCOCK_BEAT_FREQ_MHZ})")
    print(f"wall time: {r['wall_time_s']:.2f}s")
    print("results → benchmarks/results/tetrad_trp.json")
