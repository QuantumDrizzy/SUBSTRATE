"""
benchmarks/bench_chi_scaling.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBSTRATE: Operation Scaling (Turbo Edition)

Goal: Reach chi=2500 rapidly by performing only a few steps per chi.
This generates the Performance scaling curve (Time vs Chi) and 
the VRAM usage curve for the paper.
"""

import os
import time
import json
import logging
import argparse
from pathlib import Path

import numpy as np

# --- Windows DLL Healing ---
import os
from pathlib import Path
if os.name == 'nt':
    import site
    search_paths = site.getsitepackages()
    if site.getusersitepackages():
        search_paths.append(site.getusersitepackages())
    for sp in search_paths:
        nvidia_path = Path(sp) / "nvidia"
        if nvidia_path.exists():
            subpackages = ["cublas", "cusolver", "cufft", "curand", "cusparse", "cuda_runtime", "cuda_nvrtc", "cudnn", "nvjitlink"]
            for sub in subpackages:
                bin_path = nvidia_path / sub / "bin"
                if bin_path.exists():
                    os.add_dll_directory(str(bin_path))
# ---------------------------

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from cryptotn.radical_pair import ercry4a_config, SystemConfig, RadicalConfig, NuclearSpin
from cryptotn.cuda.engine   import CuTDVPSolver, build_mpo
from cryptotn.observables   import singlet_yield

RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "chi_scaling_turbo.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "bench_scaling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("scaling_turbo")

class ScalingSolver(CuTDVPSolver):
    def _build(self) -> None:
        cfg = self.config
        self._mpo = build_mpo(cfg)
        xp = self._xp
        local_eye = np.array([0.5, 0, 0, 0.5], dtype=complex).reshape(1, 1, 4)
        self._mps = [xp.array(local_eye, dtype=complex) for _ in range(cfg.n_sites)]
        self._Q_S_dense = np.eye(4)

    def _observables(self):
        return 0.25, 1.0

def get_synthetic_40_sites():
    base_cfg = ercry4a_config(n_nuc=30)
    extra_fad = [NuclearSpin(f"ext1_{i}", 0.001) for i in range(3)] 
    extra_trp = [NuclearSpin(f"ext2_{i}", 0.001) for i in range(5)] 
    return SystemConfig(
        name="ErCry4a_Scaling_40",
        radical_1=RadicalConfig(label="FAD", g_factor=2.0033, nuclei=base_cfg.radical_1.nuclei + extra_fad),
        radical_2=RadicalConfig(label="W", g_factor=2.0032, nuclei=base_cfg.radical_2.nuclei + extra_trp),
        k_S_us=0.26, k_T_us=0.26, J_MHz=0.0, B_mT=0.05
    )

def run_turbo_scaling(chi_list, n_steps=10):
    cfg = get_synthetic_40_sites()
    logger.info(f"TURBO SCALING: {cfg.n_sites} sites, {n_steps} steps per point.")
    
    device_id = cp.cuda.Device().id
    gpu_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode()
    logger.info(f"GPU: {gpu_name}")

    for chi in chi_list:
        logger.info(f"--- Benchmarking chi={chi} ---")
        cp.get_default_memory_pool().free_all_blocks()
        
        try:
            solver = ScalingSolver(cfg, chi=chi)
            t0 = time.perf_counter()
            # Only run a few steps to measure performance
            solver.run_2site(t_max_us=1.0, n_steps=n_steps)
            wall = time.perf_counter() - t0
            
            used_gb = cp.get_default_memory_pool().used_bytes() / 1e9
            time_per_step = wall / n_steps
            
            rec = {
                "chi": chi,
                "time_per_step_s": round(time_per_step, 4),
                "vram_used_gb": round(used_gb, 3),
                "est_total_200_steps_hr": round((time_per_step * 200) / 3600, 2)
            }
            
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(rec) + "\n")
            
            logger.info(f"Chi={chi} | Time/Step: {time_per_step:.2f}s | VRAM: {used_gb:.2f}GB | Est 200 steps: {rec['est_total_200_steps_hr']}h")
            
        except Exception as e:
            logger.error(f"Chi={chi} FAILED: {e}")
            if "out of memory" in str(e).lower(): break

if __name__ == "__main__":
    # Full scaling sweep up to 2500
    chi_list = [64, 128, 256, 512, 1024, 1500, 2048, 2500]
    run_turbo_scaling(chi_list, n_steps=10)
