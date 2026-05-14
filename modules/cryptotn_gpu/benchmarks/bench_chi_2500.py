"""
benchmarks/bench_chi_2500.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SUBSTRATE: Operation Tensor Sweep (Blackwell Edition)

FIXED:
- Logging initialization order
- Unicode logging error
- OOM on initialization
- Windows DLL healing
"""

import os
import time
import json
import logging
import argparse
from pathlib import Path

# --- Early Logging Setup ---
# We need this BEFORE DLL healing so we can log findings
RESULTS_DIR = Path("benchmarks/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_FILE = RESULTS_DIR / "chi_sweep_2500.jsonl"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(RESULTS_DIR / "bench_chi_2500.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("bench_chi_2500")

# --- Windows DLL Healing ---
if os.name == 'nt':
    logger.info("Initializing Windows DLL healing...")
    cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_PATH_V12_5') or os.environ.get('CUDA_PATH_V12_4')
    if cuda_path:
        cuda_bin = Path(cuda_path) / "bin"
        if cuda_bin.exists():
            logger.info(f"[DLL] Adding CUDA_PATH: {cuda_bin}")
            os.add_dll_directory(str(cuda_bin))
            
    # Check common default locations
    defaults = [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
    ]
    for d in defaults:
        if Path(d).exists():
            logger.info(f"[DLL] Adding default: {d}")
            os.add_dll_directory(d)

    # Check site-packages (pip install nvidia-*)
    try:
        import site
        search_paths = site.getsitepackages()
        if site.getusersitepackages():
            search_paths.append(site.getusersitepackages())
            
        for sp in search_paths:
            nvidia_path = Path(sp) / "nvidia"
            if nvidia_path.exists():
                subpackages = [
                    "cublas", "cusolver", "cufft", "curand", 
                    "cusparse", "cuda_runtime", "cuda_nvrtc", "cudnn",
                    "nvjitlink", "nvfatbin"
                ]
                for sub in subpackages:
                    bin_path = nvidia_path / sub / "bin"
                    if bin_path.exists():
                        logger.info(f"[DLL] Adding site-package: {bin_path}")
                        os.add_dll_directory(str(bin_path))
    except Exception as e:
        logger.warning(f"DLL healing site-package scan failed: {e}")

import numpy as np

# Set path to include parent dir
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from cryptotn.radical_pair import ercry4a_config, SystemConfig, RadicalConfig, NuclearSpin
from cryptotn.cuda.engine   import CuTDVPSolver, build_mpo, _init_boundary_envs
from cryptotn.observables   import singlet_yield

class DirectMPSCuTDVPSolver(CuTDVPSolver):
    """
    Overridden solver that builds the initial MPS directly as a product state
    to avoid OOM with dense rho0 for large N.
    """
    def _build(self) -> None:
        cfg = self.config
        logger.info(f"[TDVP-GPU] DIRECT building {cfg.name} "
                    f"({cfg.n_sites} sites, chi={self.chi})...")
        t0 = time.perf_counter()

        self._mpo = build_mpo(cfg)
        xp = self._xp
        
        # Build a maximally mixed initial MPS: rho = I/2^N
        local_eye = np.array([0.5, 0, 0, 0.5], dtype=complex).reshape(1, 1, 4)
        self._mps = [xp.array(local_eye, dtype=complex) for _ in range(cfg.n_sites)]
        
        self._Q_S_dense = np.eye(4) # Dummy
        
        build_time = time.perf_counter() - t0
        logger.info(f"[TDVP-GPU] direct build: {build_time*1e3:.1f} ms")

    def _observables(self):
        return 0.25, 1.0 # Mock Phi_S and Trace for scaling test

def get_synthetic_40_sites():
    base_cfg = ercry4a_config(n_nuc=30)
    extra_fad = [NuclearSpin(f"ext1_{i}", 0.001) for i in range(3)] 
    extra_trp = [NuclearSpin(f"ext2_{i}", 0.001) for i in range(5)] 
    return SystemConfig(
        name="ErCry4a_Synthetic_40",
        radical_1=RadicalConfig(
            label=base_cfg.radical_1.label, g_factor=base_cfg.radical_1.g_factor,
            nuclei=base_cfg.radical_1.nuclei + extra_fad
        ),
        radical_2=RadicalConfig(
            label=base_cfg.radical_2.label, g_factor=base_cfg.radical_2.g_factor,
            nuclei=base_cfg.radical_2.nuclei + extra_trp
        ),
        k_S_us=base_cfg.k_S_us, k_T_us=base_cfg.k_T_us, J_MHz=base_cfg.J_MHz,
        B_mT=base_cfg.B_mT, description="Synthetic 40-site ErCry4a"
    )

def run_bench(chi_list, n_sites_target=32, t_max=10.0, n_steps=200):
    if not HAS_CUPY:
        logger.error("Cupy not found. GPU benchmark aborted.")
        return

    if n_sites_target == 40:
        cfg = get_synthetic_40_sites()
    else:
        cfg = ercry4a_config(n_nuc=30) 

    logger.info(f"STARTING SWEEP: {cfg.name} ({cfg.n_sites} sites)")
    logger.info(f"Target Chi List: {chi_list}")
    
    device_id = cp.cuda.Device().id
    gpu_name = cp.cuda.runtime.getDeviceProperties(device_id)['name'].decode()
    logger.info(f"GPU: {gpu_name} (ID: {device_id})")

    results = []
    for chi in chi_list:
        logger.info(f"--- Running chi={chi} ---")
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        
        try:
            solver = DirectMPSCuTDVPSolver(cfg, chi=chi)
            t0 = time.perf_counter()
            t, P_S, trace = solver.run_2site(t_max_us=t_max, n_steps=n_steps)
            wall = time.perf_counter() - t0
            
            mempool = cp.get_default_memory_pool()
            used_gb = mempool.used_bytes() / 1e9
            
            rec = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "system": cfg.name, "n_sites": cfg.n_sites, "chi": chi,
                "wall_time_s": round(wall, 2), "vram_used_gb": round(used_gb, 3)
            }
            results.append(rec)
            with open(RESULTS_FILE, "a") as f:
                f.write(json.dumps(rec) + "\n")
            logger.info(f"Chi={chi} DONE. Time={wall:.1f}s, VRAM={used_gb:.2f}GB")
        except Exception as e:
            logger.error(f"Chi={chi} FAILED: {str(e)}")
            if "out of memory" in str(e).lower(): break
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sites", type=int, default=32, choices=[32, 40])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--tmax", type=float, default=10.0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    chi_list = [64, 128, 256] if args.quick else [64, 128, 256, 512, 1024, 1500, 1800, 2048, 2200, 2500]
    run_bench(chi_list, n_sites_target=args.sites, t_max=args.tmax, n_steps=args.steps)
