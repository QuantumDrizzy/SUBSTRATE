#!/usr/bin/env python3
"""
verify_gpu.py — end-to-end GPU execution check for CuTDVPSolver
────────────────────────────────────────────────────────────────
Confirms that CuTDVPSolver runs on GPU and does NOT fall back to
CPU silently.  Prints device, VRAM, and timing at each step.

Usage (WSL2):
    python verify_gpu.py

Exit codes:
    0  GPU OK
    1  GPU not available (cupy missing or no CUDA device)
    2  Runtime error during GPU execution
"""
from __future__ import annotations

import sys
import time
import traceback as _tb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# ── ANSI colours (works in WSL2 terminals) ────────────────────────────────────
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"

PASS = f"{_GREEN}[PASS]{_RESET}"
FAIL = f"{_RED}[FAIL]{_RESET}"
INFO = f"{_CYAN}[INFO]{_RESET}"
WARN = f"{_YELLOW}[WARN]{_RESET}"

def _hr(char="─", width=64):
    print(char * width)

def _section(title: str):
    _hr()
    print(f"  {title}")
    _hr()


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — cupy
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("  verify_gpu.py  —  CuTDVPSolver GPU end-to-end check")
print("=" * 64)
_section("Step 1: cupy import")

try:
    import cupy as cp
    print(f"{PASS}  cupy {cp.__version__} imported successfully")
except ImportError as exc:
    print(f"{FAIL}  cupy not installed: {exc}")
    print(f"       Fix: pip install cupy-cuda12x")
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — CUDA device
# ═════════════════════════════════════════════════════════════════════════════
_section("Step 2: CUDA device detection")

try:
    n_devices = cp.cuda.runtime.getDeviceCount()
    if n_devices == 0:
        print(f"{FAIL}  no CUDA devices found (driver installed?)")
        sys.exit(1)

    cp.cuda.Device(0).use()

    # device name — robust across cupy versions
    try:
        dev_name = cp.cuda.Device(0).name
        if isinstance(dev_name, bytes):
            dev_name = dev_name.decode()
    except Exception:
        try:
            props    = cp.cuda.runtime.getDeviceProperties(0)
            dev_name = props["name"]
            if isinstance(dev_name, bytes):
                dev_name = dev_name.decode()
        except Exception:
            dev_name = "unknown"

    free_mem, total_mem = cp.cuda.runtime.memGetInfo()
    free_gb  = free_mem  / 1024 ** 3
    total_gb = total_mem / 1024 ** 3
    used_gb_before = (total_mem - free_mem) / 1024 ** 3

    print(f"{PASS}  Device 0: {dev_name}")
    print(f"{INFO}  VRAM: {free_gb:.2f} GB free / {total_gb:.2f} GB total "
          f"({used_gb_before:.2f} GB already used)")
    print(f"{INFO}  CUDA devices visible: {n_devices}")

except Exception as exc:
    print(f"{FAIL}  CUDA device error: {exc}")
    _tb.print_exc()
    sys.exit(1)


# ═════════════════════════════════════════════════════════════════════════════
# Step 3 — Build config
# ═════════════════════════════════════════════════════════════════════════════
_section("Step 3: build ErCry4a config (n_nuc=3, chi=8)")

from cryptotn.radical_pair import ercry4a_config
from cryptotn.observables  import singlet_yield

cfg = ercry4a_config(n_nuc=3)
cfg.B_mT = 0.05
print(f"{PASS}  {cfg.name}: {cfg.n_sites} sites, B={cfg.B_mT} mT, "
      f"k_S={cfg.k_S_us} μs⁻¹, k_T={cfg.k_T_us} μs⁻¹")


# ═════════════════════════════════════════════════════════════════════════════
# Step 4 — Instantiate CuTDVPSolver (GPU mode)
# ═════════════════════════════════════════════════════════════════════════════
_section("Step 4: instantiate CuTDVPSolver  (chi=8, krylov_dim=20)")

from cryptotn.cuda.engine import CuTDVPSolver, HAS_CUPY, HAS_CUTN

print(f"{INFO}  HAS_CUPY={HAS_CUPY}   HAS_CUTN={HAS_CUTN}")

try:
    solver = CuTDVPSolver(cfg, chi=8, krylov_dim=20)  # _numpy_mode=False by default
except RuntimeError as exc:
    print(f"{FAIL}  CuTDVPSolver init failed: {exc}")
    sys.exit(1)

xp = solver._xp
if xp is cp:
    print(f"{PASS}  solver._xp = cupy  (GPU confirmed)")
else:
    print(f"{FAIL}  solver._xp = {xp.__name__}  — solver is on CPU, not GPU!")
    print(f"       This should not happen when cupy is available.")
    sys.exit(2)


# ═════════════════════════════════════════════════════════════════════════════
# Step 5 — GPU run
# ═════════════════════════════════════════════════════════════════════════════
_section("Step 5: CuTDVPSolver.run_2site(t_max=2.0 μs, n_steps=20)")

N_STEPS = 20
T_MAX   = 2.0

try:
    # warm-up sync so first step timing is clean
    cp.cuda.Device(0).synchronize()
    t_start = time.perf_counter()

    t_arr, P_S, trace = solver.run_2site(t_max_us=T_MAX, n_steps=N_STEPS)

    cp.cuda.Device(0).synchronize()
    gpu_wall = time.perf_counter() - t_start

except Exception as exc:
    print(f"{FAIL}  GPU run raised an exception:")
    _tb.print_exc()
    sys.exit(2)

print(f"{PASS}  run_2site completed without error")
print(f"{INFO}  wall time : {gpu_wall:.3f} s  "
      f"({gpu_wall / N_STEPS * 1e3:.1f} ms / step)")

phi_s = singlet_yield(t_arr, P_S, trace, cfg.k_S_us)
print(f"{INFO}  Φ_S       = {phi_s:.6f}")
print(f"{INFO}  trace[-1] = {trace[-1]:.4f}  (expected: 0 < x < 1)")

# Verify MPS tensors are cupy arrays (definitive proof of GPU execution)
if solver._mps is None:
    print(f"{WARN}  solver._mps is None — build may not have run")
else:
    first_tensor = solver._mps[0]
    if isinstance(first_tensor, cp.ndarray):
        dev_id = int(first_tensor.device)
        print(f"{PASS}  MPS tensors: cp.ndarray on device {dev_id}  "
              f"(shape {first_tensor.shape})")
    else:
        print(f"{FAIL}  MPS tensors are {type(first_tensor).__name__}, "
              f"not cp.ndarray — running on CPU!")
        sys.exit(2)

# Physical sanity checks
if not (0.0 < phi_s < 1.0):
    print(f"{FAIL}  Φ_S = {phi_s:.6f} is outside physical range (0, 1)")
    sys.exit(2)
if not (0.0 < trace[-1] < 1.0):
    print(f"{FAIL}  trace[-1] = {trace[-1]:.4f} is outside (0, 1)")
    sys.exit(2)

print(f"{PASS}  physical sanity checks passed")


# ═════════════════════════════════════════════════════════════════════════════
# Step 6 — VRAM after run
# ═════════════════════════════════════════════════════════════════════════════
_section("Step 6: VRAM after run")

free_after, total_after = cp.cuda.runtime.memGetInfo()
used_after_gb  = (total_after - free_after) / 1024 ** 3
delta_gb       = used_after_gb - used_gb_before

print(f"{INFO}  VRAM used  : {used_after_gb:.3f} GB / {total_after/1024**3:.2f} GB total")
print(f"{INFO}  VRAM delta : +{delta_gb:.3f} GB  (allocated by this run)")


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
print()
print("=" * 64)
print("  RESULT SUMMARY")
print("=" * 64)
print(f"  Status      : {PASS}  GPU OK")
print(f"  Device      : 0  ({dev_name})")
print(f"  Backend     : {solver._xp.__name__}")
print(f"  MPS on GPU  : yes  (cp.ndarray, device {int(solver._mps[0].device)})")
print(f"  VRAM used   : {used_after_gb:.3f} GB / {total_gb:.2f} GB")
print(f"  Φ_S (B=50μT): {phi_s:.6f}")
print(f"  Wall time   : {gpu_wall:.3f} s  ({gpu_wall/N_STEPS*1e3:.1f} ms/step)")
print("=" * 64)
print()

sys.exit(0)
