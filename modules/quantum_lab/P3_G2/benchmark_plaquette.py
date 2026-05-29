"""
P3_G2/benchmark_plaquette.py
Honest, defensible benchmark of the U(1) plaquette/Wilson action:
    JAX (jit)  vs  hand-written CUDA kernel (CuPy RawKernel)

Fixes every methodological flaw in main.py's quick check:
  - Warms up BOTH sides (XLA compilation / kernel JIT excluded from timing)
  - block_until_ready() on JAX (it is async by default)
  - CUDA events for kernel-only GPU time (CPU perf_counter cannot time a sub-ms kernel)
  - N repetitions -> reports MEDIAN and STDEV, not a single noisy sample
  - Separates kernel-only (compute) from end-to-end (H2D + compute + D2H)
  - Sweeps lattice size L to expose the launch-overhead-bound -> compute-bound crossover

Run:  python benchmark_plaquette.py
"""
import time
import statistics
import numpy as np

import jax
import jax.numpy as jnp

import cupy as cp
from cuda_accelerator import plaquette_kernel
from lattice_hmc import action as action_jax  # @jax.jit  S = -beta * sum(cos(P))

BETA = 1.0
LATTICE_SIZES = [8, 16, 32, 64, 128, 256, 512]
N_ITERS = 1000      # measured iterations
N_WARMUP = 50       # warmup iterations (excluded)


# ----------------------------------------------------------------------------
# CUDA kernel paths, split so we can time compute and transfers separately
# ----------------------------------------------------------------------------
def _launch_kernel(theta_gpu, beta, L, block_sums_gpu, grid, block, shmem):
    """Pure on-device work: kernel launch + reduction. No host transfers."""
    plaquette_kernel(grid, block, (theta_gpu, block_sums_gpu, L), shared_mem=shmem)
    total = cp.sum(block_sums_gpu)
    return -beta * total  # still a device scalar


def cuda_config(L):
    tx, ty = 8, 8
    bx = (L + tx - 1) // tx
    by = (L + ty - 1) // ty
    grid = (bx, by)
    block = (tx, ty)
    shmem = tx * ty * 4
    num_blocks = bx * by
    return grid, block, shmem, num_blocks


# ----------------------------------------------------------------------------
# Timers
# ----------------------------------------------------------------------------
def time_jax(theta_np, L):
    """JAX action, warmed up, fully synchronized. Returns (median_ms, std_ms, value)."""
    theta = jnp.asarray(theta_np)
    # warmup (triggers XLA compile on first call)
    for _ in range(N_WARMUP):
        r = action_jax(theta, BETA)
    r.block_until_ready()

    samples = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        r = action_jax(theta, BETA)
        r.block_until_ready()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), statistics.stdev(samples), float(r)


def time_cuda_kernel_only(theta_np, L):
    """Kernel-only GPU time via CUDA events. theta already on device, no transfers."""
    grid, block, shmem, num_blocks = cuda_config(L)
    theta_gpu = cp.asarray(theta_np, dtype=cp.float32)
    block_sums = cp.zeros(num_blocks, dtype=cp.float32)

    for _ in range(N_WARMUP):
        _launch_kernel(theta_gpu, BETA, L, block_sums, grid, block, shmem)
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    end = cp.cuda.Event()
    samples = []
    val = None
    for _ in range(N_ITERS):
        start.record()
        val = _launch_kernel(theta_gpu, BETA, L, block_sums, grid, block, shmem)
        end.record()
        end.synchronize()
        samples.append(cp.cuda.get_elapsed_time(start, end))  # ms
    return statistics.median(samples), statistics.stdev(samples), float(val.get())


def time_cuda_end_to_end(theta_np, L):
    """Full path: H2D copy + kernel + reduction + D2H copy. What action_cuda() actually does."""
    grid, block, shmem, num_blocks = cuda_config(L)

    def full():
        theta_gpu = cp.asarray(theta_np, dtype=cp.float32)        # H2D
        block_sums = cp.zeros(num_blocks, dtype=cp.float32)
        plaquette_kernel(grid, block, (theta_gpu, block_sums, L), shared_mem=shmem)
        return (-BETA * cp.sum(block_sums)).get()                 # D2H

    for _ in range(N_WARMUP):
        full()
    cp.cuda.Stream.null.synchronize()

    samples = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        full()
        cp.cuda.Stream.null.synchronize()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples), statistics.stdev(samples)


# ----------------------------------------------------------------------------
# Main sweep
# ----------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(42)
    print(f"JAX backend : {jax.devices()[0].platform.upper()}  ({jax.devices()[0]})")
    print(f"CUDA device : {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
    print(f"Iters       : {N_ITERS} measured, {N_WARMUP} warmup\n")

    hdr = (f"{'L':>5} {'cells':>8} | {'JAX ms':>12} | {'CUDA k-only ms':>16} | "
           f"{'CUDA e2e ms':>14} | {'kspeedup':>9} {'e2espeed':>9} | {'max|diff|':>9}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    for L in LATTICE_SIZES:
        theta = rng.uniform(-np.pi, np.pi, (2, L, L)).astype(np.float32)

        jax_med, jax_std, jax_val = time_jax(theta, L)
        k_med, k_std, k_val = time_cuda_kernel_only(theta, L)
        e_med, e_std = time_cuda_end_to_end(theta, L)

        diff = abs(jax_val - k_val)
        k_speedup = jax_med / k_med
        e_speedup = jax_med / e_med

        print(f"{L:>5} {2*L*L:>8} | {jax_med:>8.4f}+/-{jax_std:<3.1f} | "
              f"{k_med:>11.5f}+/-{k_std:<3.2f} | {e_med:>9.4f}+/-{e_std:<3.1f} | "
              f"{k_speedup:>8.1f}x {e_speedup:>8.1f}x | {diff:>9.2e}")

        results.append(dict(L=L, cells=2*L*L, jax_ms=jax_med, cuda_kernel_ms=k_med,
                            cuda_e2e_ms=e_med, kernel_speedup=k_speedup,
                            e2e_speedup=e_speedup, max_abs_diff=diff))

    print("\nNotes:")
    print("  - JAX runs on", jax.devices()[0].platform.upper(),
          "(this build has no GPU backend) -> 'speedup' is CUDA-GPU vs JAX-CPU, not GPU-vs-GPU.")
    print("  - kernel-only = compute on device. e2e = +H2D/D2H transfers (dominates at small L).")
    print("  - At small L the workload is launch-overhead-bound; the crossover is the honest story.")

    import json, os
    out = os.path.join(os.path.dirname(__file__), "benchmark_plaquette_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
