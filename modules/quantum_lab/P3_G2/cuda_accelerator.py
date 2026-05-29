"""
P3_G2/cuda_accelerator.py
Native C++/CUDA kernel for Front 3.
Computes the U(1) plaquette (Wilson action) and its block reduction (sum)
directly in the VRAM of the RTX 5060 Ti, bypassing the Python GIL.
"""
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
    print("[CUDA] Warning: CuPy not detected. The CUDA accelerator will not run.")

# ==============================================================================
# THE RAW C++ KERNEL (JIT-compiled at runtime by nvcc)
# ==============================================================================
# This kernel computes cos(P) at every site (x, y) of the lattice and then
# performs a block-level reduction using shared memory and __syncthreads()
# to guarantee a race-free sum.
# ==============================================================================

cuda_source = r'''
extern "C" __global__
void compute_plaquette_kernel(const float* theta, float* block_sums, int L) {
    // Dynamic shared memory for the intra-block reduction
    extern __shared__ float sdata[];

    // Absolute lattice coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Linear thread ID within the current block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;

    float val = 0.0f;

    // Stay inside the physical lattice bounds
    if (x < L && y < L) {
        // Map the 3D NumPy/CuPy tensor to contiguous 1D in C++:
        // theta shape is (2, L, L)
        // Channel 0: link in x (mu_0)
        // Channel 1: link in y (mu_1)

        int offset_c0 = 0 * (L * L);
        int offset_c1 = 1 * (L * L);

        int idx_t0_xy = offset_c0 + x * L + y;
        int idx_t1_xy = offset_c1 + x * L + y;

        // Periodic boundary conditions (torus)
        int xp1 = (x + 1) % L;
        int yp1 = (y + 1) % L;

        int idx_t1_xp1_y = offset_c1 + xp1 * L + y;
        int idx_t0_x_yp1 = offset_c0 + x * L + yp1;

        // P(x) = theta_0(x,y) + theta_1(x+1,y) - theta_0(x,y+1) - theta_1(x,y)
        float t0 = theta[idx_t0_xy];
        float t1_shifted = theta[idx_t1_xp1_y];
        float t0_shifted = theta[idx_t0_x_yp1];
        float t1 = theta[idx_t1_xy];

        float P = t0 + t1_shifted - t0_shifted - t1;
        val = cosf(P);
    }

    // Store this thread's value into the block's shared memory
    sdata[tid] = val;
    __syncthreads(); // BARRIER: wait until every thread has written its value

    // PARALLEL TREE REDUCTION
    // Iteratively sum the halves of the block (assumes power-of-two block size)
    int block_size = blockDim.x * blockDim.y;
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // BARRIER: synchronize at every reduction step
    }

    // The block's master thread (tid == 0) writes the partial sum to global memory
    if (tid == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        block_sums[block_id] = sdata[0];
    }
}
'''

# CuPy JIT compiler
if cp is not None:
    plaquette_kernel = cp.RawKernel(cuda_source, 'compute_plaquette_kernel')

def action_cuda(theta_np, beta, L):
    """
    Python wrapper to invoke the C++ kernel for the Wilson action.
    theta_np: ndarray (2, L, L)
    """
    if cp is None:
        raise RuntimeError("CuPy is not available.")

    # Copy host RAM to device VRAM (H2D)
    theta_gpu = cp.asarray(theta_np, dtype=cp.float32)

    # Configure the CUDA thread topology
    threads_x = 8
    threads_y = 8
    # Enough blocks to cover the full lattice L
    blocks_x = (L + threads_x - 1) // threads_x
    blocks_y = (L + threads_y - 1) // threads_y

    num_blocks = blocks_x * blocks_y
    block_sums_gpu = cp.zeros(num_blocks, dtype=cp.float32)

    # 1 float (4 bytes) per thread in the block for shared memory
    shared_mem_bytes = (threads_x * threads_y) * 4

    # Launch the kernel
    # signature: (grid, block, args, shared_mem)
    plaquette_kernel((blocks_x, blocks_y), (threads_x, threads_y),
                     (theta_gpu, block_sums_gpu, L),
                     shared_mem=shared_mem_bytes)

    # Final reduction over the per-block partial sums (often a single block for small L)
    total_sum = cp.sum(block_sums_gpu)

    # Action = -beta * sum(cos(P))
    action_val = -beta * total_sum

    # Bring the final scalar back to the CPU (D2H)
    return action_val.get()
