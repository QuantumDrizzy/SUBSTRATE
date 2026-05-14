// SUBSTRATE — radical-pair quantum coherence kernel
// Ported from modules/cryptotn_gpu CUDA backend (sm_120 Blackwell target)
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

// ─── Hyperfine coupling constants (stub — real values from cryptotn HFC table) ──
__device__ float hfc_coupling(int i, int j) {
    return 0.5f / (float)(i + j + 1);  // synthetic 1/r decay
}

// ─── Density-matrix Larmor precession step ───────────────────────────────────
__global__ void radical_pair_step(
    float* __restrict__ rho_re,
    float* __restrict__ rho_im,
    int   n_spins,
    float dt
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_spins * n_spins) return;

    int i = tid / n_spins;
    int j = tid % n_spins;
    float hfc  = hfc_coupling(i, j);
    float theta = hfc * dt;
    float c = cosf(theta), s = sinf(theta);

    float re_new =  rho_re[tid] * c - rho_im[tid] * s;
    float im_new =  rho_re[tid] * s + rho_im[tid] * c;
    rho_re[tid]  = re_new;
    rho_im[tid]  = im_new;
}

// ─── Singlet-yield accumulation ──────────────────────────────────────────────
__global__ void singlet_yield(
    const float* __restrict__ rho_re,
    float* __restrict__       result,
    int n_spins
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float s = 0.0f;
        for (int k = 0; k < n_spins; ++k)
            s += rho_re[k * n_spins + k];
        *result = s / (float)n_spins;
    }
}

// ─── Host launcher ───────────────────────────────────────────────────────────
extern "C" float launch_radical_pair(int n_spins, float dt, int n_steps) {
    int    n2   = n_spins * n_spins;
    size_t sz   = n2 * sizeof(float);
    float *d_re, *d_im, *d_yield;

    cudaMalloc(&d_re,    sz);
    cudaMalloc(&d_im,    sz);
    cudaMalloc(&d_yield, sizeof(float));
    cudaMemset(d_re,    0, sz);
    cudaMemset(d_im,    0, sz);

    // Initialise to |S⟩ singlet: ρ[0,0] = ρ[n-1,n-1] = 0.5
    float half = 0.5f;
    cudaMemcpy(d_re,                              &half, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_re + (n_spins-1)*n_spins+(n_spins-1), &half, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks  = (n2 + threads - 1) / threads;
    for (int t = 0; t < n_steps; ++t)
        radical_pair_step<<<blocks, threads>>>(d_re, d_im, n_spins, dt);

    cudaDeviceSynchronize();
    singlet_yield<<<1, 1>>>(d_re, d_yield, n_spins);
    cudaDeviceSynchronize();

    float host_yield = 0.0f;
    cudaMemcpy(&host_yield, d_yield, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_re);
    cudaFree(d_im);
    cudaFree(d_yield);
    return host_yield;
}
