/*
 * lbm.cu — SUBSTRATE: Lattice Boltzmann D2Q9 CUDA kernel
 * =========================================================
 * GPU-native LBM para flujo astenosférico 2D.
 * Equivalente CUDA al lbm_core.py (actualmente NumPy/CuPy).
 *
 * Compila:
 *   nvcc -O3 -arch=sm_89 -shared -Xcompiler -fPIC -o lbm_cuda.so lbm.cu
 *
 * Expuesto a Python via ctypes o cffi:
 *   from ctypes import CDLL
 *   lib = CDLL("kernels/lbm_cuda.so")
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ── D2Q9 constantes ──────────────────────────────────────────────────────────

#define Q 9
static __constant__ float W[Q]  = {4.f/9,1.f/9,1.f/9,1.f/9,1.f/9,1.f/36,1.f/36,1.f/36,1.f/36};
static __constant__ int   EX[Q] = {0, 1, 0,-1, 0, 1,-1,-1, 1};
static __constant__ int   EY[Q] = {0, 0, 1, 0,-1, 1, 1,-1,-1};

// ── Kernel: colisión BGK ─────────────────────────────────────────────────────

__global__ void collide_kernel(
    float* __restrict__ f,       // distribuciones (NX*NY*Q)
    float* __restrict__ rho,     // densidades salida (NX*NY)
    float* __restrict__ ux,      // velocidad x (NX*NY)
    float* __restrict__ uy,      // velocidad y (NX*NY)
    int NX, int NY, float omega  // omega = dt/tau
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    int base = (y * NX + x) * Q;

    // Densidad y velocidad macroscópica
    float r = 0.f, vx = 0.f, vy = 0.f;
    for (int k = 0; k < Q; k++) {
        float fk = f[base + k];
        r  += fk;
        vx += EX[k] * fk;
        vy += EY[k] * fk;
    }
    if (r > 1e-9f) { vx /= r; vy /= r; }

    rho[y * NX + x] = r;
    ux [y * NX + x] = vx;
    uy [y * NX + x] = vy;

    // BGK: f_eq y colisión
    float v2 = vx*vx + vy*vy;
    for (int k = 0; k < Q; k++) {
        float eu = EX[k]*vx + EY[k]*vy;
        float feq = W[k] * r * (1.f + 3.f*eu + 4.5f*eu*eu - 1.5f*v2);
        f[base + k] += omega * (feq - f[base + k]);
    }
}

// ── Kernel: streaming ────────────────────────────────────────────────────────

__global__ void stream_kernel(
    const float* __restrict__ f_in,
    float*       __restrict__ f_out,
    int NX, int NY
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= NX || y >= NY) return;

    for (int k = 0; k < Q; k++) {
        int xs = (x - EX[k] + NX) % NX;
        int ys = (y - EY[k] + NY) % NY;
        f_out[(y*NX + x)*Q + k] = f_in[(ys*NX + xs)*Q + k];
    }
}

// ── API C exportada ──────────────────────────────────────────────────────────

extern "C" {

/*
 * lbm_run_steps:
 *   Ejecuta n_steps de LBM D2Q9 en la GPU.
 *   f:     puntero device a distribuciones (NX*NY*Q floats)
 *   rho:   puntero device salida densidad (NX*NY floats)
 *   ux/uy: punteros device velocidad (NX*NY floats cada uno)
 *   Devuelve 0 si OK, -1 si error.
 */
int lbm_run_steps(
    float* f, float* rho, float* ux, float* uy,
    int NX, int NY, float omega, int n_steps
) {
    float* f_tmp;
    size_t sz = (size_t)NX * NY * Q * sizeof(float);
    if (cudaMalloc(&f_tmp, sz) != cudaSuccess) return -1;

    dim3 block(16, 16);
    dim3 grid((NX + 15) / 16, (NY + 15) / 16);

    for (int s = 0; s < n_steps; s++) {
        collide_kernel<<<grid, block>>>(f, rho, ux, uy, NX, NY, omega);
        stream_kernel <<<grid, block>>>(f, f_tmp, NX, NY);
        float* tmp = f; f = f_tmp; f_tmp = tmp;  // swap
    }

    cudaDeviceSynchronize();
    cudaFree(f_tmp);

    cudaError_t err = cudaGetLastError();
    return (err == cudaSuccess) ? 0 : -1;
}

/*
 * lbm_alloc_device: reserva memoria device y copia f_host → device.
 * Devuelve puntero device (o NULL si falla).
 */
float* lbm_alloc_device(const float* f_host, int NX, int NY) {
    size_t sz = (size_t)NX * NY * Q * sizeof(float);
    float* d_f;
    if (cudaMalloc(&d_f, sz) != cudaSuccess) return NULL;
    if (cudaMemcpy(d_f, f_host, sz, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_f);
        return NULL;
    }
    return d_f;
}

void lbm_free_device(float* d_f) { cudaFree(d_f); }

void lbm_copy_to_host(float* d_f, float* h_f, int NX, int NY) {
    cudaMemcpy(h_f, d_f, (size_t)NX * NY * Q * sizeof(float), cudaMemcpyDeviceToHost);
}

} // extern "C"
