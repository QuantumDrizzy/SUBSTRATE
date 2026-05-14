// SUBSTRATE — tensor-network contraction kernel
// Connects to modules/quantum_lab P3_G2 lattice-QCD backend (MPS contraction)
#include <cuda_runtime.h>
#include <math.h>

#define BOND_DIM  32
#define PHYS_DIM   2

// ─── Pairwise MPS tensor contraction: C[i,j] = Σ_{d,k} A[i,d,k] · B[k,d,j] ─
__global__ void contract_mps_pair(
    const float* __restrict__ A,   // [BOND x PHYS x BOND]
    const float* __restrict__ B,   // [BOND x PHYS x BOND]
          float* __restrict__ C,   // [BOND x BOND]
    int bond, int phys
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= bond || col >= bond) return;

    float acc = 0.0f;
    for (int d = 0; d < phys; ++d)
        for (int k = 0; k < bond; ++k)
            acc += A[row * phys * bond + d * bond + k]
                 * B[k   * phys * bond + d * bond + col];
    C[row * bond + col] = acc;
}

// ─── Von-Neumann entanglement entropy from singular values ───────────────────
__global__ void entanglement_entropy(
    const float* __restrict__ sv,   // singular values (length bond)
    float*       __restrict__ out,
    int bond
) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < bond; ++i) {
            float p = sv[i] * sv[i];
            if (p > 1e-12f) s -= p * logf(p);
        }
        *out = s;
    }
}

// ─── Host launcher: contract a random MPS pair and return entropy ─────────────
extern "C" float launch_tensor_contraction(int bond, int phys) {
    int   n2  = bond * bond;
    int   n3  = bond * phys * bond;
    float *dA, *dB, *dC, *dSV, *dEnt;

    cudaMalloc(&dA,   n3 * sizeof(float));
    cudaMalloc(&dB,   n3 * sizeof(float));
    cudaMalloc(&dC,   n2 * sizeof(float));
    cudaMalloc(&dSV,  bond * sizeof(float));
    cudaMalloc(&dEnt, sizeof(float));

    // Fill with synthetic MPS tensors (identity-like for stub)
    cudaMemset(dA, 0, n3 * sizeof(float));
    cudaMemset(dB, 0, n3 * sizeof(float));
    // Set diagonal elements to 1/sqrt(bond) to represent uniform MPS
    float diag_val = 1.0f / sqrtf((float)bond);
    for (int k = 0; k < bond; ++k) {
        int idx = k * phys * bond + 0 * bond + k;
        cudaMemcpy(dA + idx, &diag_val, sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dB + idx, &diag_val, sizeof(float), cudaMemcpyHostToDevice);
        // Singular values = uniform maximally entangled
        cudaMemcpy(dSV + k,  &diag_val, sizeof(float), cudaMemcpyHostToDevice);
    }

    dim3 threads(16, 16);
    dim3 blocks((bond + 15) / 16, (bond + 15) / 16);
    contract_mps_pair<<<blocks, threads>>>(dA, dB, dC, bond, phys);
    cudaDeviceSynchronize();

    entanglement_entropy<<<1, 1>>>(dSV, dEnt, bond);
    cudaDeviceSynchronize();

    float host_ent = 0.0f;
    cudaMemcpy(&host_ent, dEnt, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaFree(dSV); cudaFree(dEnt);
    return host_ent;
}
