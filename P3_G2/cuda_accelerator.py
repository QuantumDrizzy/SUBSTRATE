"""
P3_G2/cuda_accelerator.py
Kernel de C++/CUDA nativo para el Frente 3.
Realiza el cálculo de la plaqueta U(1) y su reducción (Suma)
directamente en la VRAM de la RTX 5060 Ti esquivando el GIL de Python.
"""
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None
    print("[CUDA] Advertencia: CuPy no detectado. El acelerador CUDA no funcionará.")

# ==============================================================================
# EL KERNEL C++ PURO (A compilado en runtime por nvcc)
# ==============================================================================
# Este Kernel calcula cos(P) en cada punto (x,y) de la matriz y
# luego ejecuta una "Block Reduction" usando memoria compartida
# y __syncthreads() para garantizar cero errores de carrera.
# ==============================================================================

cuda_source = r'''
extern "C" __global__
void compute_plaquette_kernel(const float* theta, float* block_sums, int L) {
    // Memoria compartida dinámica para la reducción dentro del bloque de hilos
    extern __shared__ float sdata[];
    
    // Coordenadas absolutas de la malla
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Índice lineal (thread ID) dentro del bloque actual
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    
    float val = 0.0f;
    
    // Evitamos salirnos de los límites de la matriz física
    if (x < L && y < L) {
        // Mapeo del tensor 3D de NumPy/CuPy a 1D contiguo en C++:
        // theta shape es (2, L, L)
        // Canal 0: enlace x (hat_0)
        // Canal 1: enlace y (hat_1)
        
        int offset_c0 = 0 * (L * L);
        int offset_c1 = 1 * (L * L);
        
        int idx_t0_xy = offset_c0 + x * L + y;
        int idx_t1_xy = offset_c1 + x * L + y;
        
        // Condiciones de contorno periódicas (Toroide)
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
    
    // Cargamos el valor a la VRAM compartida del bloque
    sdata[tid] = val;
    __syncthreads(); // BARRERA: Esperamos a que todos los hilos guarden su valor
    
    // REDUCCIÓN PARALELA EN ÁRBOL
    // Sumamos iterativamente las mitades del bloque
    int block_size = blockDim.x * blockDim.y;
    for (unsigned int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads(); // BARRERA: Sincronizamos en cada paso de la reducción
    }
    
    // El hilo maestro de este bloque (tid == 0) escribe la suma total en global memory
    if (tid == 0) {
        int block_id = blockIdx.y * gridDim.x + blockIdx.x;
        block_sums[block_id] = sdata[0];
    }
}
'''

# Compilador JIT de CuPy
if cp is not None:
    plaquette_kernel = cp.RawKernel(cuda_source, 'compute_plaquette_kernel')

def action_cuda(theta_np, beta, L):
    """
    Wrapper Python para invocar el Kernel C++ de la acción de Wilson.
    theta_np: ndarray (2, L, L)
    """
    if cp is None:
        raise RuntimeError("CuPy no está disponible.")
        
    # Copiamos la memoria RAM (Host) a la VRAM (Device)
    theta_gpu = cp.asarray(theta_np, dtype=cp.float32)
    
    # Configuramos la topología de Hilos CUDA
    threads_x = 8
    threads_y = 8
    # Nos aseguramos de que haya bloques suficientes para cubrir todo L
    blocks_x = (L + threads_x - 1) // threads_x
    blocks_y = (L + threads_y - 1) // threads_y
    
    num_blocks = blocks_x * blocks_y
    block_sums_gpu = cp.zeros(num_blocks, dtype=cp.float32)
    
    # 1 float (4 bytes) por hilo en el bloque para shared memory
    shared_mem_bytes = (threads_x * threads_y) * 4
    
    # Ejecutamos el asalto CUDA
    # firma: (grid, block, args, shared_mem)
    plaquette_kernel((blocks_x, blocks_y), (threads_x, threads_y), 
                     (theta_gpu, block_sums_gpu, L), 
                     shared_mem=shared_mem_bytes)
    
    # Suma final de los bloques (suele ser 1 solo bloque para L=8, así que es trivial)
    total_sum = cp.sum(block_sums_gpu)
    
    # Acción = -beta * sum(cos(P))
    action_val = -beta * total_sum
    
    # Traemos el número final de vuelta a la CPU
    return action_val.get()
