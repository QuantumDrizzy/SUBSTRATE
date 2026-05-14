"""
P3_G2/tn_quimb.py
Red Tensorial 2D para U(1) usando la expansión en Caracteres (Funciones de Bessel)
y contracción de Projected Entangled Pair States (PEPS) usando quimb.
"""
import quimb as qu
import quimb.tensor as qtn
import numpy as np
from scipy.special import iv

def u1_character_tensor(beta, cutoff=1):
    """
    Construye el tensor de vértice (Plaqueta) para U(1) en 2D puro.
    Utiliza la expansión en caracteres (Bessel) para discretizar el grupo de Lie continuo.
    """
    dim = 2 * cutoff + 1
    T = np.zeros((dim, dim, dim, dim), dtype=float)
    
    # Restricción topológica de divergencia cero para los modos de Fourier:
    # La suma orientada de los momentos en la plaqueta debe ser cero.
    # n1 + n2 - n3 - n4 = 0
    for i1, n1 in enumerate(range(-cutoff, cutoff + 1)):
        for i2, n2 in enumerate(range(-cutoff, cutoff + 1)):
            for i3, n3 in enumerate(range(-cutoff, cutoff + 1)):
                n4 = n1 + n2 - n3
                if abs(n4) <= cutoff:
                    i4 = n4 + cutoff
                    # Peso estadístico exacto de la plaqueta U(1) discreta
                    weight = iv(n1, beta) * iv(n2, beta) * iv(n3, beta) * iv(n4, beta)
                    T[i1, i2, i3, i4] = weight
                    
    return T

def contract_lattice(L=8, beta=1.0, backend='auto'):
    print(f"[Quimb] Construyendo Red Tensorial PEPS para Lattice {L}x{L} (U(1), beta={beta})")
    
    # Construcción del tensor
    T_val = u1_character_tensor(beta, cutoff=1)
    
    # Seleccionar backend
    actual_backend = 'numpy'
    if backend == 'cupy' or backend == 'auto':
        try:
            import cupy
            actual_backend = 'cupy'
            print(f"[Quimb] GPU Backend Detectado: {cupy.cuda.Device(0).compute_capability}")
        except ImportError:
            if backend == 'cupy':
                print("[Quimb] WARNING: cupy requested but not found. Falling back to numpy.")
            actual_backend = 'numpy'

    # Ensamblar PEPS 2D
    tensors = []
    for i in range(L):
        for j in range(L):
            inds = (f'v_{i}_{j}', f'v_{(i+1)%L}_{j}', f'h_{i}_{j}', f'h_{i}_{(j+1)%L}')
            # Convertir a array del backend seleccionado
            data = T_val
            if actual_backend == 'cupy':
                import cupy
                data = cupy.array(T_val)
            tensors.append(qtn.Tensor(data, inds=inds, tags={f'T_{i}_{j}'}))
            
    peps = qtn.TensorNetwork(tensors)
    
    print(f"[Quimb] Iniciando contracción (Backend: {actual_backend})...")
    try:
        # Contracción usando el backend especificado
        Z = peps.contract(optimize='auto-hq', backend=actual_backend)
        
        # Convertir resultado a float de CPU para evitar problemas de serialización
        if actual_backend == 'cupy':
            Z = float(Z.get())
        else:
            Z = float(Z)
            
        print(f"[Quimb] Contracción exitosa. Z={Z:.6e}")
        return Z
    except Exception as e:
        print(f"[Quimb] Error en la contracción: {e}")
        return None
