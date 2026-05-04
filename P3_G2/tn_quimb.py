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

def contract_lattice(L=8, beta=1.0):
    print(f"[Quimb] Construyendo Red Tensorial PEPS para Lattice {L}x{L} (U(1), beta={beta})")
    print(f"[Quimb] Mapeo de variable de grupo continuo a variable discreta (Cutoff de Bessel = 1)...")
    
    # Construcción del tensor
    T_val = u1_character_tensor(beta, cutoff=1)
    
    # Ensamblar PEPS 2D explícitamente con Condiciones de Contorno Periódicas (Toroide)
    tensors = []
    for i in range(L):
        for j in range(L):
            inds = (
                f'v_{i}_{j}',           # Arriba
                f'v_{(i+1)%L}_{j}',     # Abajo
                f'h_{i}_{j}',           # Izquierda
                f'h_{i}_{(j+1)%L}'      # Derecha
            )
            tensors.append(qtn.Tensor(T_val, inds=inds, tags={f'T_{i}_{j}'}))
            
    peps = qtn.TensorNetwork(tensors)
    
    # Contracción aproximada o exacta
    print(f"[Quimb] Iniciando contracción del volumen 2D...")
    try:
        # optimize='auto-hq' busca el camino de contracción óptimo
        Z = peps.contract(optimize='auto-hq')
        log_Z = np.log(Z)
        print(f"[Quimb] Contracción exitosa.")
        print(f"        Log(Z) [Función de Partición] = {log_Z:.6f}")
        print(f"        Energía Libre por plaqueta  = {-log_Z / (L*L):.6f}")
        return Z
    except Exception as e:
        print(f"[Quimb] Error en la contracción: {e}")
        return None
