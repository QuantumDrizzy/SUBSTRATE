import torch
import torch.nn as nn
import numpy as np

class HolographicMERA(nn.Module):
    """
    Simula una red tensorial MERA (Multi-scale Entanglement Renormalization Ansatz).
    Mapea el bulk gravitatorio (4x4) a un borde holográfico de menor dimensión.
    """
    def __init__(self):
        super(HolographicMERA, self).__init__()
        # Capa 1: Entrelazadores y Disentanglers (Simulados)
        self.bulk_to_mid = nn.Sequential(
            nn.Linear(16, 14),
            nn.Tanh(),
            nn.Linear(14, 12),
            nn.ReLU()
        )
        # Capa 2: Isometrías (Compresión holográfica)
        self.mid_to_boundary = nn.Sequential(
            nn.Linear(12, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.Sigmoid()
        )

    def forward(self, bulk_foam):
        """
        bulk_foam: matriz 4x4 de espuma cuántica.
        Retorna: Vector latente de 8 dimensiones (el borde holográfico).
        """
        # Aplanar la espuma 4x4 -> 16
        x = torch.from_numpy(bulk_foam.flatten()).float()
        
        # Proyectar a través del puente holográfico
        mid = self.bulk_to_mid(x)
        boundary = self.mid_to_boundary(mid)
        
        return boundary

if __name__ == "__main__":
    from quantum_foam import QuantumFoam
    foam = QuantumFoam()
    substrate = foam.generate_fluctuation()
    
    bridge = HolographicMERA()
    boundary_vector = bridge(substrate)
    
    print("[HOLOGRAPHY] Bulk gravitatorio proyectado al borde:")
    print(boundary_vector.detach().numpy())
    print(f"[HOLOGRAPHY] Dimensión del borde: {len(boundary_vector)}")
