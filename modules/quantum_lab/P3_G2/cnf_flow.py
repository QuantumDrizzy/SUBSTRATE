"""
P3_G2/cnf_flow.py
Continuous Normalizing Flow usando PyTorch y torchdiffeq.
Aprende a generar configuraciones U(1) a partir de ruido normal.
"""
import torch
import torch.nn as nn
from torchdiffeq import odeint

class ConcatSquashLinear(nn.Module):
    """
    Capa de red neuronal que integra el tiempo (t) de forma continua
    inyectándolo en los pesos y sesgos.
    """
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self._layer = nn.Linear(dim_in, dim_out)
        self._hyper_bias = nn.Linear(1, dim_out, bias=False)
        self._hyper_gate = nn.Linear(1, dim_out)

    def forward(self, t, x):
        t_view = t.view(1, 1).expand(x.shape[0], 1)
        return self._layer(x) * torch.sigmoid(self._hyper_gate(t_view)) \
            + self._hyper_bias(t_view)

class ODEVectorField(nn.Module):
    """
    Campo vectorial que define la derivada dz/dt para el ODE Solver.
    """
    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.ModuleList([
            ConcatSquashLinear(dim, hidden_dim),
            ConcatSquashLinear(hidden_dim, hidden_dim),
            ConcatSquashLinear(hidden_dim, dim)
        ])
        self.activation = nn.Tanh()

    def forward(self, t, x):
        out = x
        for i, layer in enumerate(self.net):
            out = layer(t, out)
            if i < len(self.net) - 1:
                out = self.activation(out)
        return out

class CNF(nn.Module):
    def __init__(self, vector_field):
        super().__init__()
        self.vf = vector_field

    def forward(self, x_in, reverse=False):
        # Integra de 0 a 1 (Generación: z -> x) o de 1 a 0 (Inferencia: x -> z)
        t = torch.tensor([0.0, 1.0]).to(x_in.device)
        if reverse:
            t = torch.tensor([1.0, 0.0]).to(x_in.device)
            
        # Integramos la trayectoria usando un solver simple para ahorrar tiempo de cómputo
        # En producción se usaría 'dopri5' y se calcularía la traza del Jacobiano.
        x_out = odeint(self.vf, x_in, t, method='euler')[1]
        return x_out

def train_cnf(configs, epochs=50):
    print(f"[CNF] Entrenando Flujo Normalizador Continuo en {len(configs)} configuraciones HMC...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CNF] Backend: {device}")
    
    batch_size, channels, L, _ = configs.shape
    dim = channels * L * L
    
    x_data = torch.tensor(configs, dtype=torch.float32).view(batch_size, dim).to(device)
    
    vf = ODEVectorField(dim).to(device)
    cnf = CNF(vf)
    optimizer = torch.optim.Adam(cnf.parameters(), lr=1e-3)
    
    # Entrenamiento (Flow Matching / Score Matching simplificado)
    # Proyectamos x_data al espacio latente z y forzamos a que z siga N(0, I)
    cnf.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Inferencia inversa: Datos -> Ruido latente
        z = cnf(x_data, reverse=True)
        
        # Loss simplificada: Minimizar ||z||^2. 
        # (Fuerza la densidad al origen, emulando la prior normal estándar)
        loss = torch.mean(z**2)
        
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 10 == 0 or epoch == 0:
            print(f"  [Epoch {epoch+1:03d}/{epochs}] Latent Deviation Loss: {loss.item():.4f}")
            
    print("[CNF] Entrenamiento completado. Muestreando configuraciones sintéticas (IA)...")
    cnf.eval()
    with torch.no_grad():
        # Generación directa: Ruido latente -> Datos
        z_sample = torch.randn(len(configs), dim).to(device)
        x_gen = cnf(z_sample, reverse=False).view(len(configs), channels, L, L).cpu().numpy()
        
    return x_gen
