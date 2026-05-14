"""
P1_LHCB/gnn_autoencoder.py
Autoencoder basado en PyTorch Geometric para la detección de anomalías en B -> K mu mu.
Restricciones de VRAM: Uso estricto de float32. Grafo totalmente conectado filtrado por Delta R.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import pandas as pd
import numpy as np

class GNNAutoencoder(nn.Module):
    def __init__(self, in_channels=5, hidden_channels=32, latent_dim=8):
        super(GNNAutoencoder, self).__init__()
        # --- ENCODER ---
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.latent_proj = nn.Linear(hidden_channels, latent_dim)
        
        # --- DECODER ---
        # Reconstruye las características de los nodos a partir del vector latente del evento
        self.dec_lin1 = nn.Linear(latent_dim, hidden_channels)
        self.dec_lin2 = nn.Linear(hidden_channels, in_channels)

    def encode(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        # Pooling global: condensa las características de todas las partículas en 1 solo vector por evento
        h_graph = global_mean_pool(x, batch)
        z = self.latent_proj(h_graph)
        return z

    def decode(self, z, batch):
        # Broadcast del vector latente a todos los nodos del evento correspondiente
        z_node = z[batch]
        h = self.dec_lin1(z_node)
        h = F.relu(h)
        x_recon = self.dec_lin2(h)
        return x_recon

    def forward(self, x, edge_index, batch):
        z = self.encode(x, edge_index, batch)
        x_recon = self.decode(z, batch)
        return x_recon, z

def build_graphs_from_df(df: pd.DataFrame, delta_r_threshold: float = 0.5):
    """
    Convierte el DataFrame tabular en una lista de grafos de torch_geometric.
    Nodos: partículas [pT, eta, phi, E, charge].
    Aristas: creadas entre partículas si su distancia Delta R < delta_r_threshold.
    """
    print("[GNN] Construyendo topologías de grafos desde el DataFrame...")
    graphs = []
    grouped = df.groupby('event_id')
    
    for event_id, group in grouped:
        # Extraer características relevantes del nodo
        features = group[['pt', 'eta', 'phi', 'e', 'charge']].values.astype(np.float32)
        
        # Normalización Z-score por evento o global (aquí aplicamos normalización por evento 
        # para estabilizar la red. En un entorno real se usaría StandardScaler global).
        means = np.mean(features, axis=0)
        stds = np.std(features, axis=0) + 1e-6
        features_norm = (features - means) / stds
        
        x = torch.tensor(features_norm, dtype=torch.float32)
        
        num_nodes = len(group)
        edge_indices = []
        
        # Calcular Delta R para construir el grafo (evitamos O(N^2) masivo porque num_nodes es < 10)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    eta1, phi1 = features[i][1], features[i][2]
                    eta2, phi2 = features[j][1], features[j][2]
                    
                    d_eta = eta1 - eta2
                    d_phi = phi1 - phi2
                    # Envolver d_phi en [-pi, pi]
                    d_phi = (d_phi + np.pi) % (2 * np.pi) - np.pi
                    
                    delta_r = np.sqrt(d_eta**2 + d_phi**2)
                    
                    if delta_r < delta_r_threshold:
                        edge_indices.append([i, j])
                        
        if len(edge_indices) > 0:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            # Fallback a self-loops si las partículas están muy dispersas (Delta R alto)
            edge_index = torch.empty((2, 0), dtype=torch.long)
            
        # Target (Label de anomalía) para evaluación
        y = torch.tensor([group['is_anomaly'].iloc[0]], dtype=torch.float32)
        
        data = Data(x=x, edge_index=edge_index, y=y)
        graphs.append(data)
        
    return graphs
