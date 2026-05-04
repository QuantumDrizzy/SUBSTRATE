"""
P1_LHCB/main.py
Orquestador Principal - Caza de Anomalías del Mesón B.
Pipeline: Ingesta -> Conversión a Grafo -> Entrenamiento GNN Autoencoder -> Extracción Latente -> Quantum Boltzmann Machine.
"""

import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import roc_curve, auc

# Módulos del proyecto
from ingestion import load_data
from gnn_autoencoder import build_graphs_from_df, GNNAutoencoder
from qbm_pennylane import QuantumBoltzmannMachine

def plot_anomaly_scores(scores, labels, filename="anomaly_scores.png"):
    """
    Dibuja el histograma de errores de reconstrucción separando Fondo (SM) y Anomalías (BSM).
    """
    plt.figure(figsize=(10, 6))
    
    sm_scores = [s for s, l in zip(scores, labels) if l == 0]
    bsm_scores = [s for s, l in zip(scores, labels) if l == 1]
    
    plt.hist(sm_scores, bins=50, alpha=0.6, label='SM Background', color='blue', density=True)
    if bsm_scores:
        bins_bsm = min(50, max(10, len(bsm_scores)))
        plt.hist(bsm_scores, bins=bins_bsm, alpha=0.6, label='BSM Anomaly', color='red', density=True)
        
    plt.xlabel('Reconstruction MSE (Anomaly Score)')
    plt.ylabel('Density')
    plt.yscale('log') # Escala logarítmica para ver la cola de la anomalía
    plt.title('GNN Autoencoder Anomaly Detection (CERN Open Data)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Main] Histograma guardado en '{filename}'.")

def plot_roc_curve(scores, labels, filename="roc_auc_curve.png"):
    """
    Calcula y dibuja la curva ROC con el AUC para medir la precisión del detector.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Background Acceptance)')
    plt.ylabel('True Positive Rate (Anomaly Detection Efficiency)')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Main] Curva ROC guardada en '{filename}'. AUC: {roc_auc:.3f}")

def main():
    print("="*60)
    print(" PIPELINE DE CAZA DE ANOMALIAS: MESON B (LHCb) ")
    print("="*60)
    
    # 0. Setup Hardware (16GB VRAM constraint managed mostly via batching and float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Setup] Dispositivo de Cómputo IA Clásica: {device}")
    
    # 1. Ingesta (Usando CERN Open Data HZZ como proxy LHCb)
    t0 = time.time()
    df = load_data(mode="real", num_events=2000)
    print(f"[Timing] Ingesta completada en {time.time()-t0:.2f}s")
    
    # 2. Construcción de topologías
    t1 = time.time()
    graphs = build_graphs_from_df(df, delta_r_threshold=0.5)
    print(f"[Timing] Construcción de grafos completada en {time.time()-t1:.2f}s")
    
    # Separación Train/Val/Test
    # Train y Val solo tienen fondo SM para que la GNN aprenda la representación normal
    sm_graphs = [g for g in graphs if g.y.item() == 0.0]
    bsm_graphs = [g for g in graphs if g.y.item() == 1.0]
    
    num_sm = len(sm_graphs)
    train_size = int(0.8 * num_sm)
    
    train_graphs = sm_graphs[:train_size]
    val_graphs = sm_graphs[train_size:]
    test_graphs = val_graphs + bsm_graphs # Mezcla para la inferencia
    
    train_loader = DataLoader(train_graphs, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=128, shuffle=False)
    
    # 3. Inicializar Modelo Clásico
    model = GNNAutoencoder(in_channels=5, hidden_channels=32, latent_dim=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    # --- FASE 1: ENTRENAMIENTO GNN ---
    print("\n" + "-"*40)
    print(" FASE 1: Entrenamiento Autoencoder GNN")
    print("-"*40)
    
    epochs = 100
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            x_recon, _ = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(x_recon, batch.x)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
        avg_train_loss = total_loss / len(train_loader.dataset)
        
        # Validation pass
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                x_recon, _ = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(x_recon, batch.x)
                val_loss += loss.item() * batch.num_graphs
        avg_val_loss = val_loss / len(val_loader.dataset)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d}/{epochs} | Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f}")
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping en epoch {epoch+1}! Mejor Val MSE: {best_val_loss:.6f}")
                break
            
    # --- FASE 2: EVALUACIÓN Y EXTRACCIÓN LATENTE ---
    print("\n" + "-"*40)
    print(" FASE 2: Inferencia y Detección de Anomalías")
    print("-"*40)
    
    model.eval()
    all_scores = []
    all_labels = []
    latent_vectors = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            x_recon, z = model(batch.x, batch.edge_index, batch.batch)
            
            # El error al cuadrado por nodo
            mse_nodes = torch.mean((x_recon - batch.x)**2, dim=1)
            
            # Agrupar el MSE nodo a nivel de evento (grafo) usando la librería estándar de PyTorch
            # Iteramos sobre los grafos del batch para promediar el error
            batch_indices = batch.batch.cpu().numpy()
            mse_nodes_np = mse_nodes.cpu().numpy()
            
            for i in range(batch.num_graphs):
                event_mask = (batch_indices == i)
                event_mse = np.mean(mse_nodes_np[event_mask])
                all_scores.append(event_mse)
                
            all_labels.extend(batch.y.cpu().numpy())
            # Acumulamos el vector latente (shape: [num_graphs_in_batch, 12])
            latent_vectors.append(z.cpu())
            
    latent_tensor = torch.cat(latent_vectors, dim=0)
    
    plot_anomaly_scores(all_scores, all_labels)
    plot_roc_curve(all_scores, all_labels)
    
    # --- FASE 3: PIPELINE CUÁNTICO ---
    print("\n" + "-"*40)
    print(" FASE 3: Modelado Generativo Cuántico (QBM)")
    print("-"*40)
    
    print(f"[Main] Tensor Latente Extraído. Forma: {latent_tensor.shape}")
    
    # Inicializamos la Quantum Boltzmann Machine
    qbm = QuantumBoltzmannMachine(num_qubits=8, layers=2)
    
    # Preparar el espacio latente para el circuito cuántico (Codificación de Ángulo)
    # Tanh lo comprime a [-1, 1], y luego mapeamos a [-pi, pi]
    latent_angles = torch.tanh(latent_tensor) * np.pi
    
    print("[Main] Alimentando la QBM con los primeros 10 eventos latentes...")
    try:
        # Simulamos solo un batch de 10 para validación de compilación del pipeline
        t_q = time.time()
        qbm_output = qbm.forward(latent_angles[:10])
        print(f"[Main] Tiempo Forward Pass Cuántico (10 eventos): {time.time()-t_q:.3f}s")
        print(f"[Main] Expectation values (Z) del circuito:\n{qbm_output.detach().numpy()}")
        print("\n[SUCCESS] El circuito PyTorch -> PennyLane -> cuQuantum compila y ejecuta sin errores.")
    except Exception as e:
        print(f"\n[ERROR] Falló la simulación cuántica: {e}")
        
    print("\n" + "="*60)
    print(" PIPELINE FINALIZADO. REVISA 'anomaly_scores.png'.")
    print("="*60)

if __name__ == "__main__":
    main()
