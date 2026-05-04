"""
P1_LHCB/qbm_pennylane.py
Quantum Boltzmann Machine (QBM) implementada con PennyLane.
Diseñada para modelado generativo del espacio latente de la GNN.
Aceleración GPU vía qiskit.aer y cuQuantum, con fallback automático a CPU.
"""

import pennylane as qml
from pennylane import numpy as np # IMPORTANTE: numpy de pennylane es diferenciable
import torch
import warnings

# Ocultar warnings menores de integración
warnings.filterwarnings("ignore", category=UserWarning)

class QuantumBoltzmannMachine:
    def __init__(self, num_qubits=8, layers=2):
        self.num_qubits = num_qubits
        self.layers = layers
        
        print("[QBM] Inicializando Quantum Boltzmann Machine...")
        try:
            # 1. Intento de inicializar el backend acelerado por GPU
            self.dev = qml.device('qiskit.aer', wires=num_qubits, gpu=True)
            print("[QBM] ÉXITO: Backend qiskit.aer con aceleración GPU (cuQuantum) enlazado.")
        except Exception as e:
            # Fallback robusto a simulación clásica pura si falla el stack NVIDIA/qiskit
            print(f"[QBM] AVISO: GPU fallback a CPU. Usando default.qubit. Razón: {str(e)[:100]}...")
            self.dev = qml.device('default.qubit', wires=num_qubits)

        # 2. Definición del Circuito Cuántico (QNode)
        # Usamos la interfaz torch y permitimos backpropagation exacta (parameter-shift u optimizada)
        @qml.qnode(self.dev, interface='torch', diff_method='best')
        def qnode(inputs, weights):
            
            # A. Encoding de datos clásicos (el vector latente de la GNN)
            # Asumimos que `inputs` está escalado en [-pi, pi]
            for i in range(num_qubits):
                # Usamos inputs[..., i] para soportar batching implícito en PennyLane
                qml.RY(inputs[..., i], wires=i)
                
            # B. Ansatz Variacional: Entrelazamiento y Rotaciones parametriza
            for layer in range(layers):
                for i in range(num_qubits):
                    qml.RY(weights[layer, i, 0], wires=i)
                    qml.RZ(weights[layer, i, 1], wires=i)
                
                # CNOT en cadena lineal para entrelazar el estado
                for i in range(num_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                # Condición de contorno periódica (anillo cerrado)
                qml.CNOT(wires=[num_qubits-1, 0])
                
            # C. Medición: Retornamos el valor esperado de la magnetización en Z
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self.qnode = qnode
        
        # 3. Empaquetado como capa nativa de PyTorch
        # Forma del tensor de pesos: [capas, qubits, 2 (RY, RZ)]
        weight_shapes = {"weights": (layers, num_qubits, 2)}
        
        self.qlayer = qml.qnn.TorchLayer(self.qnode, weight_shapes)

    def forward(self, inputs):
        """
        Pase hacia adelante (Forward pass) híbrido.
        Inputs: Tensor de PyTorch de shape [batch_size, 12] desde la GNN.
        Returns: Tensor de PyTorch de shape [batch_size, 12] con las medidas Z.
        """
        # Asegurar tipo correcto para interoperabilidad PennyLane-Torch
        inputs = inputs.to(torch.float32)
        return self.qlayer(inputs)

if __name__ == "__main__":
    # Test aislado del módulo
    print("\n--- Ejecutando Test Aislado QBM ---")
    qbm = QuantumBoltzmannMachine(num_qubits=12)
    mock_latent_space = torch.rand(2, 12) * np.pi # Batch de 2 eventos
    out = qbm.forward(mock_latent_space)
    print(f"Forward Pass completado. Forma del tensor de salida: {out.shape}")
    print(out)
