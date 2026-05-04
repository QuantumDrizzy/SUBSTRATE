# Plan de Implementación: P1_LHCB (Anomalía Mesón B)

El objetivo es construir el pipeline para el Proyecto 1, aislando eventos anómalos $b \to s\ell\ell$ mediante un Autoencoder GNN clásico y modelando el fondo con una Quantum Boltzmann Machine (QBM).

## Arquitectura Propuesta

### Entorno y Dependencias (`P1_LHCB/requirements.txt`)
Definición del stack de dependencias exacto para GPU local:
- `uproot`, `pandas`, `awkward` (Ingesta)
- `torch`, `torch-geometric` (IA Clásica)
- `pennylane`, `qiskit-aer[gpu]`, `cuquantum` (IA Cuántica)

### Pipeline Core

#### 1. `P1_LHCB/ingestion.py`
Módulo encargado de leer las ntuples del CERN Open Data usando `uproot`. Transformará las variables cinemáticas de muones y kaones en tensores de PyTorch.
**Ajuste Táctico:** Como no disponemos del archivo real `lhcb_data.root` descargado ahora mismo, incluiré un modo *mock* (generación de un DataFrame sintético con la misma estructura cinemática esperada: $q^2$, $p_T$, $\eta$, $\phi$). Esto nos permite probar el pipeline de extremo a extremo hoy mismo.

#### 2. `P1_LHCB/gnn_autoencoder.py`
Implementación del modelo GNN usando `torch_geometric`. 
- **Encoder:** Reduce el grafo del evento (nodos = partículas, aristas = distancias $\Delta R$) a un espacio latente de 12 dimensiones.
- **Decoder:** Reconstruye la cinemática. Calcula el *Mean Squared Error* (MSE) como "Anomaly Score".

#### 3. `P1_LHCB/qbm_pennylane.py`
Implementación de la Quantum Boltzmann Machine en 12 qubits usando `PennyLane`. 
- Entrenará sobre el espacio latente del GNN para generar muestras del fondo del Modelo Estándar usando un ansatz variacional parametrizado.

#### 4. `P1_LHCB/main.py`
Orquestador principal. Ejecutará el pipeline secuencial: 
1. Carga de datos $\to$ 2. Entrenamiento GNN $\to$ 3. Extracción Latente $\to$ 4. Entrenamiento QBM $\to$ 5. Evaluación de Anomalías.
