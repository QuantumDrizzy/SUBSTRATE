# Laboratorio de Física de Partículas Cuántica/IA

Este repositorio contiene tres proyectos de física de partículas independientes, diseñados para ejecutarse en un clúster local heterogéneo (GPU 16GB, CPU multicore, Simulador cuántico 20-qubits).

## ⚛️ PROYECTO 1: Caza de Anomalías en el Mesón B (LHCb)
**Objetivo:** Detectar señales de nueva física en desintegraciones $b \to s\ell\ell$ usando datos abiertos del CERN.
*   **Pipeline Clásico:** Ingesta con `uproot`, reducción de dimensionalidad con Autoencoder GNN (`PyTorch Geometric`).
*   **Pipeline Cuántico:** Modelado generativo de fondos en el espacio latente mediante Quantum Boltzmann Machine (`PennyLane` + `qiskit.aer` + `cuQuantum`).
*   **Directorio:** `P1_LHCB/`

## 💎 PROYECTO 2: Unfolding Cuántico del Bosón W
**Objetivo:** Validar una arquitectura de unfolding cuántico secuencial para recuperar distribuciones cinemáticas.
*   **Pipeline Clásico:** Generación de *toy data* con pico jacobiano, fits diferenciables con `JAX`.
*   **Pipeline Cuántico:** Unfolding mediante *sliding window* mapeado a QUBO resuelto con `cirq`/`PennyLane`.
*   **Directorio:** `P2_WBOSON/`

## 🌀 PROYECTO 3: Redes Tensoriales para la Polarización del Vacío (g-2)
**Objetivo:** Validar métodos computacionales (Normalizing Flows + Tensor Networks) en teorías gauge U(1) y Z2.
*   **Pipeline IA:** Generación de configuraciones gauge eludiendo el *critical slowing down* vía Equivariant Continuous Normalizing Flows (`PyTorch` + Neural ODEs).
*   **Pipeline Tensor Networks:** Representación del vacío como PEPS 2+1D, contracción aproximada mediante CTMRG o BMPS usando `quimb` sobre GPU (`CuPy`).
*   **Directorio:** `P3_G2/`
