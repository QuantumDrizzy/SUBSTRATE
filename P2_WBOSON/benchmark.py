"""
P2_WBOSON/benchmark.py
Orquestador Principal.
Ejecuta Unfolding Clásico (Inversión Tikhonov) vs Unfolding Cuántico (QUBO).
Calcula y plotea la recuperación del pico jacobiano del Bosón W.
"""
import numpy as np
import matplotlib.pyplot as plt
from data_generator import get_toy_data
from quantum_unfolder import QuantumUnfolder

def classical_unfolding_dagostini(y, R, iterations=100):
    """
    Unfolding clásico: Método Iterativo Bayesiano de D'Agostini (Richardson-Lucy).
    Es el estándar utilizado en HEP. Iteramos para recuperar el espectro original.
    """
    # Inicialización uniforme (Prior)
    x = np.ones(len(y)) * (np.sum(y) / len(y))
    
    # Pre-calcular eficiencias (probabilidad de que algo en j se mida en cualquier i)
    eff = np.sum(R, axis=0)
    # Evitar división por cero
    eff[eff == 0] = 1.0
    
    for _ in range(iterations):
        # Forward folding (Predicción actual de y)
        y_pred = R @ x
        
        # Evitar división por cero
        y_pred[y_pred == 0] = 1e-9
        
        # Actualización bayesiana
        x = x * (R.T @ (y / y_pred)) / eff
        
    return x

def main():
    print("="*60)
    print(" P2_WBOSON: QUANTUM UNFOLDING BENCHMARK ")
    print("="*60)
    
    # 1. Generación de Toy Data
    n_bins = 20
    print(f"[Benchmark] Generando Espectro Sintético del Bosón W ({n_bins} bins)...")
    masses, truth, measured, R = get_toy_data(n_bins=n_bins)
    
    # 2. Unfolding Clásico (D'Agostini)
    # Calibramos a 100 iteraciones para ser comparable al esfuerzo de Simulated Annealing
    print("[Benchmark] Ejecutando Unfolding Clásico (D'Agostini - 100 iteraciones)...")
    x_cls = classical_unfolding_dagostini(measured, R, iterations=100)
    
    # 3. Unfolding Cuántico (Formulación Delta)
    # 5 bins por ventana * 8 bits = 40 variables por resolución QUBO
    print("[Benchmark] Ejecutando Unfolding Cuántico (Sliding Window QUBO + Delta Formulation)...")
    qunf = QuantumUnfolder(R, measured, x_prior=x_cls, window_size=5, bits_per_bin=8, lmbda=0.1)
    x_q = qunf.unfold()
    
    # 4. Cálculo de Errores (Fidelidad)
    mse_cls = np.mean((truth - x_cls)**2)
    mse_q = np.mean((truth - x_q)**2)
    
    print("\n" + "-"*40)
    print(" RESULTADOS FINALES DE FIDELIDAD (MSE vs TRUTH)")
    print("-" * 40)
    print(f" Error Clásico (D'Agostini) : {mse_cls:.2f}")
    print(f" Error Cuántico (QUBO)    : {mse_q:.2f}")
    if mse_q < mse_cls:
        print(" VENTAJA CUANTICA! El QUBO recuperó mejor el gradiente abrupto.")
    else:
        print(" DERROTA TEMPORAL: El método clásico fue más preciso en esta ejecución.")
        
    # 5. Visualización
    print("\n[Benchmark] Generando gráfico comparativo...")
    plt.figure(figsize=(12, 7))
    
    # Dibujar usando step para representar distribuciones de bines en física de partículas
    plt.step(masses, truth, where='mid', label='Truth (Jacobian Peak)', color='black', linewidth=2.5)
    plt.step(masses, measured, where='mid', label='Measured (Smeared + Noise)', color='gray', linestyle='--', alpha=0.7)
    
    plt.plot(masses, x_cls, 'o-', label=f"Classical Unfolded (D'Agostini MSE: {mse_cls:.0f})", color='blue', alpha=0.7)
    plt.plot(masses, x_q, 's-', label=f'Quantum Unfolded (MSE: {mse_q:.0f})', color='red', alpha=0.9, linewidth=2)
    
    plt.axvline(x=80.4, color='green', linestyle=':', label='$M_W$ (80.4 GeV)')
    
    plt.xlabel('W Transverse Mass $M_T$ (GeV)', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    plt.title('Quantum vs Classical Unfolding for W Boson Mass Reconstruction', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    
    filename = "unfolding_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"[DONE] Gráfico guardado en '{filename}'.")
    print("="*60)

if __name__ == "__main__":
    main()
