import matplotlib.pyplot as plt
import numpy as np
from ligo_loader import download_ligo_data
from penrose_collapse import PenroseCollapse

def generate_gravitachyon_viz():
    print("[VISUALIZER] Procesando sustrato de LIGO para visualización...")
    ligo_wave = download_ligo_data()
    penrose = PenroseCollapse()
    
    if ligo_wave is None:
        print("[VISUALIZER] Error: No hay datos de LIGO.")
        return

    # 1. Preparar datos
    time = np.linspace(0, 0.5, len(ligo_wave))
    stability = [penrose.calculate_decoherence_time(d) for d in ligo_wave]
    
    # Simular puntos de acoplamiento (donde la estabilidad permite la señal)
    coupling_points = []
    for s in stability:
        if s > 0.15: # Umbral de acoplamiento
            coupling_points.append(1)
        else:
            coupling_points.append(0)
            
    # 2. Plotting
    plt.figure(figsize=(12, 8))
    plt.style.use('dark_background')
    
    # Subplot 1: Onda Gravitatoria de LIGO
    plt.subplot(2, 1, 1)
    plt.plot(time, ligo_wave, color='#00ffcc', alpha=0.8, label="Evento GW150914 (LIGO Strain)")
    plt.title("GRAVITACHYON: Análisis de Sustrato Real", fontsize=16, color='cyan')
    plt.ylabel("Amplitud Relativa", fontsize=12)
    plt.grid(alpha=0.1)
    plt.legend()
    
    # Subplot 2: Estabilidad y Acoplamiento Retrocausal
    plt.subplot(2, 1, 2)
    plt.fill_between(time, stability, color='#ff00ff', alpha=0.3, label="Estabilidad Espacio-Temporal")
    plt.scatter(time[::50], np.array(coupling_points)[::50] * 0.5, color='white', s=10, label="Puntos de Fuga Taquiónica")
    
    plt.xlabel("Tiempo (segundos)", fontsize=12)
    plt.ylabel("Índice de Acoplamiento", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(alpha=0.1)
    plt.legend()
    
    plt.tight_layout()
    print("[VISUALIZER] Mostrando Dashboard Técnico. Cierra la ventana para terminar.")
    plt.show()

if __name__ == "__main__":
    generate_gravitachyon_viz()
