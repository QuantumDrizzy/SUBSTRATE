import matplotlib.pyplot as plt
import numpy as np
from ligo_loader import download_ligo_data
from penrose_collapse import PenroseCollapse

# Light theme, shared across the repository's figures.
BG, TEXT, GRID = "#ffffff", "#1a1d21", "#d8dce0"
ACCENT, PURPLE = "#0b6ea8", "#6a3d9a"

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
        if s > 0.15:  # coupling threshold
            coupling_points.append(1)
        else:
            coupling_points.append(0)
            
    # 2. Plotting
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "grid.color": GRID,
    })
    
    # Subplot 1: LIGO gravitational wave
    plt.subplot(2, 1, 1)
    plt.plot(time, ligo_wave, color=ACCENT, alpha=0.9, label="GW150914 event (LIGO strain)")
    plt.title("GRAVITACHYON — real substrate analysis", fontsize=16, color=TEXT)
    plt.ylabel("Relative amplitude", fontsize=12)
    plt.grid(alpha=0.5)
    plt.legend()
    
    # Subplot 2: stability and retrocausal coupling
    plt.subplot(2, 1, 2)
    plt.fill_between(time, stability, color=PURPLE, alpha=0.35, label="Spacetime stability")
    plt.scatter(time[::50], np.array(coupling_points)[::50] * 0.5, color=TEXT, s=10, label="Tachyonic leakage points")
    
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Coupling index", fontsize=12)
    plt.ylim(0, 1.1)
    plt.grid(alpha=0.5)
    plt.legend()
    
    plt.tight_layout()
    print("[VISUALIZER] Rendering technical dashboard. Close the window to finish.")
    plt.show()

if __name__ == "__main__":
    generate_gravitachyon_viz()
