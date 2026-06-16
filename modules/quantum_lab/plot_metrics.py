import matplotlib.pyplot as plt
from query import compare_metric
import os
import argparse

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def plot_metric(project, metric_name):
    timestamps, values = compare_metric(project, metric_name)
    if not values:
        print(f"No hay datos para '{metric_name}' en el proyecto '{project}'")
        return
    
    plt.figure(figsize=(10, 5))
    # Usar estilo oscuro tipo dashboard
    plt.style.use('dark_background')
    
    plt.plot(timestamps, values, marker='o', linestyle='-', linewidth=2, markersize=8, color='cyan')
    plt.title(f"[{project}] - Evolución de {metric_name}", fontsize=14, color='white')
    plt.xlabel("Fecha de Ejecución", fontsize=10)
    plt.ylabel(metric_name, fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.2, color='white')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"{project}_{metric_name}.png")
    plt.savefig(output_path, dpi=150, facecolor='black', edgecolor='black')
    print(f"[Plot] Gráfico de telemetría renderizado en {output_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Quantum Lab Metrics")
    parser.add_argument("project", type=str, help="Nombre del proyecto (ej. P3_G2)")
    parser.add_argument("metric", type=str, help="Nombre de la métrica a graficar (ej. diff_cuda)")
    args = parser.parse_args()
    
    plot_metric(args.project, args.metric)
