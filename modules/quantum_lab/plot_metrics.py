import matplotlib.pyplot as plt
from query import compare_metric
import os
import argparse

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Light theme, shared across the repository's figures: these plots end up in
# READMEs and PDFs that are read on white.
BG, TEXT, GRID, ACCENT = "#ffffff", "#1a1d21", "#d8dce0", "#0b6ea8"


def plot_metric(project, metric_name):
    timestamps, values = compare_metric(project, metric_name)
    if not values:
        print(f"No data for '{metric_name}' in project '{project}'")
        return

    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
        "xtick.color": TEXT, "ytick.color": TEXT,
        "text.color": TEXT, "grid.color": GRID,
    })
    plt.figure(figsize=(10, 5))

    plt.plot(timestamps, values, marker='o', linestyle='-', linewidth=2,
             markersize=8, color=ACCENT)
    plt.title(f"[{project}] — {metric_name} over time", fontsize=14, color=TEXT)
    plt.xlabel("Run date", fontsize=10)
    plt.ylabel(metric_name, fontsize=10)
    plt.xticks(rotation=45, ha='right')
    plt.grid(alpha=0.5, color=GRID)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f"{project}_{metric_name}.png")
    plt.savefig(output_path, dpi=150, facecolor=BG, edgecolor=BG)
    print(f"[Plot] Telemetry chart written to {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Quantum Lab metrics")
    parser.add_argument("project", type=str, help="Project name (e.g. P3_G2)")
    parser.add_argument("metric", type=str, help="Metric to plot (e.g. diff_cuda)")
    args = parser.parse_args()

    plot_metric(args.project, args.metric)
