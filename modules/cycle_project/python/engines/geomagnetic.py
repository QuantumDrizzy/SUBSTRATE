"""
gnn_numpy.py — CYCLE_DETECT: Numpy/sklearn baseline anomaly detector.

Mathematically equivalent to a LINEAR GNN autoencoder:
  - A linear autoencoder reduces to PCA (Baldi & Hornik 1989)
  - Message passing on fully-connected graph = weighted PCA per window
  - Reconstruction error = distance from the principal subspace

This runs CPU-only with no torch dependency.
Use gnn_prototype.py --device cuda for the full nonlinear GraphSAGE version.

Algorithm per sliding window:
  1. Build correlation graph (Pearson |r| matrix → weighted adjacency A)
  2. Graph-diffuse features: X' = (I + D^-1/2 A D^-1/2) X  (one GCN step)
  3. PCA on X' → keep k latent dims
  4. Reconstruct X'_hat from k dims
  5. Anomaly score = mean MSE across all nodes

Output: anomaly_scores.parquet, anomaly_map.png, training_loss.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"

PROXY_COLS_NORM = [
    "gisp2_d18o_norm",
    "vostok_deuterium_norm",
    "vostok_co2_norm",
    "grip_be10_norm",
    "sint2000_norm",
]
PROXY_LABELS = ["GISP2 δ¹⁸O", "Vostok ΔTs", "Vostok CO₂", "GRIP Be-10", "Sint-2000 VADM"]
PALETTE      = ["#0A84FF", "#30D158", "#A78BFA", "#FF9F0A", "#FF453A"]

EVENTS_YR = {
    "Younger Dryas": 12_900,
    "8.2 ka event":   8_200,
    "Laschamp":      41_000,
    "LGM":           20_000,
}


# ---------------------------------------------------------------------------
# Graph diffusion
# ---------------------------------------------------------------------------

def correlation_adjacency(window: np.ndarray, threshold: float) -> np.ndarray:
    """
    window: (n_nodes, T)
    Returns: (n_nodes, n_nodes) weighted adjacency (|Pearson r|, zeroed below threshold)
    """
    n = window.shape[0]
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi, xj = window[i], window[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 5:
                continue
            r = np.corrcoef(xi[mask], xj[mask])[0, 1]
            if np.isfinite(r) and abs(r) >= threshold:
                A[i, j] = abs(r)
    return A


def graph_diffuse(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    One step of symmetric normalised graph diffusion:
      X' = (I + D^{-1/2} A D^{-1/2}) X
    X: (n_nodes, T)
    """
    D_diag = A.sum(axis=1)
    D_inv_sqrt = np.where(D_diag > 0, 1.0 / np.sqrt(D_diag + 1e-9), 0.0)
    D_mat = np.diag(D_inv_sqrt)
    A_norm = D_mat @ A @ D_mat
    propagated = (np.eye(A.shape[0]) + A_norm) @ X
    return propagated


# ---------------------------------------------------------------------------
# Linear autoencoder via PCA (= optimal linear reconstruction)
# ---------------------------------------------------------------------------

def reconstruction_error(X_diffused: np.ndarray, n_components: int) -> float:
    """
    PCA-based reconstruction error: optimal linear autoencoder.
    X_diffused: (n_nodes, T) — treat each node as a sample, T as features.
    Returns mean MSE across all nodes.
    """
    # Replace NaN with 0 (masked nodes contribute 0 to score)
    X_clean = np.nan_to_num(X_diffused, nan=0.0)

    if X_clean.shape[0] <= n_components:
        # Degenerate case: fewer nodes than components
        return 0.0

    k = min(n_components, X_clean.shape[0] - 1, X_clean.shape[1] - 1)
    if k < 1:
        return 0.0

    pca = PCA(n_components=k)
    try:
        z = pca.fit_transform(X_clean)        # (n_nodes, k)
        X_recon = pca.inverse_transform(z)    # (n_nodes, T)
        mse = np.mean((X_clean - X_recon) ** 2)
    except Exception:
        mse = 0.0
    return float(mse)


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(
    aligned: pd.DataFrame,
    proxy_cols: list,
    window_size: int,
    stride: int,
) -> tuple:
    data   = aligned[proxy_cols].values.T   # (n_nodes, n_times)
    ages   = aligned["age_bp"].values
    n_t    = data.shape[1]
    windows, centres = [], []
    for start in range(0, n_t - window_size + 1, stride):
        end = start + window_size
        windows.append(data[:, start:end].copy())
        centres.append(ages[start + window_size // 2])
    return windows, np.array(centres)


# ---------------------------------------------------------------------------
# "Training" = fit global PCA on all windows (one pass, no epochs)
# ---------------------------------------------------------------------------

def fit_and_score(
    windows: list,
    centres: np.ndarray,
    corr_threshold: float,
    n_components: int,
) -> pd.DataFrame:
    scores = []
    for w in windows:
        A  = correlation_adjacency(w, corr_threshold)
        Xd = graph_diffuse(w, A)
        s  = reconstruction_error(Xd, n_components)
        scores.append(s)

    return pd.DataFrame({"age_bp": centres, "anomaly_score": scores})


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_anomaly(scores_df, threshold, aligned, proxy_cols, out_path):
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.04}
    )
    fig.patch.set_facecolor("#0F172A")

    x_p = aligned["age_bp"].values / 1000.0
    for col, lbl, clr in zip(proxy_cols, PROXY_LABELS, PALETTE):
        ax_top.plot(x_p, aligned[col].values, color=clr, lw=0.8,
                    alpha=0.85, label=lbl)
    ax_top.set_ylabel("z-score", color="#94A3B8", fontsize=9)
    ax_top.legend(loc="upper right", fontsize=7, framealpha=0.15,
                  labelcolor="#E2E8F0")
    ax_top.set_facecolor("#0F172A")
    ax_top.tick_params(colors="#64748B")
    for sp in ax_top.spines.values(): sp.set_edgecolor("#1E293B")
    ax_top.axhline(0, color="#334155", lw=0.5, ls="--")

    x_s = scores_df["age_bp"].values / 1000.0
    y_s = scores_df["anomaly_score"].values
    ax_bot.plot(x_s, y_s, color="#A78BFA", lw=1.2)
    ax_bot.fill_between(x_s, y_s, alpha=0.25, color="#A78BFA")
    ax_bot.axhline(threshold, color="#FF453A", lw=0.9, ls="--",
                   label=f"p95 threshold ({threshold:.5f})")
    ax_bot.set_ylabel("Anomaly score\n(recon. error)", color="#94A3B8", fontsize=8)
    ax_bot.set_xlabel("Age (ka BP)", color="#94A3B8", fontsize=9)
    ax_bot.set_facecolor("#0F172A")
    ax_bot.tick_params(colors="#64748B")
    for sp in ax_bot.spines.values(): sp.set_edgecolor("#1E293B")
    ax_bot.legend(fontsize=7, framealpha=0.15, labelcolor="#E2E8F0")

    for evt_name, evt_yr in EVENTS_YR.items():
        evt_ka = evt_yr / 1000.0
        x_all = x_p
        if x_all.min() <= evt_ka <= x_all.max():
            for ax in (ax_top, ax_bot):
                ax.axvline(evt_ka, color="#F1FA8C", lw=0.7, ls=":", alpha=0.8)
            ax_top.text(evt_ka + 0.3,
                        np.nanmax(aligned[proxy_cols[0]].values) * 0.85,
                        evt_name, color="#F1FA8C", fontsize=6,
                        rotation=90, va="top")

    # Shade anomalous windows
    anomalous = scores_df[scores_df["anomaly_score"] > threshold]
    for _, row in anomalous.iterrows():
        ka = row["age_bp"] / 1000.0
        for ax in (ax_top, ax_bot):
            ax.axvspan(ka - 2.5, ka + 2.5, alpha=0.10, color="#FF453A", lw=0)

    ax_top.invert_xaxis()
    fig.suptitle(
        "CYCLE_DETECT — Graph-Diffused PCA Anomaly Map  [numpy baseline]",
        color="#E2E8F0", fontsize=10, fontweight="bold", y=0.999
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] → {out_path}")


def plot_score_dist(scores_df, threshold, out_path):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")
    s = scores_df["anomaly_score"].values
    ax.hist(s, bins=40, color="#A78BFA", alpha=0.7, edgecolor="#1E293B")
    ax.axvline(threshold, color="#FF453A", lw=1.2, ls="--",
               label=f"p95 = {threshold:.5f}")
    ax.set_xlabel("Anomaly score", color="#94A3B8")
    ax.set_ylabel("Count", color="#94A3B8")
    ax.tick_params(colors="#64748B")
    for sp in ax.spines.values(): sp.set_edgecolor("#1E293B")
    ax.legend(fontsize=8, framealpha=0.2, labelcolor="#E2E8F0")
    fig.suptitle("Score distribution", color="#E2E8F0", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window",       type=int,   default=50)
    parser.add_argument("--stride",       type=int,   default=5)
    parser.add_argument("--corr-thresh",  type=float, default=0.35)
    parser.add_argument("--latent",       type=int,   default=2,
                        help="PCA components (latent dim)")
    parser.add_argument("--anom-pct",     type=float, default=95.0)
    args = parser.parse_args()

    # Load
    aligned_path = PROC / "aligned.parquet"
    if not aligned_path.exists():
        print(f"[ERROR] {aligned_path} not found. Run fetch_data.py first.")
        sys.exit(1)

    aligned = pd.read_parquet(aligned_path)
    available = [c for c in PROXY_COLS_NORM if c in aligned.columns]
    if len(available) < 2:
        print(f"[ERROR] Need >= 2 proxies, found: {available}")
        sys.exit(1)

    # Interpolate short gaps
    for col in available:
        aligned[col] = aligned[col].interpolate(
            method="linear", limit=5, limit_direction="both"
        )

    labels_used = [PROXY_LABELS[PROXY_COLS_NORM.index(c)] for c in available]
    print(f"[proxies]  {labels_used}")
    print(f"[data]     {len(aligned)} time steps | "
          f"{aligned['age_bp'].min():.0f}–{aligned['age_bp'].max():.0f} BP")
    print(f"[config]   window={args.window} steps ({args.window*100} yr) | "
          f"stride={args.stride} | corr>{args.corr_thresh} | latent={args.latent}")

    # Windows
    windows, centres = make_windows(aligned, available, args.window, args.stride)
    print(f"[windows]  {len(windows)} windows generated")

    # Score
    print("[scoring]  Graph-diffuse + PCA reconstruction …")
    scores_df = fit_and_score(windows, centres, args.corr_thresh, args.latent)
    scores_path = PROC / "anomaly_scores.parquet"
    scores_df.to_parquet(scores_path, index=False)

    # Threshold + report
    threshold = np.percentile(scores_df["anomaly_score"].values, args.anom_pct)
    anomalous = scores_df[scores_df["anomaly_score"] > threshold]
    print(f"\n[result]   p{args.anom_pct:.0f} threshold = {threshold:.6f}")
    print(f"           flagged windows: {len(anomalous)} "
          f"({100*len(anomalous)/len(scores_df):.1f}%)")

    print("\n[events]   Top-15 anomalous windows:")
    top = scores_df.nlargest(15, "anomaly_score")
    for _, row in top.iterrows():
        age = row["age_bp"]
        score = row["anomaly_score"]
        # Tag known events
        tag = ""
        for evt_name, evt_yr in EVENTS_YR.items():
            if abs(age - evt_yr) <= 3000:
                tag = f"  ← {evt_name}"
                break
        flag = ">>>" if score > threshold else "   "
        print(f"  {flag}  {age:8.0f} BP   score={score:.6f}{tag}")

    # Plots
    print()
    plot_anomaly(scores_df, threshold, aligned, available,
                 PROC / "anomaly_map.png")
    plot_score_dist(scores_df, threshold,
                    PROC / "score_distribution.png")

    print("\n[done] Outputs:")
    for f in ["anomaly_scores.parquet", "anomaly_map.png", "score_distribution.png"]:
        p = PROC / f
        sz = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "MISSING"
        print(f"  {f:35s} {sz}")

    print("\n  → Run the full nonlinear GraphSAGE version on your RTX 5060 Ti:")
    print("    python src/cycle_detect/gnn_prototype.py --epochs 200 --device cuda")
    print()
