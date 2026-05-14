"""
gnn_prototype.py — CYCLE_DETECT: Temporal Graph Neural Network
Unsupervised anomaly detection over geological proxy time series.

Architecture:
  Sliding window over aligned.parquet → per-window graph:
    Nodes  = proxy channels (up to 4: gisp2, vostok, be10, vadm)
    Edges  = |Pearson r| > threshold (dynamic per window)
    Feats  = z-scored proxy values in window (node feature vector, dim=window_size)
  Encoder: GraphSAGE (mean aggregation) → latent node embeddings
  Decoder: MLP → reconstruct input features
  Loss:    MSE reconstruction error per node
  Anomaly score: mean reconstruction error across all nodes in a window

Events of interest flagged at top-N% anomaly score.

Usage:
  python src/cycle_detect/gnn_prototype.py [--epochs N] [--device cuda]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Reproducibility seed ──────────────────────────────────────────────────────
_SEED = 42
np.random.seed(_SEED)
torch.manual_seed(_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(_SEED)
from torch_geometric.data import Data, Batch
from torch_geometric.nn import SAGEConv

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
PROC = ROOT / "data" / "processed"

PROXY_COLS_NORM = [
    "gisp2_d18o_norm",
    "vostok_dd_norm",
    "grip_be10_norm",
    "sint2000_vadm_norm",
]

PROXY_LABELS = ["GISP2 δ¹⁸O", "Vostok ΔTs", "GRIP Be-10", "Sint-2000 VADM"]
PALETTE = ["#0A84FF", "#30D158", "#FF9F0A", "#FF453A"]

EVENTS_YR = {
    "Younger Dryas": 12_900,
    "8.2 ka event":   8_200,
    "Laschamp":      41_000,
    "Last Glacial Max": 20_000,
}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph(window: np.ndarray, corr_threshold: float = 0.4) -> Data:
    """
    window: (n_nodes, window_size) — each row is a proxy time series in the window.
    Returns a PyG Data object with:
      x          : (n_nodes, window_size)  — node features
      edge_index : (2, n_edges)
      edge_attr  : (n_edges, 1)            — |Pearson r|
    """
    n = window.shape[0]
    x = torch.tensor(window, dtype=torch.float32)

    # Pairwise Pearson correlation (ignore NaN rows)
    src_list, dst_list, w_list = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            xi = window[i]
            xj = window[j]
            mask = np.isfinite(xi) & np.isfinite(xj)
            if mask.sum() < 5:
                continue
            r = np.corrcoef(xi[mask], xj[mask])[0, 1]
            if np.isfinite(r) and abs(r) >= corr_threshold:
                src_list.append(i)
                dst_list.append(j)
                w_list.append(abs(r))

    if src_list:
        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_attr  = torch.tensor(w_list, dtype=torch.float32).unsqueeze(1)
    else:
        # No edges: fully connected fallback (self-loop only for isolated nodes)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class GraphAutoEncoder(nn.Module):
    """
    GraphSAGE encoder → MLP decoder.
    Input/output dim = window_size (feature per node).
    """
    def __init__(self, in_dim: int, hidden: int = 64, latent: int = 32,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.dropout = dropout

        enc_layers = []
        dims = [in_dim] + [hidden] * (n_layers - 1) + [latent]
        for i in range(len(dims) - 1):
            enc_layers.append(SAGEConv(dims[i], dims[i + 1]))
        self.enc_layers = nn.ModuleList(enc_layers)

        # Decoder: latent → in_dim
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.enc_layers):
            x = conv(x, edge_index)
            if i < len(self.enc_layers) - 1:
                x = F.gelu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward(self, data: Data):
        z = self.encode(data.x, data.edge_index)
        x_hat = self.dec(z)
        return x_hat

    def anomaly_score(self, data: Data) -> float:
        """Per-window scalar anomaly score (mean node reconstruction MSE)."""
        self.eval()
        with torch.no_grad():
            x_hat = self.forward(data)
            # Replace NaN features with 0 before loss (masked nodes)
            x_clean = torch.nan_to_num(data.x, nan=0.0)
            x_hat_clean = torch.nan_to_num(x_hat, nan=0.0)
            score = F.mse_loss(x_hat_clean, x_clean).item()
        return score


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------

def make_windows(
    aligned: pd.DataFrame,
    proxy_cols: list[str],
    window_size: int = 50,
    stride: int = 5,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Slide a window over the time axis.
    Returns:
      windows    : list of (n_nodes, window_size) arrays
      center_ages: array of centre ages (years BP) for each window
    """
    data = aligned[proxy_cols].values.T  # (n_nodes, n_times)
    n_t  = data.shape[1]
    ages = aligned["age_bp"].values

    windows, centres = [], []
    for start in range(0, n_t - window_size + 1, stride):
        end = start + window_size
        windows.append(data[:, start:end].copy())
        centres.append(ages[start + window_size // 2])

    return windows, np.array(centres)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(
    model: GraphAutoEncoder,
    graphs: list[Data],
    epochs: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    losses = []
    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        # Mini-batch: process all windows, each as independent graph
        np.random.shuffle(graphs)
        for g in graphs:
            g = g.to(device)
            opt.zero_grad()
            x_hat = model(g)
            x_clean     = torch.nan_to_num(g.x,   nan=0.0)
            x_hat_clean = torch.nan_to_num(x_hat, nan=0.0)
            loss = F.mse_loss(x_hat_clean, x_clean)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        avg = epoch_loss / len(graphs)
        losses.append(avg)
        if ep % 20 == 0 or ep == 1:
            print(f"  epoch {ep:4d}/{epochs}  loss={avg:.6f}  "
                  f"lr={sched.get_last_lr()[0]:.2e}")
    return losses


# ---------------------------------------------------------------------------
# Analysis & plotting
# ---------------------------------------------------------------------------

def compute_scores(
    model: GraphAutoEncoder,
    graphs: list[Data],
    centres: np.ndarray,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    scores = []
    for g in graphs:
        s = model.anomaly_score(g.to(device))
        scores.append(s)
    return pd.DataFrame({"age_bp": centres, "anomaly_score": scores})


def plot_anomaly(
    scores_df: pd.DataFrame,
    threshold: float,
    aligned: pd.DataFrame,
    proxy_cols: list[str],
    out_path: Path,
):
    fig, (ax_main, ax_scores) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
    )
    fig.patch.set_facecolor("#0F172A")

    x_proxy = aligned["age_bp"].values / 1000.0

    # -- Proxy panel
    for col, label, color in zip(proxy_cols, PROXY_LABELS, PALETTE):
        y = aligned[col].values
        ax_main.plot(x_proxy, y, color=color, linewidth=0.8,
                     alpha=0.8, label=label)
    ax_main.set_ylabel("Normalised proxy (z-score)", color="#94A3B8", fontsize=9)
    ax_main.legend(loc="upper right", fontsize=7, framealpha=0.2,
                   labelcolor="#E2E8F0")
    ax_main.set_facecolor("#0F172A")
    ax_main.tick_params(colors="#64748B")
    for sp in ax_main.spines.values():
        sp.set_edgecolor("#1E293B")

    # -- Anomaly score panel
    x_s = scores_df["age_bp"].values / 1000.0
    y_s = scores_df["anomaly_score"].values
    ax_scores.plot(x_s, y_s, color="#A78BFA", linewidth=1.0)
    ax_scores.fill_between(x_s, y_s, alpha=0.25, color="#A78BFA")
    ax_scores.axhline(threshold, color="#FF453A", linewidth=0.8,
                      linestyle="--", label=f"threshold ({threshold:.4f})")
    ax_scores.set_ylabel("Anomaly score", color="#94A3B8", fontsize=9)
    ax_scores.set_xlabel("Age (ka BP)", color="#94A3B8", fontsize=9)
    ax_scores.set_facecolor("#0F172A")
    ax_scores.tick_params(colors="#64748B")
    for sp in ax_scores.spines.values():
        sp.set_edgecolor("#1E293B")
    ax_scores.legend(fontsize=7, framealpha=0.2, labelcolor="#E2E8F0")

    # -- Event markers
    for evt_name, evt_age_yr in EVENTS_YR.items():
        evt_ka = evt_age_yr / 1000.0
        x_arr = x_proxy
        if x_arr.min() <= evt_ka <= x_arr.max():
            for ax in (ax_main, ax_scores):
                ax.axvline(evt_ka, color="#F1FA8C", linewidth=0.7,
                           linestyle=":", alpha=0.75)
            ax_main.text(evt_ka + 0.3,
                         ax_main.get_ylim()[1] * 0.9,
                         evt_name, color="#F1FA8C", fontsize=6,
                         rotation=90, va="top")

    # -- Highlight anomalous windows
    anomalous = scores_df[scores_df["anomaly_score"] > threshold]
    for _, row in anomalous.iterrows():
        for ax in (ax_main, ax_scores):
            ax.axvspan(row["age_bp"] / 1000.0 - 2.5,
                       row["age_bp"] / 1000.0 + 2.5,
                       alpha=0.08, color="#FF453A", linewidth=0)

    ax_main.invert_xaxis()
    fig.suptitle("CYCLE_DETECT — GNN Anomaly Scores",
                 color="#E2E8F0", fontsize=11, fontweight="bold", y=0.998)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CYCLE_DETECT GNN anomaly detector")
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden",         type=int,   default=64)
    parser.add_argument("--latent",         type=int,   default=32)
    parser.add_argument("--layers",         type=int,   default=2)
    parser.add_argument("--window",         type=int,   default=50)
    parser.add_argument("--stride",         type=int,   default=5)
    parser.add_argument("--corr-thresh",    type=float, default=0.4)
    parser.add_argument("--anom-pct",       type=float, default=95.0,
                        help="Percentile threshold for anomaly flagging")
    parser.add_argument("--device",         type=str,   default="auto")
    args = parser.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[device] {device}")
    if device.type == "cuda":
        print(f"         {torch.cuda.get_device_name(0)}")

    # Load aligned data
    aligned_path = PROC / "aligned.parquet"
    if not aligned_path.exists():
        print(f"[ERROR] {aligned_path} not found. Run fetch_data.py first.")
        sys.exit(1)

    aligned = pd.read_parquet(aligned_path)

    # Filter to available proxy columns
    available = [c for c in PROXY_COLS_NORM if c in aligned.columns]
    if len(available) < 2:
        print(f"[ERROR] Need ≥2 proxy columns, found: {available}")
        sys.exit(1)
    print(f"[proxies] {available}")

    labels_avail = [PROXY_LABELS[PROXY_COLS_NORM.index(c)] for c in available]

    # Fill short NaN gaps by linear interpolation, leave long gaps as NaN
    for col in available:
        aligned[col] = aligned[col].interpolate(
            method="linear", limit=5, limit_direction="both"
        )

    print(f"[data] {len(aligned)} time steps | "
          f"age {aligned['age_bp'].min():.0f}–{aligned['age_bp'].max():.0f} BP")

    # Build windows
    print(f"\n[windows] size={args.window}, stride={args.stride} …")
    windows, centres = make_windows(aligned, available, args.window, args.stride)
    print(f"  {len(windows)} windows generated")

    # Build graphs
    print(f"[graphs] corr_threshold={args.corr_thresh} …")
    graphs = [build_graph(w, corr_threshold=args.corr_thresh) for w in windows]
    n_edges = [g.edge_index.shape[1] for g in graphs]
    print(f"  edges/window: mean={np.mean(n_edges):.1f}, "
          f"min={min(n_edges)}, max={max(n_edges)}")

    # Model
    model = GraphAutoEncoder(
        in_dim=args.window,
        hidden=args.hidden,
        latent=args.latent,
        n_layers=args.layers,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[model] GraphAutoEncoder — {n_params:,} params")

    # Train
    print(f"[train] {args.epochs} epochs …")
    losses = train(model, graphs, args.epochs, args.lr, device)

    # Save model
    ckpt_path = PROC / "gnn_autoencoder.pt"
    torch.save({
        "model_state": model.state_dict(),
        "args": vars(args),
        "proxy_cols": available,
    }, ckpt_path)
    print(f"[saved] checkpoint → {ckpt_path}")

    # Anomaly scores
    print("\n[score] Computing anomaly scores …")
    scores_df = compute_scores(model, graphs, centres, device)
    scores_path = PROC / "anomaly_scores.parquet"
    scores_df.to_parquet(scores_path, index=False)

    threshold = np.percentile(scores_df["anomaly_score"].values, args.anom_pct)
    anomalous = scores_df[scores_df["anomaly_score"] > threshold]
    print(f"  threshold (p{args.anom_pct:.0f}): {threshold:.6f}")
    print(f"  flagged windows: {len(anomalous)}")

    if not anomalous.empty:
        print("\n[events] Top anomalous periods:")
        top = anomalous.nlargest(10, "anomaly_score")
        for _, row in top.iterrows():
            print(f"  {row['age_bp']:8.0f} BP  score={row['anomaly_score']:.6f}")

    # Export anomaly scores as JSON for downstream modules
    top_all = anomalous.nlargest(len(anomalous), "anomaly_score").reset_index(drop=True)
    anomaly_json = {
        "anomalies": [
            {"age_bp": int(row["age_bp"]), "score": round(float(row["anomaly_score"]), 6), "rank": i + 1}
            for i, (_, row) in enumerate(top_all.iterrows())
        ],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": "GraphAutoEncoder",
        "threshold": round(float(threshold), 6),
    }
    anomaly_json_path = PROC / "anomaly_scores.json"
    with open(anomaly_json_path, "w") as fh:
        json.dump(anomaly_json, fh, indent=2)
    print(f"[saved] anomaly scores JSON → {anomaly_json_path}")

    # Plot
    print("\n[plot] Generating anomaly map …")
    plot_anomaly(
        scores_df, threshold, aligned, available,
        PROC / "anomaly_map.png"
    )

    # Training loss plot
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.patch.set_facecolor("#0F172A")
    ax.plot(losses, color="#0A84FF", linewidth=1.2)
    ax.set_xlabel("Epoch", color="#94A3B8")
    ax.set_ylabel("Reconstruction loss", color="#94A3B8")
    ax.set_facecolor("#0F172A")
    ax.tick_params(colors="#64748B")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1E293B")
    fig.suptitle("CYCLE_DETECT — Training Loss",
                 color="#E2E8F0", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PROC / "training_loss.png", dpi=120,
                facecolor=fig.get_facecolor())
    plt.close(fig)

    print("\n[done] Outputs:")
    for f in ["aligned.parquet", "anomaly_scores.parquet",
              "gnn_autoencoder.pt", "anomaly_map.png", "training_loss.png"]:
        p = PROC / f
        size = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "MISSING"
        print(f"  {f:35s} {size}")


if __name__ == "__main__":
    main()
