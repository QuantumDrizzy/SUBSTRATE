# requirements: numpy, pandas, pyarrow, matplotlib, torch

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    torch = None

PROJECT_ROOT = Path(__file__).parents[2]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "aligned.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

DT = 100          # years per sample
SEQ_LEN = 10      # 1000-yr lookback window
HIDDEN = 64
N_LAYERS = 2
DROPOUT = 0.30
N_EPOCHS = 60
BATCH_SIZE = 64
LR = 1e-3

N_ENSEMBLE = 50   # MC-dropout draws
N_STEPS = 50      # 5000-yr / 100-yr = 50 forecast steps


# ── model ────────────────────────────────────────────────────────────────────

if _TORCH_OK:
    class VadmLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=HIDDEN,
                num_layers=N_LAYERS,
                batch_first=True,
                dropout=DROPOUT,
            )
            self.drop = nn.Dropout(DROPOUT)
            self.fc = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            # x: (batch, seq_len, 1)
            out, _ = self.lstm(x)
            out = self.drop(out[:, -1, :])
            return self.fc(out).squeeze(-1)  # (batch,)
else:
    class VadmLSTM:
        pass


# ── dataset helpers ───────────────────────────────────────────────────────────

def _make_windows(series, seq_len):
    """Sliding-window dataset: (x[i:i+seq_len], x[i+seq_len])."""
    xs, ys = [], []
    for i in range(len(series) - seq_len):
        xs.append(series[i : i + seq_len])
        ys.append(series[i + seq_len])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ── training ─────────────────────────────────────────────────────────────────

def _train(model, loader, device, n_epochs, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * len(xb)
        scheduler.step()
        if epoch % 10 == 0 or epoch == 1:
            avg = total_loss / len(loader.dataset)
            print(f"[LSTM] Epoch {epoch:3d}/{n_epochs}  loss={avg:.6f}")


# ── MC-dropout forecast ───────────────────────────────────────────────────────

def _mc_forecast(model, seed_np, n_steps, n_ensemble, device):
    """
    Returns samples: np.ndarray shape (n_ensemble, n_steps).
    seed_np: shape (seq_len,) — the initial 1000-yr window.
    """
    model.train()  # keep dropout active
    samples = []
    seed_t = torch.tensor(seed_np, dtype=torch.float32, device=device).reshape(1, SEQ_LEN, 1)

    with torch.no_grad():
        for _ in range(n_ensemble):
            x = seed_t.clone()
            preds = []
            for _ in range(n_steps):
                y = model(x)          # (1,)
                preds.append(y.item())
                new_step = y.reshape(1, 1, 1)
                x = torch.cat([x[:, 1:, :], new_step], dim=1)
            samples.append(preds)

    return np.array(samples, dtype=float)  # (n_ensemble, n_steps)


# ── main ─────────────────────────────────────────────────────────────────────

def run_lstm_ensemble(parquet_path=None, output_dir=None):
    if not _TORCH_OK:
        return {
            "forecast_mean": np.zeros(N_STEPS),
            "forecast_std": np.zeros(N_STEPS),
            "forecast_all": np.zeros((N_ENSEMBLE, N_STEPS)),
            "future_yr": np.arange(1, N_STEPS + 1) * DT,
            "note": "STUB — torch not installed",
        }
    parquet_path = Path(parquet_path or PARQUET_PATH)
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[LSTM] Loading VADM time series...")
    df = pd.read_parquet(parquet_path).sort_values("age_bp", ascending=False).reset_index(drop=True)
    # Descending age_bp → chronological order: index 0 = oldest (~80 kyr), index -1 = most recent
    vadm = df["sint2000_norm"].values.astype(np.float32)
    N = len(vadm)
    print(f"[LSTM] {N} samples in chronological order")

    # ── dataset ───────────────────────────────────────────────────────────────
    xs, ys = _make_windows(vadm, SEQ_LEN)
    print(f"[LSTM] {len(xs)} training windows (seq_len={SEQ_LEN} = {SEQ_LEN*DT} yr)")

    xs_t = torch.from_numpy(xs).unsqueeze(-1)   # (N_win, seq_len, 1)
    ys_t = torch.from_numpy(ys)                 # (N_win,)
    dataset = TensorDataset(xs_t, ys_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[LSTM] Training on {device}")

    torch.manual_seed(0)
    model = VadmLSTM().to(device)
    _train(model, loader, device, N_EPOCHS, LR)

    # ── seed: last 1000 yr (most recent SEQ_LEN points) ─────────────────────
    seed = vadm[-SEQ_LEN:]   # shape (10,)
    print(f"[LSTM] Seed window: last {SEQ_LEN*DT:.0f} yr  VADM={seed.mean():.3f}±{seed.std():.3f}")

    # ── MC-dropout ensemble forecast ──────────────────────────────────────────
    print(f"[LSTM] Running {N_ENSEMBLE} ensemble members × {N_STEPS} steps ({N_STEPS*DT:.0f} yr)...")
    forecast_all = _mc_forecast(model, seed, N_STEPS, N_ENSEMBLE, device)
    # forecast_all: (N_ENSEMBLE, N_STEPS)

    forecast_mean = forecast_all.mean(axis=0)  # (N_STEPS,)
    forecast_std = forecast_all.std(axis=0)
    p05 = np.percentile(forecast_all, 5, axis=0)
    p95 = np.percentile(forecast_all, 95, axis=0)

    future_yr = np.arange(1, N_STEPS + 1) * DT  # 100, 200, ..., 5000

    # Report milestones
    for mark_yr in [1_000, 2_000, 5_000]:
        idx = mark_yr // DT - 1
        m, s = forecast_mean[idx], forecast_std[idx]
        print(f"[LSTM] VADM at +{mark_yr:,} yr: {m:.3f} ± {s:.3f}")

    # ── plot ──────────────────────────────────────────────────────────────────
    hist_yr = -df["age_bp"].values[::-1]  # negative = past (years from now)
    hist_vadm = vadm[::-1]                # chronological reversed to future-positive axis

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#111111")
    ax.set_facecolor("#1a1a2e")

    # Historical record (last 10 kyr for readability)
    hist_mask = hist_yr >= -10_000
    ax.plot(hist_yr[hist_mask] / 1000, hist_vadm[hist_mask],
            color="#aaddff", lw=1.0, alpha=0.7, label="Sint-2000 (hist.)")

    # Ensemble trajectories (thin)
    for i in range(N_ENSEMBLE):
        ax.plot(future_yr / 1000, forecast_all[i], color="#ff8844", alpha=0.05, lw=0.8)

    # Mean + 90% band
    ax.fill_between(future_yr / 1000, p05, p95, color="#ff8844", alpha=0.25, label="90% PI")
    ax.plot(future_yr / 1000, forecast_mean, color="#ff4400", lw=2.2, label="LSTM mean")

    ax.axhline(0.25, color="white", lw=0.9, ls=":", alpha=0.7, label="Laschamp threshold (0.25)")
    ax.axvline(0, color="#888888", lw=0.8, ls="--", alpha=0.5, label="Present")

    ax.set_xlabel("Years from now (kyr; negative = past)", color="white", fontsize=11)
    ax.set_ylabel("VADM_norm", color="white", fontsize=11)
    ax.set_title(f"VADM LSTM Ensemble Forecast (N={N_ENSEMBLE}, MC-Dropout)", color="white",
                 fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(fontsize=9, facecolor="#222", labelcolor="white")
    ax.set_ylim(0, max(hist_vadm[hist_mask].max() * 1.1, 0.6))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = output_dir / "vadm_lstm_forecast.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[LSTM] Saved → {out_path}")
    plt.show()
    plt.close()

    return {
        "forecast_mean": forecast_mean,        # shape (N_STEPS,)  ← test checks len == 50
        "forecast_std": forecast_std,           # shape (N_STEPS,)
        "forecast_all": forecast_all,           # shape (N_ENSEMBLE, N_STEPS)
        "future_yr": future_yr,
    }


if __name__ == "__main__":
    run_lstm_ensemble()
