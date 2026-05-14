"""
fetch_data.py — CYCLE_DETECT: Phase 0 data ingestion
Downloads, cleans and aligns 4 geological proxy time series to a common 100-year grid.

Proxies:
  1. GISP2 δ18O           — temperature proxy, Greenland ice core
  2. Vostok dD            — temperature proxy, Antarctic ice core
  3. Beryllium-10 (GRIP)  — cosmic ray / geomagnetic field strength proxy
  4. Sint-2000 VADM       — Virtual Axial Dipole Moment (paleomagnetic stack)

Output:
  data/processed/<proxy>.parquet   — each series on its own time axis
  data/processed/aligned.parquet   — all 4 series on 100-year common grid
  data/processed/overview.png      — multi-panel time series plot

Usage:
  python src/cycle_detect/fetch_data.py [--force]
"""

import os
import sys
import argparse
import urllib.request
import io
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset registry
# NOTE: URLs verified against NOAA NCEI as of 2025-01.
#       If a URL 404s, check: https://www.ncei.noaa.gov/pub/data/paleo/
# ---------------------------------------------------------------------------
DATASETS = {
    "gisp2_d18o": {
        "url": (
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
            "summit/gisp2/isotopes/gisp2_d18o_201.txt"
        ),
        "raw_file": "gisp2_d18o.txt",
        "description": "GISP2 δ18O — Stuiver et al. 1995",
        # ⚠ VERIFY: if 404, try .../isotopes/gisp2isotopes.txt
    },
    "vostok_dd": {
        "url": (
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/"
            "vostok/deuterium.txt"
        ),
        "raw_file": "vostok_deuterium.txt",
        "description": "Vostok δD — Petit et al. 1999",
    },
    "grip_be10": {
        "url": (
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
            "summit/grip/beryllium/grip_be10.txt"
        ),
        "raw_file": "grip_be10.txt",
        "description": "GRIP Be-10 — Yiou et al. 1997",
        # ⚠ VERIFY: may need to use Muscheler 2004 from PANGAEA instead
        # Fallback PANGAEA: https://doi.pangaea.de/10.1594/PANGAEA.59453
    },
    "sint2000_vadm": {
        "url": (
            "https://www.ncei.noaa.gov/pub/data/paleo/magnet/"
            "sint2000.txt"
        ),
        "raw_file": "sint2000_vadm.txt",
        "description": "Sint-2000 VADM — Guyodo & Valet 1999",
        # ⚠ VERIFY: check https://www.ncei.noaa.gov/pub/data/paleo/magnet/
        # Alternative: https://www.earthbyte.org/category/resources/
    },
}

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download(url: str, dest: Path, force: bool = False) -> bool:
    """Download url → dest. Returns True on success, False on HTTP error."""
    if dest.exists() and not force:
        print(f"  [cache] {dest.name}")
        return True
    print(f"  [download] {url}")
    try:
        with TqdmUpTo(unit="B", unit_scale=True, miniters=1, desc=dest.name) as t:
            urllib.request.urlretrieve(url, dest, reporthook=t.update_to)
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False


# ---------------------------------------------------------------------------
# Parsers — each NOAA file has its own header/format
# ---------------------------------------------------------------------------

def _skip_header(lines: list[str], marker: str = "----") -> int:
    """Return index of first data line after the header separator."""
    for i, line in enumerate(lines):
        if marker in line:
            return i + 1
    # Fallback: skip comment lines starting with #
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith("#"):
            return i
    return 0


def parse_gisp2_d18o(path: Path) -> pd.DataFrame:
    """
    GISP2 δ18O: two-column format (Age_yrBP, d18O)
    Age is in years BP (before 1950). Younger Dryas visible ~12,900 BP.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                age = float(parts[0])
                val = float(parts[1])
                records.append((age, val))
            except ValueError:
                continue
    df = pd.DataFrame(records, columns=["age_bp", "d18o"])
    df = df.sort_values("age_bp").reset_index(drop=True)
    df = df[df["age_bp"] >= 0]
    return df


def parse_vostok_dd(path: Path) -> pd.DataFrame:
    """
    Vostok δD: columns Age_yrBP, Depth_m, DeltaD, DeltaTs
    We use Age_yrBP and DeltaTs (temperature anomaly vs. present).
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                age = float(parts[0])
                dts = float(parts[3])   # temperature anomaly (°C)
                records.append((age, dts))
            except ValueError:
                continue
    df = pd.DataFrame(records, columns=["age_bp", "delta_ts"])
    df = df.sort_values("age_bp").reset_index(drop=True)
    df = df[df["age_bp"] >= 0]
    return df


def parse_grip_be10(path: Path) -> pd.DataFrame:
    """
    GRIP Be-10: columns Age_yrBP, Be10_conc (atoms/g)
    Higher Be-10 → weaker geomagnetic field → more cosmic rays.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                age = float(parts[0])
                be10 = float(parts[1])
                records.append((age, be10))
            except ValueError:
                continue
    df = pd.DataFrame(records, columns=["age_bp", "be10"])
    df = df.sort_values("age_bp").reset_index(drop=True)
    df = df[df["age_bp"] >= 0]
    return df


def parse_sint2000_vadm(path: Path) -> pd.DataFrame:
    """
    Sint-2000 VADM: columns Age_ka, VADM (10^22 A·m²)
    Age in kiloyears BP — convert to years BP.
    Lower VADM → field collapse → potential excursion/reversal.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                age_ka = float(parts[0])
                vadm   = float(parts[1])
                records.append((age_ka * 1000.0, vadm))
            except ValueError:
                continue
    df = pd.DataFrame(records, columns=["age_bp", "vadm"])
    df = df.sort_values("age_bp").reset_index(drop=True)
    df = df[df["age_bp"] >= 0]
    return df


PARSERS = {
    "gisp2_d18o":   parse_gisp2_d18o,
    "vostok_dd":    parse_vostok_dd,
    "grip_be10":    parse_grip_be10,
    "sint2000_vadm": parse_sint2000_vadm,
}

PROXY_COL = {
    "gisp2_d18o":   "d18o",
    "vostok_dd":    "delta_ts",
    "grip_be10":    "be10",
    "sint2000_vadm": "vadm",
}

PROXY_LABEL = {
    "gisp2_d18o":   "GISP2 δ¹⁸O (‰)",
    "vostok_dd":    "Vostok ΔTs (°C)",
    "grip_be10":    "GRIP Be-10 (atoms/g)",
    "sint2000_vadm": "Sint-2000 VADM (10²² A·m²)",
}

# ---------------------------------------------------------------------------
# Alignment & normalisation
# ---------------------------------------------------------------------------

def resample_to_grid(df: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    """
    Linear interpolation of a proxy series onto a common time grid.
    Returns NaN outside the series' temporal coverage.
    """
    return np.interp(grid, df["age_bp"].values, df[col].values,
                     left=np.nan, right=np.nan)


def z_score(arr: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation ignoring NaNs."""
    mu  = np.nanmean(arr)
    sig = np.nanstd(arr)
    return (arr - mu) / (sig + 1e-12)


def align_proxies(
    frames: dict[str, pd.DataFrame],
    t_min: float = 0.0,
    t_max: float = 150_000.0,
    step: float = 100.0,
) -> pd.DataFrame:
    """
    Resample all proxy series to a common grid [t_min, t_max] with given step.
    t_max clipped to the maximum common coverage.
    """
    # Clip t_max to actual data coverage
    actual_max = min(df["age_bp"].max() for df in frames.values())
    t_max = min(t_max, actual_max)

    grid = np.arange(t_min, t_max + step, step)
    aligned = pd.DataFrame({"age_bp": grid})

    for key, df in frames.items():
        col = PROXY_COL[key]
        interp = resample_to_grid(df, col, grid)
        aligned[key] = interp
        aligned[f"{key}_norm"] = z_score(interp)

    return aligned


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

EVENTS = {
    "Younger Dryas": 12_900,
    "8.2 ka event":   8_200,
    "Laschamp excursion": 41_000,
    "Last Glacial Max": 20_000,
}

PALETTE = ["#0A84FF", "#30D158", "#FF9F0A", "#FF453A"]


def plot_overview(aligned: pd.DataFrame, out_path: Path):
    keys = list(PROXY_COL.keys())
    fig, axes = plt.subplots(
        len(keys), 1, figsize=(14, 10), sharex=True,
        gridspec_kw={"hspace": 0.05}
    )
    fig.patch.set_facecolor("#0F172A")

    x = aligned["age_bp"].values / 1000.0  # ka BP

    for ax, key, color in zip(axes, keys, PALETTE):
        y = aligned[f"{key}_norm"].values
        ax.plot(x, y, color=color, linewidth=0.9, alpha=0.9)
        ax.fill_between(x, y, alpha=0.15, color=color)
        ax.set_ylabel(PROXY_LABEL[key], color="#94A3B8", fontsize=8)
        ax.set_facecolor("#0F172A")
        ax.tick_params(colors="#64748B", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E293B")
        ax.axhline(0, color="#334155", linewidth=0.5, linestyle="--")

        # Mark known events
        for evt_name, evt_age_yr in EVENTS.items():
            evt_ka = evt_age_yr / 1000.0
            if x.min() <= evt_ka <= x.max():
                ax.axvline(evt_ka, color="#F1FA8C", linewidth=0.7,
                           linestyle=":", alpha=0.7)
                if ax is axes[0]:
                    ax.text(evt_ka + 0.3, ax.get_ylim()[1] * 0.85,
                            evt_name, color="#F1FA8C", fontsize=6,
                            rotation=90, va="top")

    axes[-1].set_xlabel("Age (ka BP)", color="#94A3B8", fontsize=9)
    axes[-1].invert_xaxis()   # older on right, present on left

    fig.suptitle(
        "CYCLE_DETECT — Geological Proxy Alignment",
        color="#E2E8F0", fontsize=11, fontweight="bold", y=0.995
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CYCLE_DETECT: data ingestion")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cached")
    parser.add_argument("--t-max", type=float, default=150_000.0,
                        help="Max age BP for aligned grid (default 150 ka)")
    args = parser.parse_args()

    print("=" * 60)
    print("CYCLE_DETECT — Phase 0: Data Ingestion")
    print("=" * 60)

    frames: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for key, meta in DATASETS.items():
        print(f"\n[{key}] {meta['description']}")
        raw_path = RAW / meta["raw_file"]
        ok = download(meta["url"], raw_path, force=args.force)

        if not ok or not raw_path.exists():
            print(f"  [SKIP] could not obtain {key}")
            failed.append(key)
            continue

        try:
            df = PARSERS[key](raw_path)
            parquet_path = PROC / f"{key}.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"  [ok] {len(df)} records | "
                  f"age [{df['age_bp'].min():.0f}–{df['age_bp'].max():.0f}] BP "
                  f"→ {parquet_path.name}")
            frames[key] = df
        except Exception as e:
            print(f"  [PARSE ERROR] {e}")
            failed.append(key)

    if not frames:
        print("\n[FATAL] No proxy data loaded. Check URLs and network.")
        sys.exit(1)

    if failed:
        print(f"\n[WARNING] Failed proxies: {failed}")
        print("  → Continuing with available data. GNN will use partial graph.")

    print("\n[align] Resampling to 100-year common grid …")
    aligned = align_proxies(frames, t_max=args.t_max, step=100.0)
    aligned_path = PROC / "aligned.parquet"
    aligned.to_parquet(aligned_path, index=False)
    print(f"  [ok] grid points: {len(aligned)} | "
          f"age [{aligned['age_bp'].min():.0f}–{aligned['age_bp'].max():.0f}] BP "
          f"→ {aligned_path.name}")

    # Coverage report
    print("\n[coverage]")
    for key in frames:
        col = f"{key}_norm"
        n_nan = aligned[col].isna().sum()
        pct = 100 * (1 - n_nan / len(aligned))
        print(f"  {key:20s}: {pct:.1f}% grid coverage")

    print("\n[plot] Generating overview …")
    plot_overview(aligned, PROC / "overview.png")

    print("\n[done] All outputs in:", PROC)
    print("  Next: python src/cycle_detect/gnn_prototype.py")


if __name__ == "__main__":
    main()
