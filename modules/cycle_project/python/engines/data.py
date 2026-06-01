"""
fetch_data.py — SUBSTRATE / CYCLE_DETECT: Phase 1 data ingestion
=================================================================

Downloads, cleans and aligns 5 palaeoclimate proxy time series to a
common 100-year grid.  Outputs Apache Parquet files + overview PNG.

Proxies
-------
  1. GISP2 δ¹⁸O       — Greenland temperature (Alley 2000)
  2. Vostok ΔTs        — Antarctic temperature (Petit et al. 1999)
  3. Vostok CO₂        — Atmospheric CO₂ (Petit et al. 1999)
  4. GRIP Be-10        — Cosmic-ray / solar modulation (Muscheler et al. 2004)
  5. Sint-2000 VADM    — Geomagnetic dipole moment stack (Valet et al. 2005)

Outputs
-------
  data/raw/<proxy>.txt            — downloaded originals
  data/processed/<proxy>.parquet  — cleaned per-proxy series
  data/processed/aligned.parquet  — all 5 proxies on 100-yr common grid
  data/processed/overview.png     — 5-panel dark-theme alignment plot

Usage
-----
  python src/cycle_detect/fetch_data.py [--force] [--t-max 150000]

SubstrateLab API
----------------
  from cycle_detect.fetch_data import fetch_all_proxies
  paths = fetch_all_proxies(out_dir=Path("data/processed"))
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Dataset registry — URLs verified / fallback chains documented
# ---------------------------------------------------------------------------
DATASETS: dict[str, dict] = {
    "gisp2_d18o": {
        "urls": [
            # Alley 2000 — ARCHITECTURE verified URL
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
            "summit/gisp2/isotopes/gisp2_d18o_accum_alley2000.txt",
            # Fallback: older Stuiver 1997 file
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
            "summit/gisp2/isotopes/gisp2_d18o_201.txt",
        ],
        "raw_file": "gisp2_d18o.txt",
        "description": "GISP2 δ¹⁸O + accumulation, Alley 2000 — 0–110 ka BP",
        "value_col": "d18o",
        "label": "GISP2 δ¹⁸O (‰)",
    },
    "vostok_deuterium": {
        "urls": [
            # Petit 1999 — ARCHITECTURE verified URL
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/"
            "vostok/deutnat.txt",
            # Fallback: alternate filename
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/"
            "vostok/deuterium.txt",
        ],
        "raw_file": "vostok_deuterium.txt",
        "description": "Vostok ΔTs, Petit et al. 1999 — 0–420 ka BP",
        "value_col": "delta_ts",
        "label": "Vostok ΔTs (°C)",
    },
    "vostok_co2": {
        "urls": [
            # Petit 1999 — ARCHITECTURE verified URL
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/"
            "vostok/co2nat.txt",
        ],
        "raw_file": "vostok_co2.txt",
        "description": "Vostok CO₂, Petit et al. 1999 — 0–420 ka BP",
        "value_col": "co2_ppmv",
        "label": "Vostok CO₂ (ppmv)",
    },
    "grip_be10": {
        "urls": [
            # Muscheler et al. 2004 (ARCHITECTURE primary — unverified filename)
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
            "summit/grip/beryllium/grip_be10_muscheler2004.txt",
            # Yiou et al. 1997 (older record, same directory)
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/"
            "summit/grip/beryllium/grip_be10.txt",
            # PANGAEA fallback: Muscheler 2004 dataset
            "https://doi.pangaea.de/10.1594/PANGAEA.59453",
        ],
        "raw_file": "grip_be10.txt",
        "description": "GRIP Be-10 flux, Muscheler et al. 2004 — 0–110 ka BP",
        "value_col": "be10",
        "label": "GRIP ¹⁰Be (atoms/g)",
        "warn": "⚠️  Browse https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/beryllium/ to confirm filename",
    },
    "sint2000": {
        "urls": [
            # NGDC primary (ARCHITECTURE URL)
            "https://www.ngdc.noaa.gov/geomag/paleo_mag_datasets/Sint-2000.txt",
            # NCEI mirror (fetch_data.py v1 URL)
            "https://www.ncei.noaa.gov/pub/data/paleo/magnet/sint2000.txt",
            # PANGAEA: Valet et al. 2005
            "https://doi.pangaea.de/10.1594/PANGAEA.186810",
        ],
        "raw_file": "sint2000_vadm.txt",
        "description": "Sint-2000 VADM stack, Valet et al. 2005 — 0–2000 ka at 1-ka res",
        "value_col": "vadm",
        "label": "Sint-2000 VADM (10²² A·m²)",
        "warn": "⚠️  Fallback: PANGAEA DOI 10.1594/PANGAEA.186810",
    },
}

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_one(url: str, dest: Path, force: bool = False) -> bool:
    """Try to download url → dest. Returns True on success."""
    if dest.exists() and not force:
        print(f"    [cache] {dest.name}")
        return True
    try:
        headers = {"User-Agent": "Mozilla/5.0 (SUBSTRATE palaeoclimate fetcher)"}
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=30) as resp, open(dest, "wb") as f:
            f.write(resp.read())
        size_kb = dest.stat().st_size / 1024
        print(f"    [ok] {dest.name}  ({size_kb:.1f} kB)")
        return True
    except Exception as e:
        print(f"    [fail] {url}  → {e}")
        return False


def download_proxy(key: str, meta: dict, force: bool = False) -> Path | None:
    """Try each URL in the fallback chain. Returns raw file path or None."""
    dest = RAW / meta["raw_file"]
    if "warn" in meta:
        print(f"  {meta['warn']}")
    for url in meta["urls"]:
        if _download_one(url, dest, force=force):
            return dest
    print(f"  [SKIP] all URLs failed for {key}")
    return None


# ---------------------------------------------------------------------------
# Parsers — each NOAA file has its own header/column layout
# ---------------------------------------------------------------------------

def _skip_header_lines(lines: list[str]) -> int:
    """Return index of first numeric data line (skip comment/header rows)."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts:
            try:
                float(parts[0])
                return i   # first line where col-0 is numeric
            except ValueError:
                continue
    return 0


def parse_gisp2_d18o(path: Path) -> pd.DataFrame:
    """
    GISP2 δ¹⁸O (Alley 2000):
      Col 0: Age (yr BP)
      Col 1: δ¹⁸O (‰)
      Col 2: Accumulation (m/yr)  [optional, may be absent in older file]

    Younger Dryas onset visible at ~12,900 yr BP as sharp negative excursion.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header_lines(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                records.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    df = pd.DataFrame(records, columns=["age_bp", "d18o"])
    df = df[df["age_bp"] >= 0].sort_values("age_bp").reset_index(drop=True)
    return df


def parse_vostok_deuterium(path: Path) -> pd.DataFrame:
    """
    Vostok deutnat.txt (Petit et al. 1999):
      Col 0: Depth (m)
      Col 1: Ice age (yr BP)
      Col 2: Deuterium δD (‰)
      Col 3: ΔTs (°C relative to present mean)

    We use ice age (col 1) and ΔTs (col 3).
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header_lines(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                records.append((float(parts[1]), float(parts[3])))
            except ValueError:
                continue
    if not records and lines:
        # Try 2-column fallback (old deuterium.txt format)
        for line in lines[start:]:
            parts = line.split()
            if len(parts) >= 2:
                try:
                    records.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    df = pd.DataFrame(records, columns=["age_bp", "delta_ts"])
    df = df[df["age_bp"] >= 0].sort_values("age_bp").reset_index(drop=True)
    return df


def parse_vostok_co2(path: Path) -> pd.DataFrame:
    """
    Vostok co2nat.txt (Petit et al. 1999):
      Col 0: Depth (m)
      Col 1: Ice age (yr BP)
      Col 2: Mean air age (yr BP)
      Col 3: CO₂ (ppmv)

    We use mean air age (col 2) and CO₂ (col 3).
    Interglacial highs ~280 ppmv; glacial lows ~180 ppmv.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header_lines(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                records.append((float(parts[2]), float(parts[3])))
            except ValueError:
                continue
        elif len(parts) == 2:
            try:
                records.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    df = pd.DataFrame(records, columns=["age_bp", "co2_ppmv"])
    df = df[df["age_bp"] >= 0].sort_values("age_bp").reset_index(drop=True)
    return df


def parse_grip_be10(path: Path) -> pd.DataFrame:
    """
    GRIP Be-10 (Muscheler 2004 or Yiou 1997):
      Col 0: Age (yr BP)
      Col 1: Be-10 concentration or flux (atoms/g or 10⁴ atoms/g)

    Higher Be-10 → weaker geomagnetic field / higher cosmic-ray flux.
    Spike at ~12,900 BP correlates with YD onset.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header_lines(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                records.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    df = pd.DataFrame(records, columns=["age_bp", "be10"])
    df = df[df["age_bp"] >= 0].sort_values("age_bp").reset_index(drop=True)
    return df


def parse_sint2000(path: Path) -> pd.DataFrame:
    """
    Sint-2000 VADM (Valet et al. 2005 / Guyodo & Valet 1999):
      Col 0: Age (ka BP)   ← kiloyears — must convert to years
      Col 1: VADM (10²² A·m²)

    Age resolution: 1 ka.  Coverage: 0–2000 ka.
    VADM < 4 × 10²² A·m² → field weakness / potential excursion.
    Laschamps at ~41 ka: VADM drops to ~1.5.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    start = _skip_header_lines(lines)
    records = []
    for line in lines[start:]:
        parts = line.split()
        if len(parts) >= 2:
            try:
                age_ka = float(parts[0])
                vadm   = float(parts[1])
                records.append((age_ka * 1000.0, vadm))   # ka → yr BP
            except ValueError:
                continue
    df = pd.DataFrame(records, columns=["age_bp", "vadm"])
    df = df[df["age_bp"] >= 0].sort_values("age_bp").reset_index(drop=True)
    return df


PARSERS: dict[str, Any] = {
    "gisp2_d18o":       parse_gisp2_d18o,
    "vostok_deuterium": parse_vostok_deuterium,
    "vostok_co2":       parse_vostok_co2,
    "grip_be10":        parse_grip_be10,
    "sint2000":         parse_sint2000,
}

# ---------------------------------------------------------------------------
# Grid alignment
# ---------------------------------------------------------------------------

def resample_to_grid(df: pd.DataFrame, col: str, grid: np.ndarray) -> np.ndarray:
    """Linear interpolation onto common grid; NaN outside coverage."""
    return np.interp(
        grid, df["age_bp"].values, df[col].values,
        left=np.nan, right=np.nan,
    )


def z_score(arr: np.ndarray) -> np.ndarray:
    mu, sig = np.nanmean(arr), np.nanstd(arr)
    return (arr - mu) / (sig + 1e-12)


def align_proxies(
    frames: dict[str, pd.DataFrame],
    t_min: float = 0.0,
    t_max: float = 150_000.0,
    step: float = 100.0,
) -> pd.DataFrame:
    """Resample all available proxies to a common 100-yr grid."""
    # Clip t_max to shortest proxy coverage
    actual_max = min(df["age_bp"].max() for df in frames.values())
    t_max = min(t_max, actual_max)
    grid = np.arange(t_min, t_max + step, step)

    aligned = pd.DataFrame({"age_bp": grid})
    for key, df in frames.items():
        col = DATASETS[key]["value_col"]
        interp = resample_to_grid(df, col, grid)
        aligned[key] = interp
        aligned[f"{key}_norm"] = z_score(interp)

    return aligned


# ---------------------------------------------------------------------------
# Overview plot
# ---------------------------------------------------------------------------

_KNOWN_EVENTS = {
    "YD onset\n12.9 ka":       12_900,
    "YD end\n11.7 ka":         11_700,
    "8.2 ka":                   8_200,
    "Laschamps\n41 ka":        41_000,
    "LGM\n20 ka":              20_000,
}

_PALETTE = ["#0A84FF", "#30D158", "#FF9F0A", "#FF453A", "#BF5AF2"]


def plot_overview(aligned: pd.DataFrame, out_path: Path) -> None:
    keys = [k for k in DATASETS if k in aligned.columns]
    n = len(keys)
    fig, axes = plt.subplots(n, 1, figsize=(15, 11), sharex=True,
                             gridspec_kw={"hspace": 0.04})
    if n == 1:
        axes = [axes]
    fig.patch.set_facecolor("#0A0E1A")

    x = aligned["age_bp"].values / 1000.0   # ka BP

    for ax, key, color in zip(axes, keys, _PALETTE):
        y = aligned[f"{key}_norm"].values
        ax.plot(x, y, color=color, linewidth=0.85, alpha=0.92)
        ax.fill_between(x, y, alpha=0.13, color=color)
        ax.set_ylabel(DATASETS[key]["label"], color="#94A3B8", fontsize=7.5,
                      labelpad=4)
        ax.set_facecolor("#0A0E1A")
        ax.tick_params(colors="#475569", labelsize=6.5)
        for spine in ax.spines.values():
            spine.set_edgecolor("#1E293B")
        ax.axhline(0, color="#1E293B", linewidth=0.6, linestyle="--")
        ax.set_ylim(-3.5, 3.5)

        for label, age_yr in _KNOWN_EVENTS.items():
            age_ka = age_yr / 1000.0
            if x.min() <= age_ka <= x.max():
                ax.axvline(age_ka, color="#F1FA8C", linewidth=0.65,
                           linestyle=":", alpha=0.75)
                if ax is axes[0]:
                    ax.text(age_ka + 0.2, 3.0, label,
                            color="#F1FA8C", fontsize=5.5,
                            rotation=90, va="top", ha="left",
                            linespacing=1.3)

    axes[-1].set_xlabel("Age (ka BP)", color="#94A3B8", fontsize=9)
    axes[-1].invert_xaxis()   # older → right

    fig.suptitle(
        "SUBSTRATE — 5-Proxy Palaeoclimate Alignment  "
        "(z-scored,  older → right)",
        color="#E2E8F0", fontsize=10.5, fontweight="bold", y=0.997,
    )

    plt.savefig(out_path, dpi=160, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] {out_path}")


# ---------------------------------------------------------------------------
# Public API — used by SubstrateLab.geomagnetic instrument
# ---------------------------------------------------------------------------

def fetch_all_proxies(
    out_dir: Path | None = None,
    force: bool = False,
    t_max: float = 150_000.0,
) -> list[Path]:
    """
    Download all 5 proxies, parse, and write Parquet files.

    Parameters
    ----------
    out_dir : Path
        Where to write .parquet files.  Defaults to data/processed/.
    force : bool
        Re-download even if cached.
    t_max : float
        Maximum age BP for aligned grid (default 150,000).

    Returns
    -------
    List of Parquet file paths that were successfully written.
    """
    global RAW, PROC
    RAW.mkdir(parents=True, exist_ok=True)
    if out_dir:
        PROC = Path(out_dir)
    PROC.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    parquet_paths: list[Path] = []
    failed: list[str] = []

    for key, meta in DATASETS.items():
        print(f"\n[{key}]  {meta['description']}")
        raw_path = download_proxy(key, meta, force=force)
        if raw_path is None:
            failed.append(key)
            continue
        try:
            df = PARSERS[key](raw_path)
            pq = PROC / f"{key}.parquet"
            df.to_parquet(pq, index=False)
            print(f"  rows={len(df):,}  age=[{df['age_bp'].min():.0f}–"
                  f"{df['age_bp'].max():.0f}] yr BP  → {pq.name}")
            frames[key] = df
            parquet_paths.append(pq)
        except Exception as e:
            print(f"  [PARSE ERROR] {e}")
            failed.append(key)

    if not frames:
        print("\n[FATAL] No proxy data loaded.")
        return []

    if failed:
        print(f"\n[WARNING] Failed proxies: {failed}")
        print("  GNN will proceed with partial graph.")

    # Aligned grid
    print("\n[align] Resampling to 100-yr common grid …")
    aligned = align_proxies(frames, t_max=t_max)
    aligned_pq = PROC / "aligned.parquet"
    aligned.to_parquet(aligned_pq, index=False)
    parquet_paths.append(aligned_pq)
    print(f"  grid={len(aligned):,} pts  "
          f"age=[{aligned['age_bp'].min():.0f}–{aligned['age_bp'].max():.0f}]")

    for key in frames:
        col = f"{key}_norm"
        pct = 100 * (1 - aligned[col].isna().sum() / len(aligned))
        print(f"  {key:<22}: {pct:.1f}% coverage")

    # Overview plot
    print("\n[plot] Building overview …")
    plot_overview(aligned, PROC / "overview.png")

    return parquet_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SUBSTRATE — fetch and align palaeoclimate proxies"
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if cached")
    parser.add_argument("--t-max", type=float, default=150_000.0,
                        help="Max age BP for aligned grid (default 150 ka)")
    args = parser.parse_args()

    print("=" * 64)
    print("SUBSTRATE / CYCLE_DETECT — Phase 1: Data Ingestion")
    print("=" * 64)

    paths = fetch_all_proxies(force=args.force, t_max=args.t_max)

    if paths:
        print(f"\n[done] {len(paths)} files in {PROC}")
        print("  Next:  python src/cycle_detect/gnn_prototype.py")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
