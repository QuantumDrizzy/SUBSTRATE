"""
run_real_data.py
================
One-shot real-data pipeline. Run from project root on your machine:

    python run_real_data.py                   # CPU-only (numpy GNN)
    python run_real_data.py --gpu             # full GraphSAGE on CUDA
    python run_real_data.py --skip-download   # if you already have data/raw/ files

Steps:
  1. Download real NOAA/PANGAEA proxy files → data/raw/
  2. Parse + align → data/processed/aligned.parquet  (801 rows, 5 proxies)
  3. Numpy GNN anomaly scan → anomaly_scores.parquet + plots
  4. [optional] Full GraphSAGE retrain (--gpu flag)
  5. Forward probe: spectral + decay + fingerprint → probe_state.json
"""

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

HEADERS = {"User-Agent": "Mozilla/5.0 (SUBSTRATE palaeoclimate; science)"}

DATASETS = [
    {
        "key": "gisp2_d18o",
        "dest": "gisp2_d18o.txt",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/isotopes/gisp2_d18o_accum_alley2000.txt",
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/isotopes/gisp2_isotopes_accum_alley2000.txt",
        ],
    },
    {
        "key": "vostok_deuterium",
        "dest": "vostok_deuterium.txt",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/deutnat.txt",
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/deuterium.txt",
        ],
    },
    {
        "key": "vostok_co2",
        "dest": "vostok_co2.txt",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/co2nat.txt",
        ],
    },
    {
        "key": "grip_be10",
        "dest": "grip_be10.txt",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/beryllium/grip_be10_muscheler2004.txt",
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/beryllium/grip_be10.txt",
            "https://doi.pangaea.de/10.1594/PANGAEA.59453",
        ],
    },
    {
        "key": "sint2000",
        "dest": "sint2000_vadm.txt",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/magnet/sint2000.txt",
            "https://www.ngdc.noaa.gov/geomag/paleo_mag_datasets/Sint-2000.txt",
            "https://doi.pangaea.de/10.1594/PANGAEA.186810",
        ],
    },
]


# ── Step 1: Download ──────────────────────────────────────────────────────────

def download_all(raw_dir: Path) -> list[str]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for ds in DATASETS:
        dest = raw_dir / ds["dest"]
        print(f"\n[download] {ds['key']}")
        ok = False
        for url in ds["urls"]:
            print(f"  → {url}")
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                if len(data) < 500:
                    print(f"    [skip] too small ({len(data)} bytes)")
                    continue
                dest.write_bytes(data)
                print(f"    [ok]  {len(data)/1024:.1f} kB → {dest.name}")
                ok = True
                break
            except Exception as e:
                print(f"    [fail] {e}")
        if not ok:
            print(f"  *** FAILED — {ds['key']} will remain synthetic ***")
            failed.append(ds["key"])
    return failed


# ── Step 2: Fetch + align ─────────────────────────────────────────────────────

def run_fetch(force: bool = True):
    print("\n" + "=" * 55)
    print("STEP 2: Parse + align proxies")
    print("=" * 55)
    from cycle_detect.fetch_data import fetch_all_proxies
    paths = fetch_all_proxies(force=force)
    print(f"  Loaded {len(paths)} proxy datasets")
    return paths


# ── Step 3: Numpy GNN ─────────────────────────────────────────────────────────

def run_numpy_gnn():
    print("\n" + "=" * 55)
    print("STEP 3: Numpy GNN anomaly scan")
    print("=" * 55)
    import sys as _sys
    _argv_bak = _sys.argv[:]
    _sys.argv = [_sys.argv[0]]
    from cycle_detect import gnn_numpy
    gnn_numpy.main()
    _sys.argv = _argv_bak


# ── Step 4: Full GraphSAGE (GPU) ──────────────────────────────────────────────

def run_graphsage(epochs: int = 200):
    print("\n" + "=" * 55)
    print("STEP 4: Full GraphSAGE retrain (CUDA)")
    print("=" * 55)
    try:
        from cycle_detect.gnn_prototype import run_anomaly_scan, _TORCH_OK
        if not _TORCH_OK:
            print("  torch not available — skipping GraphSAGE")
            return
        result = run_anomaly_scan(
            data_root=ROOT / "data",
            epochs=epochs,
            threshold_sigma=2.0,
            device="cuda",
        )
        n_anom = len(result.get("anomaly_windows", []))
        thr    = result.get("threshold", 0)
        print(f"  anomaly_windows: {n_anom}  threshold: {thr:.6f}")
    except Exception as e:
        print(f"  GraphSAGE failed: {e}")


# ── Step 5: Forward probe ─────────────────────────────────────────────────────

def run_forward_probe():
    print("\n" + "=" * 55)
    print("STEP 5: Forward probe (spectral + decay + fingerprint)")
    print("=" * 55)
    import json, datetime
    from pathlib import Path
    import matplotlib; matplotlib.use("Agg")

    proc = ROOT / "data" / "processed"
    parquet = proc / "aligned.parquet"

    from forward_probe.spectral    import run_spectral
    from forward_probe.decay_model import run_decay_model
    from forward_probe.fingerprint import run_fingerprint

    r_spec  = run_spectral(parquet_path=parquet, output_dir=proc)
    r_decay = run_decay_model(parquet_path=parquet, output_dir=proc)
    r_fp    = run_fingerprint(parquet_path=parquet)

    instr_yr = r_decay["thresholds"].get("instrumental", {}).get("threshold_yr")

    state = {
        "pre_excursion_prob":      r_fp["probability"],
        "loo_accuracy":            r_fp["loo_accuracy"],
        "status":                  r_fp["status"],
        "spectral_top_period_yr":  int(r_spec["dominant_periods"][0]) if r_spec["dominant_periods"] else None,
        "instrumental_threshold_yr": int(instr_yr) if instr_yr else None,
        "generated_at":            datetime.datetime.utcnow().isoformat(),
    }
    out = proc / "probe_state.json"
    out.write_text(json.dumps(state, indent=2))
    print(f"\n  Probe state → {out}")
    for k, v in state.items():
        if k != "generated_at":
            print(f"    {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SUBSTRATE real-data pipeline")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download step (data/raw/ files already exist)")
    parser.add_argument("--gpu", action="store_true",
                        help="Run full GraphSAGE on CUDA after numpy GNN")
    parser.add_argument("--epochs", type=int, default=200,
                        help="GraphSAGE epochs (default 200)")
    args = parser.parse_args()

    print("=" * 55)
    print("  SUBSTRATE / cycle_project — Real Data Pipeline")
    print("=" * 55)

    raw_dir = ROOT / "data" / "raw"
    failed  = []

    if not args.skip_download:
        print("\nSTEP 1: Download real NOAA/PANGAEA data")
        failed = download_all(raw_dir)
        if failed:
            print(f"\n  WARNING: {len(failed)} dataset(s) failed to download.")
            print(f"  Will use synthetic placeholders for: {failed}")
    else:
        print("\nSTEP 1: [skipped] using existing data/raw/ files")

    run_fetch(force=False)   # files already in data/raw/ from step 1
    run_numpy_gnn()

    if args.gpu:
        run_graphsage(epochs=args.epochs)

    run_forward_probe()

    print("\n" + "=" * 55)
    print("  Pipeline complete.")
    print("  Next: open notebooks/02_gnn_analysis.ipynb to review results")
    if failed:
        print(f"  Note: {len(failed)} proxy(ies) used synthetic data: {failed}")
    print("=" * 55)


if __name__ == "__main__":
    main()
