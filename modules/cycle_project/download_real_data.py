"""
download_real_data.py
Run from project root on a machine WITH internet access:

    python download_real_data.py

Downloads the 5 real NOAA/PANGAEA proxy files into data/raw/,
overwriting any previously generated synthetic placeholders.
"""

import urllib.request
import sys
from pathlib import Path

RAW = Path(__file__).parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (SUBSTRATE palaeoclimate; science use)"}

DATASETS = [
    {
        "key": "gisp2_d18o",
        "dest": "gisp2_d18o.txt",
        "desc": "GISP2 δ18O, Alley 2000 — 0–110 ka BP",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/isotopes/gisp2_d18o_accum_alley2000.txt",
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/isotopes/gisp2_isotopes_accum_alley2000.txt",
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/gisp2/isotopes/gisp2_d18o_201.txt",
        ],
    },
    {
        "key": "vostok_deuterium",
        "dest": "vostok_deuterium.txt",
        "desc": "Vostok ΔTs (deuterium), Petit 1999 — 0–420 ka BP",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/deutnat.txt",
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/deuterium.txt",
        ],
    },
    {
        "key": "vostok_co2",
        "dest": "vostok_co2.txt",
        "desc": "Vostok CO2, Petit 1999 — 0–420 ka BP",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/co2nat.txt",
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/antarctica/vostok/co2.txt",
        ],
    },
    {
        "key": "grip_be10",
        "dest": "grip_be10.txt",
        "desc": "GRIP Be-10 flux, Muscheler 2004 — 0–80 ka BP",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/beryllium/grip_be10_muscheler2004.txt",
            "https://www.ncei.noaa.gov/pub/data/paleo/icecore/greenland/summit/grip/beryllium/grip_be10.txt",
            "https://doi.pangaea.de/10.1594/PANGAEA.59453",
        ],
    },
    {
        "key": "sint2000",
        "dest": "sint2000_vadm.txt",
        "desc": "Sint-2000 VADM, Valet 2005 — 0–2000 ka BP",
        "urls": [
            "https://www.ncei.noaa.gov/pub/data/paleo/magnet/sint2000.txt",
            "https://www.ngdc.noaa.gov/geomag/paleo_mag_datasets/Sint-2000.txt",
            "https://doi.pangaea.de/10.1594/PANGAEA.186810",
        ],
    },
]


def try_download(url: str, dest: Path) -> bool:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = resp.read()
        if len(data) < 500:
            print(f"    [skip]  response too small ({len(data)} bytes) — not a data file")
            return False
        dest.write_bytes(data)
        print(f"    [ok]  {dest.name}  ({len(data)/1024:.1f} kB)")
        return True
    except Exception as e:
        print(f"    [fail] {url}")
        print(f"           {e}")
        return False


def main():
    print("=" * 62)
    print("  SUBSTRATE / cycle_project — Real Data Downloader")
    print("=" * 62)
    print(f"  Destination: {RAW.resolve()}\n")

    results = {}
    for ds in DATASETS:
        dest = RAW / ds["dest"]
        print(f"[{ds['key']}]  {ds['desc']}")
        success = False
        for url in ds["urls"]:
            print(f"  → {url}")
            if try_download(url, dest):
                success = True
                break
        if not success:
            print(f"  *** ALL URLS FAILED — {ds['key']} will remain synthetic ***")
        results[ds["key"]] = success
        print()

    ok   = [k for k, v in results.items() if v]
    fail = [k for k, v in results.items() if not v]

    print("=" * 62)
    print(f"  Downloaded : {len(ok)}   Failed : {len(fail)}")
    if fail:
        print(f"\n  Failed: {fail}")
        print("\n  Manual fallback — save these files to data/raw/:")
        for ds in DATASETS:
            if ds["key"] in fail:
                print(f"    {ds['dest']:30s}  {ds['urls'][0]}")
    print()
    if ok:
        print("  Next step:")
        print("    python src/cycle_detect/fetch_data.py")
        print("    python src/cycle_detect/gnn_prototype.py   (GPU recommended)")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
