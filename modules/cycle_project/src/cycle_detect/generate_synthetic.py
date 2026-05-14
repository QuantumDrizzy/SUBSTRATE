"""
generate_synthetic.py — Synthetic NOAA-format data for pipeline validation.
Generates files that are byte-for-byte compatible with the real parsers.
Embeds known geophysical events so the GNN has real signal to find.

Events embedded:
  - Younger Dryas onset  ~12,900 BP  (all proxies synchronous)
  - 8.2 ka cold event    ~ 8,200 BP  (weaker signal)
  - Laschamp excursion   ~41,000 BP  (Be-10 spike + VADM collapse)
  - Last Glacial Maximum ~20,000 BP  (temperature minimum)
"""

import numpy as np
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


def _glacial_cycle(t: np.ndarray) -> np.ndarray:
    """Slow orbital forcing: 100-ka Milankovitch + 41-ka obliquity."""
    return (
        0.6 * np.sin(2 * np.pi * t / 100_000)
        + 0.3 * np.sin(2 * np.pi * t / 41_000 + 0.7)
    )


def _event_pulse(t: np.ndarray, centre: float, width: float, amp: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((t - centre) / width) ** 2)


def gisp2_d18o(path: Path):
    """GISP2 d18O: Age_yrBP   d18O(per mil)"""
    # 110,000 years, ~20-year resolution
    ages = np.arange(200, 110_001, 20, dtype=float)
    base = -35.0 + 4.0 * _glacial_cycle(ages)
    noise = RNG.normal(0, 0.3, len(ages))
    # Younger Dryas: sharp cold excursion at 12,900 BP
    sig = (
        _event_pulse(ages, 12_900, 500,  -3.5)   # YD cooling
      + _event_pulse(ages,  8_200, 300,  -1.2)   # 8.2 ka
      + _event_pulse(ages, 20_000, 2000, -2.0)   # LGM
    )
    d18o = base + sig + noise

    with open(path, "w") as f:
        f.write("# GISP2 δ18O — SYNTHETIC (pipeline validation)\n")
        f.write("# Stuiver et al. 1995 — format replica\n")
        f.write("# Age_yrBP   d18O_permil\n")
        f.write("-" * 40 + "\n")
        for a, v in zip(ages, d18o):
            f.write(f"{a:.1f}\t{v:.4f}\n")
    print(f"  [synth] gisp2_d18o.txt  ({len(ages)} records, {ages[0]:.0f}–{ages[-1]:.0f} BP)")


def vostok_deuterium(path: Path):
    """Vostok: Age_yrBP  Depth_m  dD  DeltaTs"""
    ages = np.arange(500, 420_001, 200, dtype=float)
    base_dts = 2.5 * _glacial_cycle(ages)
    noise = RNG.normal(0, 0.4, len(ages))
    sig = (
        _event_pulse(ages, 12_900, 600,  -3.0)
      + _event_pulse(ages, 20_000, 3000, -4.5)
      + _event_pulse(ages,  8_200, 400,  -1.0)
    )
    delta_ts = base_dts + sig + noise
    dD = -440 + 6.0 * delta_ts + RNG.normal(0, 2, len(ages))
    depth = np.linspace(100, 3600, len(ages))

    with open(path, "w") as f:
        f.write("# Vostok Ice Core — SYNTHETIC (pipeline validation)\n")
        f.write("# Petit et al. 1999 — format replica\n")
        f.write("# Age_yrBP  Depth_m  dD  DeltaTs\n")
        f.write("-" * 50 + "\n")
        for a, d, dd, dt in zip(ages, depth, dD, delta_ts):
            f.write(f"{a:.1f}\t{d:.2f}\t{dd:.3f}\t{dt:.4f}\n")
    print(f"  [synth] vostok_deuterium.txt  ({len(ages)} records, {ages[0]:.0f}–{ages[-1]:.0f} BP)")


def grip_be10(path: Path):
    """GRIP Be-10: Age_yrBP  Be10_atoms_per_gram"""
    ages = np.arange(300, 80_001, 100, dtype=float)
    base = 18_000 + 3000 * _glacial_cycle(ages)
    noise = RNG.normal(0, 800, len(ages))
    # Laschamp: VADM collapse → Be-10 spike at 41,000 BP
    # Younger Dryas: moderate geomagnetic weakening
    sig = (
        _event_pulse(ages, 41_000, 1500, 12_000)  # Laschamp
      + _event_pulse(ages, 12_900,  700,  4_000)  # YD
      + _event_pulse(ages,  8_200,  400,  1_500)  # 8.2 ka
    )
    be10 = base + sig + np.abs(noise)

    with open(path, "w") as f:
        f.write("# GRIP Be-10 — SYNTHETIC (pipeline validation)\n")
        f.write("# Yiou et al. 1997 — format replica\n")
        f.write("# Age_yrBP  Be10_atoms_g\n")
        f.write("-" * 40 + "\n")
        for a, v in zip(ages, be10):
            f.write(f"{a:.1f}\t{v:.2f}\n")
    print(f"  [synth] grip_be10.txt  ({len(ages)} records, {ages[0]:.0f}–{ages[-1]:.0f} BP)")


def sint2000_vadm(path: Path):
    """Sint-2000: Age_ka  VADM(10^22 A m^2)"""
    ages_ka = np.arange(0.5, 200.1, 0.5)   # ka BP
    ages_yr = ages_ka * 1000.0
    base = 8.0 + 2.5 * _glacial_cycle(ages_yr)
    noise = RNG.normal(0, 0.5, len(ages_ka))
    # Laschamp: VADM drops to ~2 at 41 ka
    # Mono Lake excursion ~34 ka: moderate dip
    sig = (
        _event_pulse(ages_yr, 41_000, 1200, -6.5)   # Laschamp
      + _event_pulse(ages_yr, 34_000,  800, -2.5)   # Mono Lake
      + _event_pulse(ages_yr, 12_900,  600, -1.5)   # YD weakening
    )
    vadm = np.clip(base + sig + noise, 1.5, 14.0)

    with open(path, "w") as f:
        f.write("# Sint-2000 VADM — SYNTHETIC (pipeline validation)\n")
        f.write("# Guyodo & Valet 1999 — format replica\n")
        f.write("# Age_ka  VADM_1e22_Am2\n")
        f.write("-" * 40 + "\n")
        for a, v in zip(ages_ka, vadm):
            f.write(f"{a:.2f}\t{v:.4f}\n")
    print(f"  [synth] sint2000_vadm.txt  ({len(ages_ka)} records, "
          f"{ages_ka[0]:.1f}–{ages_ka[-1]:.1f} ka BP)")


if __name__ == "__main__":
    print("Generating synthetic NOAA-format proxy files...")
    gisp2_d18o(RAW / "gisp2_d18o.txt")
    vostok_deuterium(RAW / "vostok_deuterium.txt")
    grip_be10(RAW / "grip_be10.txt")
    sint2000_vadm(RAW / "sint2000_vadm.txt")
    print("Done. Run: python src/cycle_detect/fetch_data.py --force")
