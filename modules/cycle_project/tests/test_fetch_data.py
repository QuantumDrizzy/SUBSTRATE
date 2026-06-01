"""
Tests for cycle_detect.fetch_data — 5-proxy parser suite.

Each parser returns short column names matching DATASETS[key]["value_col"]:
  gisp2_d18o      → "d18o"
  vostok_deuterium → "delta_ts"    (col 0 = Depth, col 1 = Ice_age, col 3 = ΔTs)
  vostok_co2      → "co2_ppmv"    (col 0 = Depth, col 2 = Air_age, col 3 = CO₂)
  grip_be10       → "be10"
  sint2000        → "vadm"         (col 0 = Age_ka — converted to yr BP)

No network required.  Synthetic NOAA-format data written to tmp_path.
"""

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from cycle_detect.fetch_data import (
    parse_gisp2_d18o,
    parse_vostok_deuterium,
    parse_vostok_co2,
    parse_grip_be10,
    parse_sint2000,
    align_proxies,
    z_score,
    resample_to_grid,
    DATASETS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(content))
    return p


# ── GISP2 δ¹⁸O parser ────────────────────────────────────────────────────────
# File format: col 0 = Age_yrBP, col 1 = d18O

GISP2_SAMPLE = """\
    # GISP2 d18O — synthetic test
    # Age_yrBP   d18O_permil
    ----------------------------------------
    200.0\t-34.5
    400.0\t-35.1
    600.0\t-33.8
    800.0\t-36.2
    1000.0\t-34.9
    12900.0\t-40.1
    41000.0\t-37.5
"""


class TestParseGISP2:
    def test_returns_dataframe(self, tmp_path):
        p = _write(tmp_path, "gisp2.txt", GISP2_SAMPLE)
        df = parse_gisp2_d18o(p)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, tmp_path):
        """Parser returns short column names: age_bp, d18o."""
        p = _write(tmp_path, "gisp2.txt", GISP2_SAMPLE)
        df = parse_gisp2_d18o(p)
        assert "age_bp" in df.columns
        assert "d18o" in df.columns       # NOT 'gisp2_d18o'

    def test_age_positive(self, tmp_path):
        p = _write(tmp_path, "gisp2.txt", GISP2_SAMPLE)
        df = parse_gisp2_d18o(p)
        assert (df["age_bp"] > 0).all()

    def test_sorted_ascending(self, tmp_path):
        p = _write(tmp_path, "gisp2.txt", GISP2_SAMPLE)
        df = parse_gisp2_d18o(p)
        assert df["age_bp"].is_monotonic_increasing

    def test_d18o_range_physical(self, tmp_path):
        """Glacial d18O typically -45 to -28 permil for GISP2."""
        p = _write(tmp_path, "gisp2.txt", GISP2_SAMPLE)
        df = parse_gisp2_d18o(p)
        assert df["d18o"].between(-50, -20).all()

    def test_no_nans_in_values(self, tmp_path):
        p = _write(tmp_path, "gisp2.txt", GISP2_SAMPLE)
        df = parse_gisp2_d18o(p)
        assert df["d18o"].notna().all()


# ── Vostok ΔTs parser ─────────────────────────────────────────────────────────
# Real file format (deutnat.txt): Depth_m  Ice_age_yrBP  dD  DeltaTs
# Parser uses col 1 (ice_age) as age_bp and col 3 (DeltaTs) as delta_ts.

VOSTOK_DEUT_SAMPLE = """\
    # Vostok ice core — synthetic
    # Depth_m  Ice_age_yrBP  dD  DeltaTs
    --------------------------------------------------
    138.0\t500.0\t-440.2\t0.5
    280.0\t1000.0\t-442.1\t0.2
    350.0\t5000.0\t-450.3\t-1.1
    800.0\t12900.0\t-465.0\t-4.2
    1400.0\t20000.0\t-480.0\t-7.5
    2200.0\t41000.0\t-458.0\t-3.0
"""


class TestParseVostokDeuterium:
    def test_returns_dataframe(self, tmp_path):
        p = _write(tmp_path, "vostok_d.txt", VOSTOK_DEUT_SAMPLE)
        df = parse_vostok_deuterium(p)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, tmp_path):
        """Parser returns: age_bp (ice_age col), delta_ts (ΔTs col)."""
        p = _write(tmp_path, "vostok_d.txt", VOSTOK_DEUT_SAMPLE)
        df = parse_vostok_deuterium(p)
        assert "age_bp" in df.columns
        assert "delta_ts" in df.columns   # NOT 'vostok_deuterium'

    def test_uses_ice_age_not_depth(self, tmp_path):
        """Parser must use col 1 (ice_age=500) not col 0 (depth=138)."""
        p = _write(tmp_path, "vostok_d.txt", VOSTOK_DEUT_SAMPLE)
        df = parse_vostok_deuterium(p)
        assert abs(df["age_bp"].iloc[0] - 500.0) < 1.0, (
            f"Expected ice_age=500, got {df['age_bp'].iloc[0]} "
            f"(would be depth=138 if wrong column)"
        )

    def test_age_positive_sorted(self, tmp_path):
        p = _write(tmp_path, "vostok_d.txt", VOSTOK_DEUT_SAMPLE)
        df = parse_vostok_deuterium(p)
        assert (df["age_bp"] > 0).all()
        assert df["age_bp"].is_monotonic_increasing

    def test_delta_ts_range_physical(self, tmp_path):
        """ΔTs relative to present: glacial troughs ~-9°C, holocene ~+3°C."""
        p = _write(tmp_path, "vostok_d.txt", VOSTOK_DEUT_SAMPLE)
        df = parse_vostok_deuterium(p)
        assert df["delta_ts"].between(-12, 5).all()

    def test_no_nans(self, tmp_path):
        p = _write(tmp_path, "vostok_d.txt", VOSTOK_DEUT_SAMPLE)
        df = parse_vostok_deuterium(p)
        assert df["delta_ts"].notna().all()


# ── Vostok CO₂ parser ─────────────────────────────────────────────────────────
# Real file format (co2nat.txt): Depth_m  Ice_age_yrBP  Air_age_yrBP  CO2_ppmv
# Parser uses col 2 (air_age) as age_bp and col 3 (CO₂) as co2_ppmv.

VOSTOK_CO2_SAMPLE = """\
    # Vostok CO2 — synthetic (Petit 1999)
    # Depth_m  Ice_age_yrBP  Air_age_yrBP  CO2_ppmv
    --------------------------------------------------
    160.0\t1200.0\t800.0\t280.5
    350.0\t3000.0\t2400.0\t272.1
    700.0\t6000.0\t5200.0\t265.3
    1500.0\t12000.0\t11200.0\t240.0
    2100.0\t20000.0\t18800.0\t185.0
    2800.0\t41000.0\t39500.0\t215.0
    3350.0\t74000.0\t72000.0\t188.0
"""


class TestParseVostokCO2:
    def test_returns_dataframe(self, tmp_path):
        p = _write(tmp_path, "vostok_co2.txt", VOSTOK_CO2_SAMPLE)
        df = parse_vostok_co2(p)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, tmp_path):
        """Parser returns: age_bp (air_age col), co2_ppmv."""
        p = _write(tmp_path, "vostok_co2.txt", VOSTOK_CO2_SAMPLE)
        df = parse_vostok_co2(p)
        assert "age_bp" in df.columns
        assert "co2_ppmv" in df.columns   # NOT 'vostok_co2'

    def test_uses_air_age_not_ice_age(self, tmp_path):
        """Parser must use col 2 (air_age=800), not col 1 (ice_age=1200)."""
        p = _write(tmp_path, "vostok_co2.txt", VOSTOK_CO2_SAMPLE)
        df = parse_vostok_co2(p)
        assert abs(df["age_bp"].iloc[0] - 800.0) < 1.0, (
            f"Expected air_age=800, got {df['age_bp'].iloc[0]} "
            f"(would be 1200 if ice_age, 160 if depth)"
        )

    def test_co2_range_physical(self, tmp_path):
        """Glacial CO₂: ~180-200 ppmv; interglacial: ~260-290 ppmv."""
        p = _write(tmp_path, "vostok_co2.txt", VOSTOK_CO2_SAMPLE)
        df = parse_vostok_co2(p)
        assert df["co2_ppmv"].between(150, 320).all(), (
            f"CO₂ out of physical range: {df['co2_ppmv'].tolist()}"
        )

    def test_glacial_co2_lower_than_interglacial(self, tmp_path):
        """LGM CO₂ (~185 ppmv) must be below Holocene (~280 ppmv)."""
        p = _write(tmp_path, "vostok_co2.txt", VOSTOK_CO2_SAMPLE)
        df = parse_vostok_co2(p)
        holocene = df[df["age_bp"] < 3000]["co2_ppmv"].mean()
        lgm      = df[(df["age_bp"] > 15_000) & (df["age_bp"] < 25_000)]["co2_ppmv"].mean()
        assert lgm < holocene, (
            f"LGM CO₂ ({lgm:.1f}) should be < Holocene ({holocene:.1f})"
        )

    def test_sorted_ascending(self, tmp_path):
        p = _write(tmp_path, "vostok_co2.txt", VOSTOK_CO2_SAMPLE)
        df = parse_vostok_co2(p)
        assert df["age_bp"].is_monotonic_increasing

    def test_no_nans(self, tmp_path):
        p = _write(tmp_path, "vostok_co2.txt", VOSTOK_CO2_SAMPLE)
        df = parse_vostok_co2(p)
        assert df["co2_ppmv"].notna().all()


# ── GRIP Be-10 parser ─────────────────────────────────────────────────────────
# File format: col 0 = Age_yrBP, col 1 = Be10

GRIP_BE10_SAMPLE = """\
    # GRIP Be-10 — synthetic (Muscheler 2004)
    # Age_yrBP  Be10_atoms_g
    ----------------------------------------
    300.0\t17500.0
    500.0\t18200.0
    1000.0\t16900.0
    12900.0\t22000.0
    41000.0\t31500.0
    55000.0\t19800.0
"""


class TestParseGRIPBe10:
    def test_returns_dataframe(self, tmp_path):
        p = _write(tmp_path, "grip_be10.txt", GRIP_BE10_SAMPLE)
        df = parse_grip_be10(p)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, tmp_path):
        """Parser returns: age_bp, be10."""
        p = _write(tmp_path, "grip_be10.txt", GRIP_BE10_SAMPLE)
        df = parse_grip_be10(p)
        assert "age_bp" in df.columns
        assert "be10" in df.columns    # NOT 'grip_be10'

    def test_age_positive_sorted(self, tmp_path):
        p = _write(tmp_path, "grip_be10.txt", GRIP_BE10_SAMPLE)
        df = parse_grip_be10(p)
        assert (df["age_bp"] > 0).all()
        assert df["age_bp"].is_monotonic_increasing

    def test_be10_positive(self, tmp_path):
        """Be-10 is a count; must be positive."""
        p = _write(tmp_path, "grip_be10.txt", GRIP_BE10_SAMPLE)
        df = parse_grip_be10(p)
        assert (df["be10"] > 0).all()

    def test_laschamp_spike(self, tmp_path):
        """Laschamp (~41 ka) should show elevated Be-10 vs Holocene baseline."""
        p = _write(tmp_path, "grip_be10.txt", GRIP_BE10_SAMPLE)
        df = parse_grip_be10(p)
        holocene = df[df["age_bp"] < 2000]["be10"].mean()
        laschamp = df[(df["age_bp"] > 38_000) & (df["age_bp"] < 44_000)]["be10"].mean()
        assert laschamp > holocene, (
            f"Laschamp Be-10 ({laschamp:.0f}) should exceed Holocene ({holocene:.0f})"
        )


# ── Sint-2000 VADM parser ─────────────────────────────────────────────────────
# File format: col 0 = Age_ka (kiloyears), col 1 = VADM
# Parser converts ka → yr BP.

SINT2000_SAMPLE = """\
    # Sint-2000 VADM — synthetic (Valet 2005)
    # Age_ka  VADM_1e22_Am2
    ----------------------------------------
    0.50\t9.50
    1.00\t9.20
    5.00\t8.80
    12.90\t7.50
    34.00\t6.00
    41.00\t2.50
    74.00\t8.20
"""


class TestParseSint2000:
    def test_returns_dataframe(self, tmp_path):
        p = _write(tmp_path, "sint2000.txt", SINT2000_SAMPLE)
        df = parse_sint2000(p)
        assert isinstance(df, pd.DataFrame)

    def test_columns(self, tmp_path):
        """Parser returns: age_bp (converted from ka), vadm."""
        p = _write(tmp_path, "sint2000.txt", SINT2000_SAMPLE)
        df = parse_sint2000(p)
        assert "age_bp" in df.columns
        assert "vadm" in df.columns    # NOT 'sint2000'

    def test_age_converted_ka_to_yr(self, tmp_path):
        """Input is in ka; parser must convert to yr BP."""
        p = _write(tmp_path, "sint2000.txt", SINT2000_SAMPLE)
        df = parse_sint2000(p)
        # First age was 0.50 ka → 500 yr
        assert abs(df["age_bp"].iloc[0] - 500.0) < 1.0, (
            f"Expected 500 yr, got {df['age_bp'].iloc[0]}"
        )

    def test_age_sorted_ascending(self, tmp_path):
        p = _write(tmp_path, "sint2000.txt", SINT2000_SAMPLE)
        df = parse_sint2000(p)
        assert df["age_bp"].is_monotonic_increasing

    def test_vadm_range_physical(self, tmp_path):
        """VADM: typical 4–12 × 10²² A m²; Laschamp can drop to ~1–2."""
        p = _write(tmp_path, "sint2000.txt", SINT2000_SAMPLE)
        df = parse_sint2000(p)
        assert df["vadm"].between(0.5, 14.0).all()

    def test_laschamp_vadm_minimum(self, tmp_path):
        """Laschamp VADM (~2.5) must be below Holocene mean (~9)."""
        p = _write(tmp_path, "sint2000.txt", SINT2000_SAMPLE)
        df = parse_sint2000(p)
        holocene = df[df["age_bp"] < 5000]["vadm"].mean()
        laschamp = df[(df["age_bp"] > 38_000) & (df["age_bp"] < 44_000)]["vadm"].mean()
        assert laschamp < holocene, (
            f"Laschamp VADM ({laschamp:.2f}) should be < Holocene ({holocene:.2f})"
        )


# ── Utility functions ─────────────────────────────────────────────────────────

class TestZScore:
    def test_mean_near_zero(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = z_score(arr)
        assert abs(z.mean()) < 1e-10

    def test_std_near_one(self):
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = z_score(arr)
        assert abs(z.std() - 1.0) < 1e-10

    def test_constant_array_returns_zeros(self):
        arr = np.full(10, 42.0)
        z = z_score(arr)
        np.testing.assert_array_equal(z, np.zeros(10))

    def test_preserves_length(self):
        arr = np.random.default_rng(0).normal(size=100)
        assert len(z_score(arr)) == 100


class TestResampleToGrid:
    def test_output_length_matches_grid(self):
        df = pd.DataFrame({"age_bp": np.arange(0, 10_001, 100, dtype=float),
                           "val":    np.random.default_rng(1).normal(size=101)})
        grid = np.arange(0, 8001, 200, dtype=float)
        out = resample_to_grid(df, "val", grid)
        assert len(out) == len(grid)

    def test_output_is_numpy_array(self):
        df = pd.DataFrame({"age_bp": np.arange(0, 5001, 500, dtype=float),
                           "val":    np.ones(11)})
        grid = np.arange(0, 4001, 500, dtype=float)
        out = resample_to_grid(df, "val", grid)
        assert isinstance(out, np.ndarray)

    def test_interpolation_monotone_signal(self):
        """Linear signal should survive resampling with low error."""
        ages = np.arange(0, 10_001, 100, dtype=float)
        vals = ages / 10_000.0            # 0.0 → 1.0 linear
        df = pd.DataFrame({"age_bp": ages, "val": vals})
        grid = np.arange(0, 10_001, 250, dtype=float)
        out = resample_to_grid(df, "val", grid)
        expected = grid / 10_000.0
        np.testing.assert_allclose(out, expected, atol=0.05)


# ── align_proxies integration ─────────────────────────────────────────────────

class TestAlignProxies:
    """
    End-to-end: pass 5 proxy DataFrames through align_proxies().

    Keys must match DATASETS entries; value columns must match
    DATASETS[key]["value_col"] (short names: d18o, delta_ts, co2_ppmv, be10, vadm).
    """

    @pytest.fixture(scope="class")
    def five_proxy_dfs(self):
        rng = np.random.default_rng(77)
        ages = np.arange(200, 80_001, 100, dtype=float)
        n = len(ages)

        dfs = {
            "gisp2_d18o":       pd.DataFrame({"age_bp": ages, "d18o":     rng.normal(-35, 1, n)}),
            "vostok_deuterium": pd.DataFrame({"age_bp": ages, "delta_ts": rng.normal(0, 2, n)}),
            "vostok_co2":       pd.DataFrame({"age_bp": ages, "co2_ppmv": rng.uniform(180, 290, n)}),
            "grip_be10":        pd.DataFrame({"age_bp": ages, "be10":     rng.normal(18000, 3000, n)}),
            "sint2000":         pd.DataFrame({"age_bp": ages, "vadm":     rng.normal(8, 2, n)}),
        }
        return dfs

    def test_returns_dataframe(self, five_proxy_dfs):
        aligned = align_proxies(five_proxy_dfs, t_max=80_000)
        assert isinstance(aligned, pd.DataFrame)

    def test_has_age_bp_column(self, five_proxy_dfs):
        aligned = align_proxies(five_proxy_dfs, t_max=80_000)
        assert "age_bp" in aligned.columns

    def test_has_all_five_norm_columns(self, five_proxy_dfs):
        aligned = align_proxies(five_proxy_dfs, t_max=80_000)
        expected = [
            "gisp2_d18o_norm",
            "vostok_deuterium_norm",
            "vostok_co2_norm",
            "grip_be10_norm",
            "sint2000_norm",
        ]
        for col in expected:
            assert col in aligned.columns, f"Missing column: {col}"

    def test_norm_columns_zero_mean(self, five_proxy_dfs):
        aligned = align_proxies(five_proxy_dfs, t_max=80_000)
        for col in ["gisp2_d18o_norm", "vostok_co2_norm", "sint2000_norm"]:
            mean = aligned[col].mean()
            assert abs(mean) < 0.1, f"{col} mean={mean:.4f} (expected ~0)"

    def test_age_within_t_max(self, five_proxy_dfs):
        aligned = align_proxies(five_proxy_dfs, t_max=80_000)
        assert (aligned["age_bp"] <= 80_000).all()

    def test_age_sorted(self, five_proxy_dfs):
        aligned = align_proxies(five_proxy_dfs, t_max=80_000)
        assert aligned["age_bp"].is_monotonic_increasing


# ── DATASETS registry ─────────────────────────────────────────────────────────

class TestDatasetsRegistry:
    def test_has_five_entries(self):
        assert len(DATASETS) == 5, f"Expected 5 datasets, got {len(DATASETS)}: {list(DATASETS)}"

    def test_all_have_required_keys(self):
        required = {"urls", "value_col", "label"}
        for key, meta in DATASETS.items():
            missing = required - meta.keys()
            assert not missing, f"DATASETS['{key}'] missing: {missing}"

    def test_urls_are_lists(self):
        for key, meta in DATASETS.items():
            assert isinstance(meta["urls"], list), f"DATASETS['{key}']['urls'] must be a list"
            assert len(meta["urls"]) >= 1

    def test_value_cols_are_strings(self):
        for key, meta in DATASETS.items():
            assert isinstance(meta["value_col"], str), f"DATASETS['{key}']['value_col'] not a str"

    def test_vostok_co2_present(self):
        assert "vostok_co2" in DATASETS, "vostok_co2 missing from DATASETS registry"

    def test_all_labels_non_empty(self):
        for key, meta in DATASETS.items():
            assert meta["label"], f"DATASETS['{key}']['label'] is empty"

    def test_value_col_names_correct(self):
        expected = {
            "gisp2_d18o": "d18o",
            "vostok_deuterium": "delta_ts",
            "vostok_co2": "co2_ppmv",
            "grip_be10": "be10",
            "sint2000": "vadm",
        }
        for key, col in expected.items():
            assert DATASETS[key]["value_col"] == col
