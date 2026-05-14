"""
Shared fixtures for cycle_project test suite.
Run from the cycle_project root:
    pytest tests/ -v
    pytest tests/ -v --tb=short -q      # quiet
    pytest tests/test_cycle_detect.py   # single module
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pytest

# ── Synthetic proxy data (no network or files needed) ────────────────────────

@pytest.fixture(scope="session")
def syn_ages():
    """801 time points, 0–80,000 BP, 100-year steps."""
    return np.arange(0, 80_001, 100, dtype=float)

@pytest.fixture(scope="session")
def syn_proxy_df(syn_ages):
    """Aligned 4-proxy DataFrame matching fetch_data output schema."""
    rng = np.random.default_rng(42)
    n = len(syn_ages)
    df = pd.DataFrame({
        "age_bp": syn_ages,
        "gisp2_d18o":   rng.normal(0, 1, n),
        "vostok_dd":    rng.normal(0, 1, n),
        "grip_be10":    rng.normal(0, 1, n),
        "sint2000_vadm": rng.normal(0, 1, n),
    })
    # Embed synthetic Laschamp signal at ~41,000 BP
    laschamp_idx = np.argmin(np.abs(syn_ages - 41_000))
    window = slice(laschamp_idx - 5, laschamp_idx + 5)
    df.loc[window, "sint2000_vadm"] -= 3.0   # VADM drop
    df.loc[window, "grip_be10"]     += 4.0   # Be-10 spike
    return df
