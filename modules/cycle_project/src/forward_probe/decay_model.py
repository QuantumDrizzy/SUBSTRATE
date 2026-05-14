# requirements: numpy, scipy, pandas, pyarrow, matplotlib

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from pathlib import Path
import matplotlib.pyplot as plt
import warnings

PROJECT_ROOT = Path(__file__).parents[2]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "aligned.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

THRESHOLD = 0.25       # Laschamp-level VADM_norm
WINDOW_YR = 5_000     # fit window: last 5000 yr of Sint-2000
FORECAST_YR = 10_000  # forward projection horizon
CI_SAMPLES = 20_000   # Monte-Carlo draws for confidence intervals


# ── model functions (t = years from now; past < 0, future > 0) ──────────────

def _linear(t, m, c):
    return m * t + c


def _exponential(t, A, k):
    return A * np.exp(k * t)


def _power_law(t, A, B, n):
    # y = A * (1 + B*t)^n; A ≈ current value, B > 0, n < 0 for decay
    inner = np.maximum(1.0 + B * t, 1e-9)
    return A * inner ** n


# ── threshold crossing helpers ───────────────────────────────────────────────

def _linear_cross(params):
    m, c = params
    if m >= 0:
        return np.inf
    return (THRESHOLD - c) / m


def _exp_cross(params):
    A, k = params
    if A <= THRESHOLD or k >= 0:
        return np.inf
    return np.log(THRESHOLD / A) / k


def _power_cross(params):
    A, B, n = params
    if A <= THRESHOLD or B <= 0:
        return np.inf
    ratio = THRESHOLD / A
    if ratio <= 0:
        return np.inf
    try:
        return (ratio ** (1.0 / n) - 1.0) / B
    except Exception:
        return np.inf


def _ci(popt, pcov, cross_fn, n_samples=CI_SAMPLES):
    """95% CI on threshold-crossing time via parameter sampling."""
    try:
        # Perturb parameters with their covariance
        samples = np.random.default_rng(0).multivariate_normal(popt, pcov, n_samples)
        crossings = np.array([cross_fn(s) for s in samples])
        crossings = crossings[np.isfinite(crossings) & (crossings > 0) & (crossings < 2e5)]
        if len(crossings) < 100:
            return np.nan, np.nan
        return float(np.percentile(crossings, 2.5)), float(np.percentile(crossings, 97.5))
    except Exception:
        return np.nan, np.nan


# ── main ─────────────────────────────────────────────────────────────────────

def run_decay_model(parquet_path=None, output_dir=None):
    parquet_path = Path(parquet_path or PARQUET_PATH)
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[DECAY] Loading Sint-2000 data...")
    df = pd.read_parquet(parquet_path).sort_values("age_bp").reset_index(drop=True)

    # Use last WINDOW_YR years (smallest age_bp values in the sorted frame)
    recent = df[df["age_bp"] <= WINDOW_YR].copy()
    # Convert to "years from now" convention: τ = −age_bp
    tau_hist = -recent["age_bp"].values.astype(float)   # in [−5000, −500]
    vadm_hist = recent["sint2000_vadm_norm"].values.astype(float)
    print(f"[DECAY] Fitting on {len(recent)} points (last {WINDOW_YR:,} yr)")

    # ── fit ──────────────────────────────────────────────────────────────────
    # Initial guess from linear regression
    slope_init, intercept_init, *_ = scipy.stats.linregress(tau_hist, vadm_hist)
    y0 = float(np.interp(0.0, tau_hist[::-1], vadm_hist[::-1]))  # approx. current value

    results = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Linear
        try:
            popt_lin, pcov_lin = scipy.optimize.curve_fit(
                _linear, tau_hist, vadm_hist,
                p0=[slope_init, intercept_init],
                maxfev=5_000,
            )
            tau_cross_lin = _linear_cross(popt_lin)
            ci_lin = _ci(popt_lin, pcov_lin, _linear_cross)
            results["linear"] = dict(popt=popt_lin, pcov=pcov_lin,
                                     threshold=tau_cross_lin, ci=ci_lin)
        except Exception as exc:
            print(f"[DECAY] Linear fit failed: {exc}")
            results["linear"] = dict(threshold=np.nan, ci=(np.nan, np.nan))

        # Exponential
        k0 = np.log(vadm_hist[-1] / vadm_hist[0]) / (tau_hist[-1] - tau_hist[0] + 1e-9)
        try:
            popt_exp, pcov_exp = scipy.optimize.curve_fit(
                _exponential, tau_hist, vadm_hist,
                p0=[y0 if y0 > 0 else 0.8, k0 if np.isfinite(k0) else -1e-4],
                bounds=([0.01, -1.0], [2.0, 0.0]),
                maxfev=5_000,
            )
            tau_cross_exp = _exp_cross(popt_exp)
            ci_exp = _ci(popt_exp, pcov_exp, _exp_cross)
            results["exponential"] = dict(popt=popt_exp, pcov=pcov_exp,
                                          threshold=tau_cross_exp, ci=ci_exp)
        except Exception as exc:
            print(f"[DECAY] Exponential fit failed: {exc}")
            results["exponential"] = dict(threshold=np.nan, ci=(np.nan, np.nan))

        # Power-law: y = A*(1 + B*t)^n
        try:
            popt_pow, pcov_pow = scipy.optimize.curve_fit(
                _power_law, tau_hist, vadm_hist,
                p0=[y0 if y0 > 0 else 0.8, 1e-4, -0.5],
                bounds=([0.01, 1e-7, -10.0], [2.0, 0.1, -1e-3]),
                maxfev=10_000,
            )
            tau_cross_pow = _power_cross(popt_pow)
            ci_pow = _ci(popt_pow, pcov_pow, _power_cross)
            results["power_law"] = dict(popt=popt_pow, pcov=pcov_pow,
                                        threshold=tau_cross_pow, ci=ci_pow)
        except Exception as exc:
            print(f"[DECAY] Power-law fit failed: {exc}")
            results["power_law"] = dict(threshold=np.nan, ci=(np.nan, np.nan))

    # Instrumental (IGRF 1840–2026): constant exponential rate -4.8%/century
    IGRF_RATE = -0.00048  # per year, negative = decay
    IGRF_V0   = 0.80      # current normalized VADM (approx)
    IGRF_CI_RATE_LO = -0.00060  # pessimistic: -6%/century (SAA expansion scenario)
    IGRF_CI_RATE_HI = -0.00030  # optimistic:  -3%/century (partial recovery)

    def _igrf_cross(rate, v0=IGRF_V0, threshold=THRESHOLD):
        if rate >= 0:
            return np.nan
        return np.log(threshold / v0) / rate

    igrf_cross    = _igrf_cross(IGRF_RATE)
    igrf_cross_lo = _igrf_cross(IGRF_CI_RATE_LO)  # faster rate → sooner crossing
    igrf_cross_hi = _igrf_cross(IGRF_CI_RATE_HI)  # slower rate → later crossing

    results["instrumental"] = dict(
        threshold = igrf_cross,
        ci        = (igrf_cross_lo, igrf_cross_hi),
        popt      = None,
        pcov      = None,
    )

    # ── report ───────────────────────────────────────────────────────────────
    MAX_PLAUSIBLE_YR = 1_000_000  # anything beyond 1 Myr is model divergence, not physics
    thresholds = {}
    for name, res in results.items():
        t = res["threshold"]
        lo, hi = res.get("ci", (np.nan, np.nan))
        # Clamp implausibly large values (positive-exponent fit = growing field)
        if not np.isfinite(t) or abs(t) > MAX_PLAUSIBLE_YR:
            t = np.nan
        if not np.isfinite(lo) or abs(lo) > MAX_PLAUSIBLE_YR:
            lo = np.nan
        if not np.isfinite(hi) or abs(hi) > MAX_PLAUSIBLE_YR:
            hi = np.nan
        mid = (hi - lo) / 2 if np.isfinite(lo) and np.isfinite(hi) else np.nan
        thresholds[name] = {
            "threshold_yr": float(t) if np.isfinite(t) else None,
            "ci_lo": float(lo) if np.isfinite(lo) else None,
            "ci_hi": float(hi) if np.isfinite(hi) else None,
        }
        ci_str = f"±{mid:,.0f}" if np.isfinite(mid) else "CI n/a"
        t_str = f"{t:,.0f}" if np.isfinite(t) else "n/a (field not decaying in this model)"
        print(f"[DECAY] Under {name:12s} decay: threshold in {t_str} {ci_str} years (95% CI)")

    # ── plot ─────────────────────────────────────────────────────────────────
    tau_future = np.linspace(0, FORECAST_YR, 500)
    tau_plot_hist = np.linspace(tau_hist.min(), 0, 200)

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="#111111")
    ax.set_facecolor("#1a1a2e")

    ax.scatter(tau_hist / 1000, vadm_hist, s=14, color="#aaddff", alpha=0.7,
               label="Sint-2000 (last 5 kyr)", zorder=5)

    colours = {"linear": "#ff6644", "exponential": "#44ff88", "power_law": "#ffcc00"}
    fn_map = {"linear": _linear, "exponential": _exponential, "power_law": _power_law}

    for name, res in results.items():
        if res.get("popt") is None:
            continue
        fn = fn_map[name]
        popt = res["popt"]
        pcov = res.get("pcov", np.zeros((len(popt), len(popt))))
        col = colours[name]

        y_hist = fn(tau_plot_hist, *popt)
        y_fut = fn(tau_future, *popt)
        ax.plot(tau_plot_hist / 1000, y_hist, color=col, lw=1.2, alpha=0.6)
        ax.plot(tau_future / 1000, y_fut, color=col, lw=2.0, label=name.replace("_", "-"))

        # CI band via Monte-Carlo
        try:
            draws = np.random.default_rng(42).multivariate_normal(popt, pcov, 300)
            y_band = np.array([fn(tau_future, *d) for d in draws])
            y_band = np.clip(y_band, 0, 2)
            ax.fill_between(
                tau_future / 1000,
                np.percentile(y_band, 2.5, axis=0),
                np.percentile(y_band, 97.5, axis=0),
                color=col, alpha=0.12,
            )
        except Exception:
            pass

    # Instrumental projection
    tau_instr = np.linspace(0, FORECAST_YR, 500)
    vadm_instr = IGRF_V0 * np.exp(IGRF_RATE * tau_instr)
    ax.plot(tau_instr / 1000, vadm_instr, color="#ff6600", lw=2.5, ls="-",
            label=f"instrumental (IGRF, −4.8%/c) → ~{igrf_cross:,.0f} yr")
    # CI band
    vadm_instr_lo = IGRF_V0 * np.exp(IGRF_CI_RATE_LO * tau_instr)
    vadm_instr_hi = IGRF_V0 * np.exp(IGRF_CI_RATE_HI * tau_instr)
    ax.fill_between(tau_instr / 1000, vadm_instr_lo, vadm_instr_hi,
                    color="#ff6600", alpha=0.15)

    ax.axhline(THRESHOLD, color="white", lw=1.0, ls=":", alpha=0.7, label=f"Laschamp threshold ({THRESHOLD})")
    ax.axvline(0, color="#888888", lw=0.8, ls="--", alpha=0.5, label="Present")

    ax.set_xlabel("Years from now (kyr; negative = past)", color="white", fontsize=11)
    ax.set_ylabel("VADM_norm", color="white", fontsize=11)
    ax.set_title("VADM Decay Forecast — 4-Model Projection", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(fontsize=9, facecolor="#222", labelcolor="white")
    ax.set_xlim(tau_hist.min() / 1000, FORECAST_YR / 1000)
    ax.set_ylim(0, max(vadm_hist.max() * 1.1, 0.5))
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = output_dir / "vadm_forecast.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"[DECAY] Saved → {out_path}")
    plt.show()
    plt.close()

    return {"thresholds": thresholds, "results": results}


if __name__ == "__main__":
    run_decay_model()
