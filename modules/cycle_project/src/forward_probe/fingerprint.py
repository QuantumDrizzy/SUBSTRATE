# requirements: numpy, scipy, pandas, pyarrow, sklearn, matplotlib

import json
import numpy as np
import pandas as pd
import scipy.stats
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).parents[2]
PARQUET_PATH = PROJECT_ROOT / "data" / "processed" / "aligned.parquet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

EXCURSION_BP = [51_000, 41_000, 34_000, 23_000, 19_000]  # known events
PRE_WINDOW_YR = 3_000   # years before each excursion to sample
STABLE_BUFFER_YR = 5_000   # exclusion zone around excursions for stable windows
N_STABLE = 20           # target number of stable windows
PROXY_COLS = ["gisp2_d18o_norm", "vostok_dd_norm", "grip_be10_norm", "sint2000_vadm_norm"]
PROB_THRESHOLD = 0.5

PROB_STATUS = {
    lambda p: p >= 0.7: "ALERT",
    lambda p: p >= 0.4: "WATCH",
}


def _status(prob):
    if prob >= 0.7:
        return "ALERT"
    if prob >= 0.4:
        return "WATCH"
    return "STABLE"


def _extract_features(df_window):
    """Return 1-D feature vector from a window DataFrame."""
    feats = []
    for col in PROXY_COLS:
        vals = df_window[col].dropna().values
        if len(vals) < 3:
            feats.extend([np.nan] * 5)
            continue
        slope, *_ = scipy.stats.linregress(np.arange(len(vals)), vals)
        feats.extend([
            float(np.mean(vals)),
            float(np.std(vals)),
            float(slope),
            float(np.min(vals)),
            float(scipy.stats.skew(vals)),
        ])
    return np.array(feats, dtype=float)


def _window_slice(df, age_lo, age_hi):
    """Return rows where age_lo <= age_bp <= age_hi, sorted ascending."""
    return df[(df["age_bp"] >= age_lo) & (df["age_bp"] <= age_hi)].sort_values("age_bp")


def run_fingerprint(parquet_path=None, output_dir=None):
    parquet_path = Path(parquet_path or PARQUET_PATH)
    output_dir = Path(output_dir or OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load anomaly-derived excursion timestamps if available
    anomaly_json_path = output_dir / "anomaly_scores.json"
    if anomaly_json_path.exists():
        try:
            with open(anomaly_json_path) as fh:
                anom_data = json.load(fh)
            # Filter to geomagnetically plausible excursions only:
            # Holocene (<15,000 BP) has no documented reversals/excursions.
            all_bp = [int(a["age_bp"]) for a in anom_data.get("anomalies", [])]
            excursion_bp = [bp for bp in all_bp if bp >= 15_000]
            if excursion_bp:
                skipped = len(all_bp) - len(excursion_bp)
                print(f"[FINGERPRINT] Loaded {len(excursion_bp)} excursion timestamps from anomaly_scores.json"
                      + (f" ({skipped} Holocene anomalies filtered)" if skipped else ""))
            else:
                raise ValueError("empty anomalies list after filtering")
        except Exception as e:
            print(f"[FINGERPRINT] Warning: could not parse anomaly_scores.json ({e}) — using hardcoded list")
            excursion_bp = EXCURSION_BP
    else:
        print("[FINGERPRINT] Warning: anomaly_scores.json not found — using hardcoded excursion list")
        excursion_bp = EXCURSION_BP

    print("[FINGERPRINT] Loading proxy data...")
    df = pd.read_parquet(parquet_path).sort_values("age_bp").reset_index(drop=True)
    age = df["age_bp"].values

    # ── pre-excursion windows ─────────────────────────────────────────────────
    # "3000 yr immediately before excursion at E" means age_bp in [E, E+3000]
    # (the 3000 yr leading up to the event, further back in time = larger age_bp)
    print("[FINGERPRINT] Extracting pre-excursion windows...")
    X_pre, valid_excursions = [], []
    for E in excursion_bp:
        win = _window_slice(df, E, E + PRE_WINDOW_YR)
        if len(win) < 5:
            print(f"[FINGERPRINT]   Skipping excursion at {E:,} BP — too few rows ({len(win)})")
            continue
        feats = _extract_features(win)
        if not np.any(np.isnan(feats)):
            X_pre.append(feats)
            valid_excursions.append(E)
        else:
            print(f"[FINGERPRINT]   Excursion at {E:,} BP has NaN features — using mean imputation")
            feats = np.nan_to_num(feats, nan=0.5)
            X_pre.append(feats)
            valid_excursions.append(E)
    X_pre = np.array(X_pre)
    n_pre = len(X_pre)
    print(f"[FINGERPRINT] {n_pre} pre-excursion windows extracted")

    # ── stable windows ───────────────────────────────────────────────────────
    print("[FINGERPRINT] Sampling stable windows...")
    excl = set()
    for E in excursion_bp:
        excl.update(range(E - STABLE_BUFFER_YR, E + STABLE_BUFFER_YR + 1, 100))

    # Candidate start ages: windows of PRE_WINDOW_YR that fit in data
    age_min, age_max = int(age.min()), int(age.max()) - PRE_WINDOW_YR
    rng = np.random.default_rng(42)
    candidates = np.arange(
        max(age_min, 1000),
        age_max,
        PRE_WINDOW_YR // 2,  # stride = half window
    )
    rng.shuffle(candidates)

    X_stable = []
    for start in candidates:
        if len(X_stable) >= N_STABLE:
            break
        end = start + PRE_WINDOW_YR
        # Check not within buffer of any excursion
        if any(abs(start - E) < STABLE_BUFFER_YR or abs(end - E) < STABLE_BUFFER_YR
               for E in excursion_bp):
            continue
        win = _window_slice(df, start, end)
        if len(win) < 5:
            continue
        feats = np.nan_to_num(_extract_features(win), nan=0.5)
        X_stable.append(feats)

    X_stable = np.array(X_stable)
    n_stable = len(X_stable)
    print(f"[FINGERPRINT] {n_stable} stable windows extracted")

    if n_pre == 0:
        print("[FINGERPRINT] No valid pre-excursion windows — cannot classify")
        return {"probability": 0.5, "loo_accuracy": np.nan, "status": "STABLE"}

    # ── current window (0–3000 BP) ────────────────────────────────────────────
    print("[FINGERPRINT] Extracting current 3,000-yr window...")
    cur_win = _window_slice(df, 0, PRE_WINDOW_YR)
    if len(cur_win) < 3:
        cur_win = df.head(30)  # fallback: most recent available rows
    X_cur = np.nan_to_num(_extract_features(cur_win), nan=0.5).reshape(1, -1)

    # ── assemble dataset ──────────────────────────────────────────────────────
    X_all = np.vstack([X_pre, X_stable])
    y_all = np.array([1] * n_pre + [0] * n_stable)

    # ── full-dataset classifier (for current-window query) ───────────────────
    print("[FINGERPRINT] Training Random Forest classifier...")
    clf_full = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        max_features="sqrt",
        random_state=42,
    )
    clf_full.fit(X_all, y_all)
    prob = float(clf_full.predict_proba(X_cur)[0, 1])

    # ── LOO cross-validation (leave-one-excursion-out) ────────────────────────
    print("[FINGERPRINT] Running LOO cross-validation...")
    loo_correct = 0
    for i in range(n_pre):
        X_train_pre = np.delete(X_pre, i, axis=0)
        X_train = np.vstack([X_train_pre, X_stable])
        y_train = np.array([1] * (n_pre - 1) + [0] * n_stable)

        clf_loo = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            max_features="sqrt",
            random_state=42,
        )
        clf_loo.fit(X_train, y_train)
        pred = clf_loo.predict(X_pre[i : i + 1])
        loo_correct += int(pred[0] == 1)

    loo_accuracy = loo_correct / n_pre
    status = _status(prob)

    print(f"[FINGERPRINT] Pre-excursion probability: {prob:.3f} (threshold {PROB_THRESHOLD}) → {status}")
    print(f"[FINGERPRINT] LOO accuracy: {loo_correct}/{n_pre} = {loo_accuracy:.2f}")

    return {
        "probability": prob,
        "loo_accuracy": loo_accuracy,
        "loo_correct": loo_correct,
        "n_excursions": n_pre,
        "status": status,
    }


if __name__ == "__main__":
    run_fingerprint()
