"""
correlate.py — MYTH_RAG: Cross-correlate myth retrieval with GNN anomaly scores.

For each top anomalous geological window (from anomaly_scores.parquet),
queries the myth corpus with temporally-informed queries, then builds a
correlation matrix: geological event × cultural tradition × similarity score.

Statistical testing (bootstrap CI + permutation p-value) is applied to each
(event, culture) mean-similarity score. The "r" column is the mean cosine
similarity, not a Pearson r — bootstrap and permutation treat it as such.

Outputs:
    data/processed/myth_correlation.csv     — ranked myth-geology matches
    data/processed/myth_correlations.json   — full stats table (CI, p-values)
    data/processed/myth_timeline.png        — combined timeline visualization
    data/processed/myth_heatmap.png         — culture × geology correlation heatmap

Usage:
    python src/myth_rag/correlate.py [--n-geo 10] [--n-myth 5]
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

PROC    = ROOT / "data" / "processed"
DB_PATH = ROOT / "data" / "myth_db"

N_BOOTSTRAP    = 10_000
N_PERMUTATIONS = 10_000

# Known geological events with associated query text
GEO_EVENTS = [
    {
        "name": "Laschamp excursion",
        "age_bp": 41_000,
        "window_bp": 4_000,
        "queries": [
            "sky on fire aurora lights celestial serpent",
            "sun disappeared darkness magnetic stars fell",
            "cosmic catastrophe geomagnetic sky fire",
        ],
        "color": "#FF453A",
    },
    {
        "name": "Younger Dryas onset",
        "age_bp": 12_900,
        "window_bp": 500,
        "queries": [
            "great flood covered mountains catastrophe",
            "fire from sky then great flood water",
            "civilization destroyed ice cold sudden winter",
            "Atlantis nine thousand years catastrophe",
        ],
        "color": "#0A84FF",
    },
    {
        "name": "8.2 ka cold event",
        "age_bp": 8_200,
        "window_bp": 200,
        "queries": [
            "flood waters rose suddenly drowned land",
            "cold winter starvation survived",
        ],
        "color": "#30D158",
    },
    {
        "name": "Last Glacial Maximum",
        "age_bp": 20_000,
        "window_bp": 3_000,
        "queries": [
            "great ice cold darkness covered land",
            "winter lasted generations no summer",
            "world ended ice survived underground",
        ],
        "color": "#FF9F0A",
    },
    {
        "name": "Mono Lake excursion",
        "age_bp": 34_000,
        "window_bp": 2_000,
        "queries": [
            "sky disturbed heavens moved strange lights",
            "previous world age destruction",
        ],
        "color": "#A78BFA",
    },
]


def bootstrap_ci(sims: list, n_iter: int = N_BOOTSTRAP) -> tuple:
    """95% bootstrap CI on the mean of a similarity list."""
    if len(sims) < 2:
        return (float("nan"), float("nan"))
    arr = np.asarray(sims, dtype=float)
    # Vectorised: draw n_iter samples of length n simultaneously
    idx = np.random.randint(0, len(arr), size=(n_iter, len(arr)))
    boot_means = arr[idx].mean(axis=1)
    return (float(np.percentile(boot_means, 2.5)),
            float(np.percentile(boot_means, 97.5)))


def permutation_pvalue(all_event_sims: list, culture_sims: list,
                       n_iter: int = N_PERMUTATIONS) -> float:
    """
    Two-sided permutation p-value.

    Null: culture label doesn't matter — the observed mean similarity is no
    different from drawing the same number of scores at random from this
    event's full pool.
    """
    if not culture_sims or len(all_event_sims) < 2:
        return float("nan")
    pool = np.asarray(all_event_sims, dtype=float)
    n    = len(culture_sims)
    obs  = float(np.mean(culture_sims))
    # Sample with replacement when culture sample is larger than the pool
    idx = np.random.randint(0, len(pool), size=(n_iter, n))
    perm_means = pool[idx].mean(axis=1)
    return float(np.mean(np.abs(perm_means) >= abs(obs)))


def sig_marker(p_bonf: float) -> str:
    if np.isnan(p_bonf): return "?"
    if p_bonf < 0.001:   return "***"
    if p_bonf < 0.01:    return "**"
    if p_bonf < 0.05:    return "*"
    return "ns"


def compute_significance_stats(results_by_event: dict,
                                n_bootstrap: int = N_BOOTSTRAP,
                                n_perms: int = N_PERMUTATIONS) -> list:
    """
    For every (event, culture) pair with at least one hit, compute:
      - mean similarity (r)
      - 95% bootstrap CI on the mean
      - two-sided permutation p-value (null = random culture assignment)
      - Bonferroni-corrected p (multiplied by total number of tests)
    Returns a list of dicts sorted by r descending.
    """
    rows = []
    for event_name, data in results_by_event.items():
        culture_scores = data["culture_scores"]
        all_event_sims: list = []
        for c_data in culture_scores.values():
            all_event_sims.extend(c_data["sims"])

        for culture, c_data in culture_scores.items():
            sims = c_data["sims"]
            if not sims:
                continue
            ci_lo, ci_hi = bootstrap_ci(sims, n_iter=n_bootstrap)
            p_val        = permutation_pvalue(all_event_sims, sims, n_iter=n_perms)
            rows.append({
                "event":   event_name,
                "culture": culture,
                "r":       float(np.mean(sims)),
                "ci_lo":   ci_lo,
                "ci_hi":   ci_hi,
                "p_value": p_val,
                "n_sims":  len(sims),
            })

    n_tests = sum(1 for r in rows if not np.isnan(r["p_value"]))
    for r in rows:
        p = r["p_value"]
        r["p_bonferroni"] = float(min(1.0, p * n_tests)) if not np.isnan(p) else float("nan")
        r["significant"]  = (not np.isnan(r["p_bonferroni"])) and r["p_bonferroni"] < 0.05
        r["sig_marker"]   = sig_marker(r["p_bonferroni"])

    rows.sort(key=lambda x: x["r"], reverse=True)
    return rows


def print_stats_table(stats: list, top_n: int = 20) -> None:
    """Print the ranked stats table with CI and p-values."""
    hdr = (f"{'Event':<28} {'Culture':<28} {'r':>6}  "
           f"{'95% CI':^18}  {'p-val':>7}  {'Bonf-p':>7}  {'Sig':>4}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in stats[:top_n]:
        ci = (f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
              if not np.isnan(r["ci_lo"]) else "     n/a      ")
        pv  = f"{r['p_value']:.4f}"     if not np.isnan(r["p_value"])     else "   n/a"
        pbf = f"{r['p_bonferroni']:.4f}" if not np.isnan(r["p_bonferroni"]) else "   n/a"
        print(f"  {r['event']:<26} × {r['culture']:<26} {r['r']:>6.4f}  "
              f"{ci:^18}  {pv:>7}  {pbf:>7}  {r['sig_marker']:>4}")


def save_stats_json(stats: list, n_bootstrap: int, n_perms: int,
                    out_path: Path) -> None:
    """Save full stats to myth_correlations.json."""
    payload = {
        "correlations": [
            {
                "event":        r["event"],
                "culture":      r["culture"],
                "r":            round(r["r"], 6),
                "ci_lo":        None if np.isnan(r["ci_lo"])  else round(r["ci_lo"],  6),
                "ci_hi":        None if np.isnan(r["ci_hi"])  else round(r["ci_hi"],  6),
                "p_value":      None if np.isnan(r["p_value"]) else round(r["p_value"], 6),
                "p_bonferroni": None if np.isnan(r["p_bonferroni"]) else round(r["p_bonferroni"], 6),
                "significant":  r["significant"],
                "n_sims":       r["n_sims"],
            }
            for r in stats
        ],
        "n_bootstrap":    n_bootstrap,
        "n_permutations": n_perms,
        "n_tests":        len(stats),
        "generated_at":   datetime.now(timezone.utc).isoformat(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"  [saved] → {out_path}")


def load_db():
    import chromadb
    from myth_rag.embeddings import MythEmbeddingFunction, load_lsa, get_backend

    backend = get_backend()
    lsa_path = DB_PATH / "lsa_model.pkl"
    if backend == "lsa" and lsa_path.exists():
        load_lsa(lsa_path)

    ef     = MythEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_collection("myth_corpus", embedding_function=ef)


def query_for_event(collection, event: dict, n_results: int) -> list[dict]:
    """Run all queries for a geological event and aggregate by myth culture."""
    from myth_rag.myth_rag_agent import query

    culture_scores: dict[str, list] = {}
    all_hits = []

    for q in event["queries"]:
        hits = query(collection, q, n_results=n_results)
        for hit in hits:
            # Temporal filter: does this myth's date range overlap the event window?
            bp_min = hit["bp_min"]
            bp_max = hit["bp_max"]
            event_age = event["age_bp"]
            window    = event["window_bp"]

            temporal_overlap = not (
                bp_max < event_age - window * 5   # myth too recent
                or bp_min > event_age + window * 5  # myth too old
            )

            c = hit["culture"]
            culture_scores.setdefault(c, {"sims": [], "temporal_overlap": temporal_overlap})
            culture_scores[c]["sims"].append(hit["similarity"])
            culture_scores[c]["temporal_overlap"] |= temporal_overlap
            all_hits.append({**hit, "geo_event": event["name"], "geo_age_bp": event["age_bp"]})

    return all_hits, culture_scores


def build_correlation_matrix(results_by_event: dict) -> pd.DataFrame:
    """Build (geo_event × culture) similarity matrix."""
    all_cultures = set()
    for data in results_by_event.values():
        all_cultures.update(data["culture_scores"].keys())

    cultures = sorted(all_cultures)
    events   = list(results_by_event.keys())

    matrix = pd.DataFrame(0.0, index=events, columns=cultures)

    for event_name, data in results_by_event.items():
        for culture, scores in data["culture_scores"].items():
            matrix.loc[event_name, culture] = np.mean(scores["sims"])

    return matrix


def plot_heatmap(matrix: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(max(12, len(matrix.columns)), 5))
    fig.patch.set_facecolor("#0F172A")
    ax.set_facecolor("#0F172A")

    data = matrix.values
    im   = ax.imshow(data, cmap="plasma", aspect="auto",
                     vmin=0, vmax=data.max())

    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right",
                       color="#94A3B8", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, color="#E2E8F0", fontsize=9)

    # Annotate cells
    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            val = data[i, j]
            if val > 0.01:
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color="white" if val > data.max()*0.5 else "#94A3B8",
                        fontsize=7)

    plt.colorbar(im, ax=ax, label="Mean semantic similarity", fraction=0.02)
    ax.set_title("MYTH_RAG — Geological Event × Cultural Tradition Similarity",
                 color="#E2E8F0", fontsize=10, pad=12)
    for sp in ax.spines.values(): sp.set_edgecolor("#1E293B")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] heatmap → {out_path}")


def plot_timeline(all_hits: list[dict], anomaly_scores: pd.DataFrame,
                  out_path: Path):
    fig, (ax_geo, ax_myth) = plt.subplots(
        2, 1, figsize=(16, 9), sharex=True,
        gridspec_kw={"height_ratios": [1.2, 2], "hspace": 0.05}
    )
    fig.patch.set_facecolor("#0F172A")

    # ── Geological anomaly scores ─────────────────────────────────────────────
    ax_geo.set_facecolor("#0F172A")
    if anomaly_scores is not None:
        x_s = anomaly_scores["age_bp"].values / 1000.0
        y_s = anomaly_scores["anomaly_score"].values
        ax_geo.plot(x_s, y_s, color="#A78BFA", lw=1.2)
        ax_geo.fill_between(x_s, y_s, alpha=0.25, color="#A78BFA")
        thresh = np.percentile(y_s, 95)
        ax_geo.axhline(thresh, color="#FF453A", lw=0.7, ls="--", alpha=0.7)
    ax_geo.set_ylabel("GNN anomaly\nscore", color="#94A3B8", fontsize=8)
    ax_geo.tick_params(colors="#64748B")
    for sp in ax_geo.spines.values(): sp.set_edgecolor("#1E293B")

    # ── Myth timeline ─────────────────────────────────────────────────────────
    ax_myth.set_facecolor("#0F172A")

    # One row per culture
    cultures = sorted({h["culture"] for h in all_hits})
    cmap     = plt.get_cmap("tab20")
    y_map    = {c: i for i, c in enumerate(cultures)}

    for hit in all_hits:
        c     = hit["culture"]
        y     = y_map[c]
        bp_min = hit["bp_min"] / 1000.0
        bp_max = hit["bp_max"] / 1000.0
        sim   = hit["similarity"]
        geo_event = hit.get("geo_event", "")

        # Find color for geo event
        evt_color = "#64748B"
        for evt in GEO_EVENTS:
            if evt["name"] == geo_event:
                evt_color = evt["color"]
                break

        ax_myth.barh(y, left=bp_min, width=bp_max - bp_min,
                     height=0.6, color=evt_color, alpha=sim * 0.8 + 0.1,
                     edgecolor="#1E293B", linewidth=0.3)

    ax_myth.set_yticks(range(len(cultures)))
    ax_myth.set_yticklabels(cultures, color="#E2E8F0", fontsize=7)
    ax_myth.set_xlabel("Age (ka BP)", color="#94A3B8", fontsize=9)
    ax_myth.set_ylabel("Culture / Tradition", color="#94A3B8", fontsize=8)
    ax_myth.tick_params(colors="#64748B")
    for sp in ax_myth.spines.values(): sp.set_edgecolor("#1E293B")
    ax_myth.invert_xaxis()

    # Legend
    patches = [mpatches.Patch(color=e["color"], label=e["name"]) for e in GEO_EVENTS]
    ax_myth.legend(handles=patches, loc="upper right", fontsize=7,
                   framealpha=0.15, labelcolor="#E2E8F0")

    # Event markers
    for evt in GEO_EVENTS:
        ka = evt["age_bp"] / 1000.0
        for ax in (ax_geo, ax_myth):
            ax.axvline(ka, color=evt["color"], lw=0.9, ls=":", alpha=0.6)

    ax_geo.invert_xaxis()
    fig.suptitle("MYTH_RAG — Myth-Geology Temporal Correlation",
                 color="#E2E8F0", fontsize=11, fontweight="bold", y=0.999)
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [plot] timeline → {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-geo",  type=int, default=10,
                        help="Top anomalous geo windows to analyze")
    parser.add_argument("--n-myth", type=int, default=6,
                        help="Myth chunks to retrieve per query")
    args = parser.parse_args()

    print("[load] ChromaDB …")
    collection = load_db()
    print(f"[db]   {collection.count()} chunks")

    # Load anomaly scores
    anomaly_path = PROC / "anomaly_scores.parquet"
    anomaly_scores = None
    if anomaly_path.exists():
        anomaly_scores = pd.read_parquet(anomaly_path)
        print(f"[geo]  {len(anomaly_scores)} anomaly windows loaded")

    # Query each geological event
    results_by_event: dict = {}
    all_hits_combined: list = []

    for event in GEO_EVENTS:
        print(f"\n[event] {event['name']} ({event['age_bp']//1000} ka BP) …")
        hits, culture_scores = query_for_event(collection, event, args.n_myth)
        results_by_event[event["name"]] = {
            "hits": hits,
            "culture_scores": culture_scores,
        }
        all_hits_combined.extend(hits)

        # Print top hits for this event
        top_by_sim = sorted(hits, key=lambda h: h["similarity"], reverse=True)[:3]
        for h in top_by_sim:
            bp = f"{h['bp_min']//1000}–{h['bp_max']//1000} ka BP"
            print(f"  [{h['similarity']:.3f}] {h['culture']:30s} {bp}")

    # Build correlation matrix
    print("\n[matrix] Building correlation matrix …")
    matrix = build_correlation_matrix(results_by_event)
    matrix_path = PROC / "myth_correlation.csv"
    matrix.to_csv(matrix_path)
    print(f"  [saved] → {matrix_path}")

    # Statistical significance testing
    print("\n[stats] Running bootstrap CI + permutation tests …")
    print(f"        N_bootstrap={N_BOOTSTRAP:,}  N_permutations={N_PERMUTATIONS:,}")
    stats = compute_significance_stats(results_by_event,
                                       n_bootstrap=N_BOOTSTRAP,
                                       n_perms=N_PERMUTATIONS)

    # Print top correlations with stats
    print("\n[correlations] Top culture × event matches (with significance):")
    print_stats_table(stats, top_n=20)

    # Significant pairs only
    sig_pairs = [r for r in stats if r["significant"]]
    print(f"\n[stats] {len(sig_pairs)}/{len(stats)} pairs significant after Bonferroni correction:")
    for r in sig_pairs[:10]:
        print(f"  {r['sig_marker']:>3}  r={r['r']:.4f}  "
              f"[{r['ci_lo']:.3f},{r['ci_hi']:.3f}]  "
              f"p_bonf={r['p_bonferroni']:.4f}  "
              f"{r['event']} × {r['culture']}")

    # Save full stats JSON
    json_path = PROC / "myth_correlations.json"
    save_stats_json(stats, N_BOOTSTRAP, N_PERMUTATIONS, json_path)

    # Plots
    print("\n[plot] Generating visualizations …")
    plot_heatmap(matrix, PROC / "myth_heatmap.png")
    plot_timeline(all_hits_combined, anomaly_scores, PROC / "myth_timeline.png")

    print(f"\n[done] Outputs:")
    for f in ["myth_correlation.csv", "myth_correlations.json",
              "myth_heatmap.png", "myth_timeline.png"]:
        p = PROC / f
        sz = f"{p.stat().st_size/1024:.1f} KB" if p.exists() else "MISSING"
        print(f"  {f:35s} {sz}")


if __name__ == "__main__":
    main()
