"""
myth_rag_agent.py — MYTH_RAG: Query interface.

Usage:
    python src/myth_rag/myth_rag_agent.py --query "great flood covered the earth"
    python src/myth_rag/myth_rag_agent.py --query "fire from the sky darkness" -n 8
    python src/myth_rag/myth_rag_agent.py --interactive
    python src/myth_rag/myth_rag_agent.py --query "pole shift axis" --theme flood

Preset queries:
    --preset yd       → Younger Dryas event queries
    --preset laschamp → Laschamp / geomagnetic queries
    --preset flood    → Universal flood motifs
    --preset fire     → Fire from sky / celestial catastrophe
"""

import argparse
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

DB_PATH = ROOT / "data" / "myth_db"

PRESETS = {
    "yd": [
        "great flood covered the mountains",
        "sudden cold winter no summer",
        "fire from sky then water",
        "civilization destroyed catastrophe",
        "survivors fled to high ground",
    ],
    "laschamp": [
        "sky on fire aurora lights in sky",
        "stars fell from heaven",
        "sun disappeared darkness covered earth",
        "magnetic pole celestial bodies moved",
        "cosmic serpent in the sky",
    ],
    "flood": [
        "great flood covered all the earth",
        "boat ark animals saved two of each",
        "rain forty days waters rose mountains",
        "divine warning before the flood",
        "rainbow covenant after the flood",
    ],
    "fire": [
        "fire fell from heaven burned the earth",
        "sun came too close earth burned",
        "celestial body impact catastrophe",
        "flames from sky conflagration",
        "destruction by fire and flood simultaneously",
    ],
}


def load_db(lsa_path: Path):
    import chromadb
    from myth_rag.embeddings import MythEmbeddingFunction, load_lsa, get_backend

    backend = get_backend()
    if backend == "lsa" and lsa_path.exists():
        load_lsa(lsa_path)

    ef = MythEmbeddingFunction()
    client     = chromadb.PersistentClient(path=str(DB_PATH))
    collection = client.get_collection("myth_corpus", embedding_function=ef)
    return collection


def query(collection, text: str, n_results: int = 6,
          theme_filter: str = None) -> list[dict]:
    """Query ChromaDB and return ranked results."""
    where = None
    if theme_filter:
        where = {"themes": {"$contains": theme_filter}}

    try:
        results = collection.query(
            query_texts=[text],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        # where filter failed — retry without
        results = collection.query(
            query_texts=[text],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )

    hits = []
    docs      = results["documents"][0]
    metas     = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        similarity = 1.0 - dist   # cosine distance → similarity
        hits.append({
            "text":          doc,
            "similarity":    similarity,
            "culture":       meta.get("culture", "?"),
            "region":        meta.get("region", "?"),
            "title":         meta.get("title", "?"),
            "myth_id":       meta.get("myth_id", "?"),
            "bp_min":        meta.get("estimated_bp_min", 0),
            "bp_max":        meta.get("estimated_bp_max", 0),
            "themes":        meta.get("themes", "").split(","),
            "temporal_notes": meta.get("temporal_notes", ""),
        })
    return hits


def format_result(hit: dict, idx: int) -> str:
    bp = f"{hit['bp_min']//1000}–{hit['bp_max']//1000} ka BP"
    themes_str = ", ".join(t for t in hit["themes"][:5] if t)
    lines = [
        f"\n{'─'*70}",
        f"#{idx+1}  [{hit['similarity']:.3f}] {hit['culture']} — {hit['region']}",
        f"   Source: {hit['title']}",
        f"   Estimated event age: {bp}",
        f"   Themes: {themes_str}",
        f"   ──",
        f"   {hit['text'][:500].strip()}{'…' if len(hit['text'])>500 else ''}",
    ]
    return "\n".join(lines)


def run_query_set(collection, queries: list[str], n: int,
                  theme_filter: str, label: str) -> dict:
    """Run a set of queries and aggregate results by culture."""
    print(f"\n{'═'*70}")
    print(f"  {label}")
    print(f"{'═'*70}")

    culture_hits: dict[str, list] = {}
    all_hits = []

    for q in queries:
        print(f"\n  Query: '{q}'")
        hits = query(collection, q, n_results=n, theme_filter=theme_filter)
        for i, hit in enumerate(hits):
            print(format_result(hit, i))
            all_hits.append(hit)
            c = hit["culture"]
            culture_hits.setdefault(c, []).append(hit["similarity"])

    return {
        "all_hits": all_hits,
        "culture_hits": culture_hits,
    }


def interactive_session(collection):
    print("\nMYTH_RAG — Interactive query session")
    print("  Commands: 'exit', 'preset:yd', 'preset:laschamp', 'preset:flood', 'preset:fire'")
    print("  Or type any natural language query.\n")

    while True:
        try:
            user_input = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input or user_input == "exit":
            break

        if user_input.startswith("preset:"):
            preset_name = user_input[7:]
            if preset_name in PRESETS:
                for q in PRESETS[preset_name]:
                    hits = query(collection, q, n_results=4)
                    print(f"\n[{q}]")
                    for i, h in enumerate(hits[:3]):
                        print(format_result(h, i))
            else:
                print(f"Unknown preset: {preset_name}. Options: {list(PRESETS.keys())}")
        else:
            hits = query(collection, user_input, n_results=6)
            for i, h in enumerate(hits):
                print(format_result(h, i))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",       type=str,   default=None)
    parser.add_argument("-n",            type=int,   default=5)
    parser.add_argument("--theme",       type=str,   default=None)
    parser.add_argument("--preset",      type=str,   default=None,
                        choices=list(PRESETS.keys()))
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--json-out",    type=str,   default=None,
                        help="Save results as JSON")
    args = parser.parse_args()

    lsa_path = DB_PATH / "lsa_model.pkl"
    print(f"[load] ChromaDB at {DB_PATH}")
    try:
        collection = load_db(lsa_path)
    except Exception as e:
        print(f"[ERROR] {e}")
        print("  Run: python src/myth_rag/ingest_myths.py first")
        sys.exit(1)
    print(f"[db]   {collection.count()} chunks indexed")

    if args.interactive:
        interactive_session(collection)
        return

    if args.preset:
        results = run_query_set(
            collection, PRESETS[args.preset], args.n, args.theme,
            label=f"PRESET: {args.preset.upper()}"
        )
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(results["all_hits"], f, indent=2)
            print(f"\n[saved] {args.json_out}")
        return

    if args.query:
        print(f"\n[query] '{args.query}'  n={args.n}"
              + (f"  theme_filter='{args.theme}'" if args.theme else ""))
        hits = query(collection, args.query, n_results=args.n, theme_filter=args.theme)
        for i, h in enumerate(hits):
            print(format_result(h, i))

        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(hits, f, indent=2)
            print(f"\n[saved] {args.json_out}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
