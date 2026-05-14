"""
ingest_myths.py — MYTH_RAG: Ingest corpus into ChromaDB (manual embeddings).
"""

import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import chromadb
DB_PATH = ROOT / "data" / "myth_db"

def chunk_text(text: str, size: int = 400, overlap: int = 80) -> list:
    text = text.strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for punct in ('.', '\n', '!', '?'):
                b = text.rfind(punct, start + size // 2, end)
                if b > 0: end = b + 1; break
        chunk = text[start:end].strip()
        if len(chunk) > 50: chunks.append(chunk)
        if end >= len(text):
            break  # prevent infinite loop: start=end-overlap never advances at EOF
        start = end - overlap
    return chunks

def build_metadata(myth: dict, chunk_idx: int) -> dict:
    return {
        "culture":          myth["culture"],
        "region":           myth.get("region", ""),
        "title":            myth["title"],
        "myth_id":          myth["id"],
        "chunk_idx":        chunk_idx,
        "composition_bce":  myth.get("composition_bce", 0),
        "bp_min":           myth.get("estimated_bp_min", 0),
        "bp_max":           myth.get("estimated_bp_max", 0),
        "temporal_notes":   myth.get("temporal_notes", ""),
        "source_url":       myth.get("source_url", ""),
        "themes":           ",".join(myth.get("themes", [])),
    }

def ingest(chunk_size, overlap, force):
    from myth_rag.corpus import MYTHS
    from myth_rag.embeddings import lsa_fit, lsa_embed, save_lsa

    print(f"[corpus] {len(MYTHS)} myths")

    # Collect all chunks first
    all_chunks, all_ids, all_metas = [], [], []
    for myth in MYTHS:
        chunks = chunk_text(myth["text"], size=chunk_size, overlap=overlap)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{myth['id']}__c{i}")
            all_metas.append(build_metadata(myth, i))

    print(f"[chunks] {len(all_chunks)} total")

    # Fit LSA on all chunk texts
    print("[LSA]   Fitting …")
    lsa_fit(all_chunks, n_components=min(100, len(all_chunks)-1))
    DB_PATH.mkdir(parents=True, exist_ok=True)
    save_lsa(DB_PATH / "lsa_model.pkl")

    # Embed all chunks
    print("[embed] Computing embeddings …")
    import numpy as np
    embeddings = lsa_embed(all_chunks).tolist()

    # ChromaDB — no embedding function, pass embeddings manually
    client = chromadb.PersistentClient(path=str(DB_PATH))
    if force:
        try: client.delete_collection("myth_corpus")
        except: pass

    collection = client.get_or_create_collection(
        name="myth_corpus",
        metadata={"hnsw:space": "cosine"},
    )

    existing = set(collection.get(include=[])["ids"])
    new_ids   = [i for i in all_ids if i not in existing or force]
    new_docs  = [all_chunks[all_ids.index(i)] for i in new_ids]
    new_embs  = [embeddings[all_ids.index(i)] for i in new_ids]
    new_metas = [all_metas[all_ids.index(i)] for i in new_ids]

    if new_ids:
        # Batch to avoid memory issues
        bs = 50
        for start in range(0, len(new_ids), bs):
            collection.add(
                documents=new_docs[start:start+bs],
                ids=new_ids[start:start+bs],
                embeddings=new_embs[start:start+bs],
                metadatas=new_metas[start:start+bs],
            )

    print(f"[done]  {len(new_ids)} chunks indexed | total: {collection.count()}")

    # Print summary by culture
    from myth_rag.corpus import MYTHS as M
    for myth in M:
        chunks = chunk_text(myth["text"], size=chunk_size, overlap=overlap)
        bp = f"{myth.get('estimated_bp_min',0)//1000}–{myth.get('estimated_bp_max',0)//1000} ka BP"
        print(f"  {myth['culture']:35s} {len(chunks):2d} chunks  {bp}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chunk-size", type=int, default=400)
    p.add_argument("--overlap",    type=int, default=80)
    p.add_argument("--force",      action="store_true")
    a = p.parse_args()
    ingest(a.chunk_size, a.overlap, a.force)

if __name__ == "__main__":
    main()
