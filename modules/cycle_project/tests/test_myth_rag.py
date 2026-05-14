"""
Tests for MYTH_RAG module.
Covers: corpus integrity, chunking, LSA embeddings, query pipeline.
CPU-only, no ChromaDB required.
"""
import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from myth_rag.corpus import MYTHS
import myth_rag.embeddings as emb
from myth_rag.ingest_myths import chunk_text, build_metadata


# ── Corpus integrity ──────────────────────────────────────────────────────────

class TestCorpus:
    def test_loads_21_myths(self):
        assert len(MYTHS) == 21, f"Expected 21 myths, got {len(MYTHS)}"

    def test_all_have_required_fields(self):
        required = {"id", "culture", "region", "title", "themes",
                    "estimated_bp_min", "estimated_bp_max", "text"}
        for myth in MYTHS:
            missing = required - myth.keys()
            assert not missing, f"Myth '{myth.get('id')}' missing: {missing}"

    def test_temporal_bounds_valid(self):
        for myth in MYTHS:
            lo = myth.get("estimated_bp_min", 0)
            hi = myth.get("estimated_bp_max", 0)
            assert lo >= 0, f"{myth['id']}: estimated_bp_min < 0"
            assert hi >= lo, f"{myth['id']}: estimated_bp_max < estimated_bp_min"

    def test_text_non_empty(self):
        for myth in MYTHS:
            assert len(myth["text"].strip()) > 50, (
                f"{myth['id']}: text too short (<50 chars)"
            )

    def test_themes_is_list(self):
        for myth in MYTHS:
            assert isinstance(myth["themes"], list), (
                f"{myth['id']}: themes must be a list"
            )
            assert len(myth["themes"]) > 0, (
                f"{myth['id']}: themes list is empty"
            )

    def test_cultures_are_diverse(self):
        cultures = {m["culture"] for m in MYTHS}
        assert len(cultures) >= 10, (
            f"Expected ≥10 distinct cultures, got {len(cultures)}: {cultures}"
        )

    def test_plato_atlantis_present(self):
        ids = [m["id"] for m in MYTHS]
        assert any("plato" in i or "atlantis" in i for i in ids), (
            "Plato/Atlantis myth missing from corpus"
        )

    def test_flood_theme_present(self):
        all_themes = [t for m in MYTHS for t in m["themes"]]
        assert any("flood" in t.lower() for t in all_themes), (
            "No flood-themed myth found"
        )

    def test_regions_non_empty(self):
        for myth in MYTHS:
            assert myth["region"], f"{myth['id']}: region is empty"

    def test_temporal_overlap_with_yd(self):
        """At least one myth should plausibly overlap with YD (~12,900 BP)."""
        yd_bp = 12_900
        overlapping = [
            m for m in MYTHS
            if m.get("estimated_bp_min", 0) <= yd_bp <= m.get("estimated_bp_max", 0)
        ]
        assert len(overlapping) >= 1, (
            "No myth temporal range overlaps with Younger Dryas (12,900 BP)"
        )

    def test_temporal_overlap_with_laschamp(self):
        """At least one myth should reference or overlap Laschamp (~41,000 BP)."""
        la_bp = 41_000
        overlapping = [
            m for m in MYTHS
            if m.get("estimated_bp_min", 0) <= la_bp <= m.get("estimated_bp_max", 0)
        ]
        assert len(overlapping) >= 1, (
            "No myth temporal range overlaps with Laschamp (41,000 BP)"
        )


# ── Chunking ──────────────────────────────────────────────────────────────────
# chunk_text is character-based: size=400 chars, overlap=80 chars

class TestChunking:
    def test_produces_at_least_one_chunk(self):
        text = "The great flood covered the land. The waters rose. Everything was destroyed."
        chunks = chunk_text(text)
        assert len(chunks) >= 1

    def test_long_text_produces_multiple_chunks(self):
        text = "a" * 2000  # 2000 chars → should produce >1 chunk at size=400
        chunks = chunk_text(text, size=400, overlap=80)
        assert len(chunks) > 1

    def test_chunks_non_empty(self):
        for myth in MYTHS[:5]:
            chunks = chunk_text(myth["text"])
            assert all(len(c.strip()) > 0 for c in chunks)

    def test_overlap_means_more_chunks(self):
        text = "x" * 2000
        no_overlap = chunk_text(text, size=400, overlap=0)
        with_overlap = chunk_text(text, size=400, overlap=200)
        assert len(with_overlap) >= len(no_overlap)

    def test_metadata_has_bp_fields(self):
        meta = build_metadata(MYTHS[0], chunk_idx=0)
        assert "bp_min" in meta
        assert "bp_max" in meta

    def test_metadata_maps_estimated_fields(self):
        """build_metadata maps estimated_bp_* → bp_* for ChromaDB storage."""
        myth = MYTHS[0]
        meta = build_metadata(myth, chunk_idx=0)
        assert meta["bp_min"] == myth.get("estimated_bp_min", 0)
        assert meta["bp_max"] == myth.get("estimated_bp_max", 0)


# ── LSA Embeddings ────────────────────────────────────────────────────────────
# lsa_fit() sets global state, lsa_embed() uses it — no return/model passing.

class TestEmbeddings:
    @pytest.fixture(scope="class", autouse=True)
    def fit_lsa(self):
        """Fit LSA on first 10 myths before any test in this class."""
        texts = [m["text"][:600] for m in MYTHS[:10]]
        emb.lsa_fit(texts, n_components=16)
        yield

    def test_get_backend_returns_string(self):
        backend = emb.get_backend()
        assert backend in ("sentence-transformers", "lsa")

    def test_lsa_embed_shape(self):
        texts = [m["text"][:600] for m in MYTHS[:5]]
        vecs = emb.lsa_embed(texts)
        assert vecs.shape[0] == 5
        assert vecs.shape[1] > 0

    def test_lsa_vectors_normalized(self):
        texts = [m["text"][:600] for m in MYTHS[:5]]
        vecs = emb.lsa_embed(texts)
        norms = np.linalg.norm(vecs, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    def test_different_texts_different_embeddings(self):
        t0 = [MYTHS[0]["text"][:600]]
        t1 = [MYTHS[1]["text"][:600]]
        v0 = emb.lsa_embed(t0)
        v1 = emb.lsa_embed(t1)
        assert not np.allclose(v0, v1, atol=1e-3)

    def test_cosine_similarity_self_is_one(self):
        texts = [m["text"][:600] for m in MYTHS[:5]]
        vecs = emb.lsa_embed(texts)
        for i in range(len(texts)):
            sim = float(vecs[i] @ vecs[i])
            np.testing.assert_allclose(sim, 1.0, atol=1e-3)

    def test_query_retrieval_order(self):
        """Query about floods should rank flood myth higher than fire myth."""
        corpus = [
            "The great flood covered the earth and all lands were submerged by rising waters.",
            "The sky burned with fire and the mountain of the gods collapsed into ash.",
        ]
        emb.lsa_fit(corpus, n_components=4)
        vecs  = emb.lsa_embed(corpus)
        query = emb.lsa_embed(["flood waters rising covering all land"])
        sims  = vecs @ query.T
        assert sims[0] >= sims[1], (
            "Flood text should rank >= fire text for flood query"
        )
