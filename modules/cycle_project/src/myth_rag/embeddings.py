"""
embeddings.py — MYTH_RAG: Embedding backend with auto-fallback.

Priority:
  1. SentenceTransformers (semantic, best quality) — if installed
  2. LSA: TF-IDF + TruncatedSVD (latent semantic analysis, CPU-only, no downloads)

The ChromaDB custom embedding function wraps whichever is available.
"""

import numpy as np
from pathlib import Path

_BACKEND = None


def _detect_backend():
    global _BACKEND
    try:
        from sentence_transformers import SentenceTransformer
        _BACKEND = "sentence_transformers"
    except ImportError:
        _BACKEND = "lsa"
    return _BACKEND


def get_backend() -> str:
    if _BACKEND is None:
        _detect_backend()
    return _BACKEND


# ─── Sentence Transformers backend ───────────────────────────────────────────

_ST_MODEL = None


def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    global _ST_MODEL
    if _ST_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _ST_MODEL = SentenceTransformer(model_name)
    return _ST_MODEL


# ─── LSA backend ─────────────────────────────────────────────────────────────

_LSA_VECTORIZER = None
_LSA_SVD        = None
_LSA_DIM        = 128


def lsa_fit(texts: list[str], n_components: int = 128):
    """Fit TF-IDF + TruncatedSVD on a corpus."""
    global _LSA_VECTORIZER, _LSA_SVD, _LSA_DIM
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    _LSA_DIM = min(n_components, len(texts) - 1)
    _LSA_VECTORIZER = TfidfVectorizer(
        max_features=8000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=1,
    )
    X = _LSA_VECTORIZER.fit_transform(texts)
    _LSA_SVD = TruncatedSVD(n_components=_LSA_DIM, random_state=42)
    _LSA_SVD.fit(X)
    var_explained = _LSA_SVD.explained_variance_ratio_.sum()
    print(f"  [LSA] {_LSA_DIM}d, TF-IDF vocab {len(_LSA_VECTORIZER.vocabulary_)}, "
          f"variance explained: {var_explained:.3f}")


def lsa_embed(texts: list[str]) -> np.ndarray:
    """Embed texts using fitted LSA model."""
    if _LSA_VECTORIZER is None:
        raise RuntimeError("Call lsa_fit() first")
    X = _LSA_VECTORIZER.transform(texts)
    Z = _LSA_SVD.transform(X)
    # L2 normalise
    norms = np.linalg.norm(Z, axis=1, keepdims=True)
    return Z / (norms + 1e-9)


def save_lsa(path: Path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump({"vectorizer": _LSA_VECTORIZER, "svd": _LSA_SVD, "dim": _LSA_DIM}, f)


def load_lsa(path: Path):
    global _LSA_VECTORIZER, _LSA_SVD, _LSA_DIM
    import pickle
    with open(path, "rb") as f:
        obj = pickle.load(f)
    _LSA_VECTORIZER = obj["vectorizer"]
    _LSA_SVD        = obj["svd"]
    _LSA_DIM        = obj["dim"]


# ─── Unified embed function ───────────────────────────────────────────────────

def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    backend = get_backend()
    if backend == "sentence_transformers":
        model = _get_st_model(model_name)
        return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    else:
        return lsa_embed(texts)


# ─── ChromaDB custom embedding function ──────────────────────────────────────

class MythEmbeddingFunction:
    def name(self) -> str:
        return f"myth_lsa_{self.backend}"
    """Drop-in ChromaDB embedding function. Auto-selects backend."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.backend    = get_backend()
        print(f"  [embed] backend = {self.backend}")

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        vecs = embed(input, self.model_name)
        return vecs.tolist()

    def embed_documents(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self.__call__(input)

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        if isinstance(input, str):
            input = [input]
        return self.__call__(input)
