"""
substrate.instruments.mythology_instrument — Mythology Corpus RAG Instrument
=============================================================================

ChromaDB + LangChain RAG agent over 7 ancient flood/catastrophe myth traditions.
Queries are matched against geological events to produce a correlation table.

Tasks: ingest | correlate_events | query
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from substrate.instruments.base import SubstrateInstrument

logger = logging.getLogger("substrate.mythology")

_TRADITIONS = [
    "Sumerian Atrahasis",
    "Egyptian Papyrus of Ipuwer",
    "Rigveda Manu",
    "Norse Fimbulwinter",
    "Aztec Suns cosmogony",
    "Mayan Popol Vuh",
    "Greek Deucalion",
]

_DEFAULT_EVENTS = [
    {"kyr_bp": 12.9, "label": "Younger Dryas onset"},
    {"kyr_bp": 11.7, "label": "YD termination"},
    {"kyr_bp": 41.0, "label": "Laschamps excursion"},
    {"kyr_bp": 74.0, "label": "Toba supervolcano"},
]


def _as_event_dict(ev) -> dict:
    """Accept float (kyr BP) or {kyr_bp, label} dict — always return dict."""
    if isinstance(ev, dict):
        return {"kyr_bp": float(ev.get("kyr_bp", 0.0)),
                "label": str(ev.get("label", "unknown"))}
    return {"kyr_bp": float(ev), "label": f"{float(ev):.1f} ka"}


def _try_import_rag():
    """Return (fn, None) or (None, error_string)."""
    try:
        from myth_rag.rag_agent import correlate_geological_events
        return correlate_geological_events, None
    except Exception as e:
        return None, str(e)


def _try_import_query():
    try:
        from myth_rag.rag_agent import semantic_query
        return semantic_query, None
    except Exception as e:
        return None, str(e)


def _try_import_ingest():
    try:
        from myth_rag.ingest import ingest_corpus
        return ingest_corpus, None
    except Exception as e:
        return None, str(e)


class MythologyInstrument(SubstrateInstrument):
    """
    Mythology Corpus RAG (Retrieval-Augmented Generation) Instrument.

    Traditions: Atrahasis · Ipuwer · Rigveda Manu · Fimbulwinter
                Aztec Suns · Popol Vuh · Deucalion
    Targets:    YD 12.9 ka · YD termination 11.7 ka · Laschamps 41 ka · Toba 74 ka
    """

    def execute(
        self,
        task: str = "correlate_events",
        data_root: Path = Path("data/processed"),
        gpu: bool = False,
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        meta: dict = {"warnings": [], "traditions": _TRADITIONS}

        if task == "ingest":
            return self._ingest(data_root, meta, **kwargs)
        if task == "correlate_events":
            return self._correlate_events(data_root, meta, **kwargs)
        if task == "query":
            return self._query(data_root, meta, **kwargs)
        raise ValueError(f"MythologyInstrument: unknown task '{task}'")

    # ------------------------------------------------------------------
    def _ingest(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        fn, err = _try_import_ingest()
        if fn is None:
            self._warn(meta, f"myth_rag.ingest unavailable: {err}")
            return {"status": "STUB — chromadb + sentence-transformers required"}, meta
        stats = fn(chroma_path=str(data_root / "chroma"), **kw)
        meta["ingest_stats"] = stats
        return stats, meta

    def _correlate_events(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        raw = kw.pop("events", _DEFAULT_EVENTS)
        events = [_as_event_dict(e) for e in raw]
        top_k = int(kw.pop("top_k", 3))
        meta.update({"events_queried": len(events), "top_k": top_k})

        fn, err = _try_import_rag()

        if fn is not None:
            table = fn(events=events,
                       chroma_path=str(data_root / "chroma"),
                       top_k=top_k, **kw)
            md = ["| Event (kyr BP) | Tradition | Passage excerpt | Score |",
                  "|---|---|---|---|"]
            for row in table:
                md.append(
                    f"| {row['kyr_bp']:.1f} — {row['label']} "
                    f"| {row['tradition']} "
                    f"| {str(row.get('excerpt',''))[:80]}… "
                    f"| {row.get('score', 0.0):.3f} |"
                )
            meta["markdown"] = "\n".join(md)
            return {"table": table, "markdown": meta["markdown"]}, meta

        # Stub path
        self._warn(meta, f"myth_rag unavailable: {err}")
        stub = [
            {
                "kyr_bp": ev["kyr_bp"],
                "label": ev["label"],
                "tradition": "STUB",
                "excerpt": "ChromaDB + LangChain required",
                "score": 0.0,
            }
            for ev in events
        ]
        return {"table": stub, "markdown": "STUB"}, meta

    def _query(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        query_text = str(kw.pop("query", "global flood catastrophe"))
        top_k = int(kw.pop("top_k", 5))
        meta.update({"query": query_text, "top_k": top_k})

        fn, err = _try_import_query()
        if fn is None:
            self._warn(meta, f"myth_rag unavailable: {err}")
            return [], meta
        return fn(query=query_text,
                  chroma_path=str(data_root / "chroma"),
                  top_k=top_k, **kw), meta
