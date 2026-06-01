"""
substrate.instruments.mythology — Mythology Corpus RAG Instrument
=================================================================

Wraps src/myth_rag/ (ChromaDB + LangChain RAG agent).

Tasks
-----
  ingest          Load the plain-text corpus into ChromaDB vector store.
                  Returns ingestion stats (chunk count, embedding model).

  correlate_events
                  Given a list of geological events (kyr BP + description),
                  retrieve thematically aligned myth passages for each.
                  Returns a correlation table as a list of dicts + markdown.

  query           Free-form semantic query against the myth corpus.
                  Returns top-k passages with similarity scores.
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


class MythologyInstrument(SubstrateInstrument):
    """
    Mythology Corpus RAG (Retrieval-Augmented Generation) Instrument.

    Maintains a ChromaDB vector store of 7 ancient flood/catastrophe myth
    traditions.  Queries are embedded with sentence-transformers and matched
    against geological events from the GeomagneticInstrument to produce
    a myth ↔ geological event correlation table.

    This is SUBSTRATE's unique cross-disciplinary bridge: it treats myth
    texts as independent data channels, not anecdote.

    Traditions : Atrahasis · Ipuwer · Rigveda Manu · Fimbulwinter
                 Aztec Suns · Popol Vuh · Deucalion
    Targets    : YD 12.9 ka · YD termination 11.7 ka · Toba 74 ka · Laschamps 41 ka
    """

    def execute(
        self,
        task: str = "correlate_events",
        data_root: Path = Path("data/processed"),
        gpu: bool = False,   # embeddings run on CPU by default
        **kwargs: Any,
    ) -> tuple[Any, dict]:
        meta: dict = {"warnings": [], "traditions": _TRADITIONS}

        if task == "ingest":
            return self._ingest(data_root, meta, **kwargs)
        elif task == "correlate_events":
            return self._correlate_events(data_root, meta, **kwargs)
        elif task == "query":
            return self._query(data_root, meta, **kwargs)
        else:
            raise ValueError(f"MythologyInstrument: unknown task '{task}'")

    def _ingest(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        try:
            from myth_rag.ingest import ingest_corpus
            stats = ingest_corpus(chroma_path=str(data_root / "chroma"), **kw)
            meta["ingest_stats"] = stats
            return stats, meta
        except ImportError as e:
            self._warn(meta, f"myth_rag.ingest not importable: {e}")
            return {"status": "STUB — chromadb + sentence-transformers required"}, meta

    @staticmethod
    def _norm_event(ev):
        """Normalise an event to dict form — accepts float (kyr BP) or dict."""
        if isinstance(ev, dict):
            return {"kyr_bp": float(ev.get("kyr_bp", 0)),
                    "label": str(ev.get("label", "unknown"))}
        return {"kyr_bp": float(ev), "label": f"{float(ev):.1f} ka"}

    def _correlate_events(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        raw_events = kw.pop(
            "events",
            [
                {"kyr_bp": 12.9, "label": "Younger Dryas onset"},
                {"kyr_bp": 11.7, "label": "YD termination"},
                {"kyr_bp": 41.0, "label": "Laschamps excursion"},
                {"kyr_bp": 74.0, "label": "Toba supervolcano"},
            ],
        )
        # Normalise before ANYTHING else — always dict form
        events = [self._norm_event(e) for e in raw_events]
        top_k: int = kw.pop("top_k", 3)
        meta.update({"events_queried": len(events), "top_k": top_k})

        rag_ok = False
        table = None
        import_err = None
        try:
            from myth_rag.rag_agent import correlate_geological_events
            table = correlate_geological_events(
                events=events,
                chroma_path=str(data_root / "chroma"),
                top_k=top_k,
                **kw,
            )
            rag_ok = True
        except (ImportError, ModuleNotFoundError) as e:
            import_err = e

        if rag_ok and table is not None:
            md_lines = ["| Event (kyr BP) | Tradition | Passage excerpt | Score |",
                        "|---|---|---|---|"]
            for row in table:
                md_lines.append(
                    f"| {row['kyr_bp']:.1f} — {row['label']} "
                    f"| {row['tradition']} | {str(row.get('excerpt',''))[:80]}… "
                    f"| {row.get('score', 0):.3f} |"
                )
            meta["markdown"] = "\n".join(md_lines)
            return {"table": table, "markdown": meta["markdown"]}, meta

        self._warn(meta, f"myth_rag not importable: {import_err} — returning stub")
        stub_table = [
            {
                "kyr_bp": ev["kyr_bp"],
                "label": ev["label"],
                "tradition": "STUB",
                "excerpt": "ChromaDB + LangChain required",
                "score": 0.0,
            }
            for ev in events
        ]
        return {"table": stub_table, "markdown": "STUB"}, meta

    def _query(self, data_root: Path, meta: dict, **kw) -> tuple[Any, dict]:
        query_text: str = kw.pop("query", "global flood catastrophe")
        top_k: int = kw.pop("top_k", 5)
        meta.update({"query": query_text, "top_k": top_k})

        try:
            from myth_rag.rag_agent import semantic_query
            results = semantic_query(
                query=query_text,
                chroma_path=str(data_root / "chroma"),
                top_k=top_k,
                **kw,
            )
            return results, meta
        except ImportError as e:
            self._warn(meta, f"myth_rag not importable: {e}")
            return [], meta
