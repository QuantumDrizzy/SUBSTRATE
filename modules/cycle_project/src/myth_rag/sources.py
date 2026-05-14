"""
sources.py — MYTH_RAG: Real download sources for full myth texts.

Run download_sources() on your machine (needs internet access).
Files saved to data/myth_texts/ — then re-run ingest_myths.py --source downloaded.

All sources are public domain or openly licensed.
"""

from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent.parent
TEXT_DIR = ROOT / "data" / "myth_texts"

DOWNLOAD_SOURCES = [
    # Project Gutenberg — public domain
    {
        "id": "gilgamesh_sandars",
        "url": "https://www.gutenberg.org/cache/epub/11000/pg11000.txt",
        "filename": "gilgamesh.txt",
        "culture": "Sumerian/Akkadian",
        "estimated_bp": (10_000, 15_000),
    },
    {
        "id": "plato_timaeus",
        "url": "https://www.gutenberg.org/cache/epub/1572/pg1572.txt",
        "filename": "plato_timaeus.txt",
        "culture": "Greek",
        "estimated_bp": (11_400, 11_800),
        "note": "Contains Atlantis account. Search for 'nine thousand years' to find key passage.",
    },
    {
        "id": "plato_critias",
        "url": "https://www.gutenberg.org/cache/epub/1571/pg1571.txt",
        "filename": "plato_critias.txt",
        "culture": "Greek",
        "estimated_bp": (11_400, 11_800),
    },
    {
        "id": "ovid_metamorphoses",
        "url": "https://www.gutenberg.org/cache/epub/21765/pg21765.txt",
        "filename": "ovid_metamorphoses.txt",
        "culture": "Roman/Greek",
        "estimated_bp": (10_000, 14_000),
        "note": "Book I contains Deucalion flood",
    },
    {
        "id": "kalevala",
        "url": "https://www.gutenberg.org/cache/epub/5765/pg5765.txt",
        "filename": "kalevala.txt",
        "culture": "Finnish",
        "estimated_bp": (12_000, 15_000),
    },
    {
        "id": "prose_edda",
        "url": "https://www.gutenberg.org/cache/epub/2973/pg2973.txt",
        "filename": "prose_edda.txt",
        "culture": "Norse",
        "estimated_bp": (12_000, 15_000),
        "note": "Contains Gylfaginning with Ragnarok and Fimbulwinter",
    },
    {
        "id": "bible_genesis",
        "url": "https://www.gutenberg.org/cache/epub/10/pg10.txt",
        "filename": "bible_king_james.txt",
        "culture": "Hebrew",
        "estimated_bp": (7_000, 15_000),
        "note": "Search for Genesis chapters 6-9 (Noah) and Exodus (plagues)",
    },
    # Sacred Texts archive (may require wget --wait on your machine)
    {
        "id": "shatapatha_brahmana",
        "url": "https://sacred-texts.com/hin/sbr/sbe12/sbe1204.htm",
        "filename": "shatapatha_manu.html",
        "culture": "Hindu",
        "estimated_bp": (10_000, 20_000),
    },
    {
        "id": "vendidad_zoroastrian",
        "url": "https://sacred-texts.com/zor/sbe04/sbe0402.htm",
        "filename": "vendidad_yima.html",
        "culture": "Zoroastrian",
        "estimated_bp": (12_000, 15_000),
    },
    {
        "id": "popol_vuh_english",
        "url": "https://sacred-texts.com/nam/maya/pvse/pvse00.htm",
        "filename": "popol_vuh.html",
        "culture": "Maya",
        "estimated_bp": (12_000, 15_000),
    },
    {
        "id": "book_of_enoch",
        "url": "https://sacred-texts.com/bib/boe/index.htm",
        "filename": "book_of_enoch.html",
        "culture": "Hebrew Apocrypha",
        "estimated_bp": (10_000, 15_000),
        "note": "Watchers, fallen angels, flood, astronomical changes",
    },
]


def download_all(force: bool = False):
    """Download all sources to data/myth_texts/. Run on your machine with internet."""
    import urllib.request
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    for src in DOWNLOAD_SOURCES:
        dest = TEXT_DIR / src["filename"]
        if dest.exists() and not force:
            print(f"  [cache] {src['filename']}")
            continue
        print(f"  [download] {src['url']}")
        try:
            urllib.request.urlretrieve(src["url"], dest)
            print(f"  [ok] {dest.stat().st_size/1024:.1f} KB")
        except Exception as e:
            print(f"  [ERROR] {e}")


if __name__ == "__main__":
    download_all()
