"""English translations shown next to verified texts.

Many people who share Islam with others (المعرّفون بالإسلام) work with
non-Arabic speakers. When the tool confirms an ayah or a hadith, it also shows
an existing published English translation, so the correct text can be shared
in both languages. Nothing is translated by the tool: the translations are
copied as published, with their source named.

* Quran: Rowwad Translation Center (QuranEnc.com), via fawazahmed0/quran-api.
* Hadith: the English editions of the six books in fawazahmed0/hadith-api.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "en"

QURAN_SOURCE = "Rowwad Translation Center, QuranEnc.com (shown unchanged)"
HADITH_SOURCE = "published English edition of the book, via hadith-api"


@lru_cache(maxsize=None)
def _load(name: str) -> dict:
    path = DATA / f"{name}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text("utf-8"))


def quran(surah: int, ayah_from: int, ayah_to: int | None = None) -> str:
    """English meaning of the ayat, joined with their numbers: "… (5) … (6)"."""
    table = _load("quran")
    ayat = range(ayah_from, (ayah_to or ayah_from) + 1)
    parts = [f"{table[f'{surah}:{a}']} ({a})" for a in ayat if f"{surah}:{a}" in table]
    return " ".join(parts)


def hadith(book_key: str, hadith_number: str) -> str:
    return _load(book_key).get(str(hadith_number), "")
