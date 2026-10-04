"""Hadith finder: the two Sahihs and the four Sunan.

Given a text presented as a hadith, find it in Sahih al-Bukhari, Sahih
Muslim, Sunan Abi Dawud, Jami' al-Tirmidhi, Sunan al-Nasa'i or Sunan Ibn
Majah, and return the book, the number and the scholar's grading.

Gradings: hadiths of the two Sahihs are authentic by scholarly consensus on
these books. For the four Sunan the grading of Shaykh al-Albani that ships
with the dataset is shown, attributed to him by name. The tool never
produces a grading of its own.

How it works (plain version):
1. Every hadith is normalized (no diacritics, one spelling) and its words
   are put in an index, like the index at the back of a book.
2. For the input we use the index to pick the 30 hadiths that share the most
   (and the rarest) words with it. This is the BM25 ranking used by search
   engines.
3. Each of those is compared character by character with the input using a
   fuzzy "partial match", which finds the best-matching stretch inside the
   long hadith text (the chain of narrators is ignored this way). The score
   is 0–100.
4. The score decides whether the text was found. What the hadith's status
   is comes only from the scholar's grading stored with it.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from rapidfuzz import fuzz

from .normalize import normalize

from . import translation
from .embeddings import WordVectors

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "hadith_vectors.npz"

# Semantic search (our own Word2Vec model, see models/train_embeddings.py):
SEM_WORD_SIM = 0.6      # two words count as the same meaning at/above this
SEM_EXPAND_TOPN = 8     # neighbours added to the query for retrieval
SEM_COVERAGE = 0.8      # share of the meaning (rarity-weighted) the hadith must cover
SEM_MIN_PARTIAL = 50    # and a minimum surface resemblance, as a sanity check

MIN_WORDS = 2
FOUND_SCORE = 88      # at or above: the text is in the Sahihs
SIMILAR_SCORE = 70    # between: close wording, may be narrated by meaning
# Share of the input's meaningful words (weighted by rarity) that must also
# appear in the hadith. Stops a common phrase like "من الإيمان" from making an
# unrelated saying look like a match.
FOUND_COVERAGE = 0.85
SIMILAR_COVERAGE = 0.70
TOP_CANDIDATES = 40

BOOKS = {
    "bukhari": "صحيح البخاري",
    "muslim": "صحيح مسلم",
    "abudawud": "سنن أبي داود",
    "tirmidhi": "جامع الترمذي",
    "nasai": "سنن النسائي",
    "ibnmajah": "سنن ابن ماجه",
}
SAHIHAIN = {"bukhari", "muslim"}

# Al-Albani's grading labels as stored in the dataset -> Arabic.
_GRADE_AR = {
    "Sahih": "صحيح", "Hasan": "حسن", "Hasan Sahih": "حسن صحيح", "Daif": "ضعيف",
    "Very Daif": "ضعيف جداً", "Mawdu": "موضوع", "Munkar": "منكر", "Shadh": "شاذ",
    "Sahih Isnaad": "صحيح الإسناد", "Daif Isnaad": "ضعيف الإسناد", "Hasan Isnaad": "حسن الإسناد",
    "Very Daif Isnaad": "ضعيف الإسناد جداً", "Sahih Hadith": "صحيح", "Sahih Mutawatir": "صحيح متواتر",
    "Sahih Lighairihi": "صحيح لغيره", "Sahih Matn": "صحيح المتن", "Hasan Sahih Isnaad": "حسن صحيح الإسناد",
    "Sahih Maqtu": "صحيح مقطوع", "Sahih Muquf": "صحيح موقوف", "Sahih Isnaad Maqtu": "صحيح الإسناد مقطوع",
    "Sahih Isnaad Mauquf": "صحيح الإسناد موقوف", "Sahih Isnaad Mursal": "صحيح الإسناد مرسل",
    "Hasan Maqtu": "حسن مقطوع", "Daif Maqtu": "ضعيف مقطوع", "Daif Muquf": "ضعيف موقوف",
    "Daif Isnaad Maqtu": "ضعيف الإسناد مقطوع", "Daif Munkar": "ضعيف منكر", "Munkar Daif": "منكر ضعيف",
    "Maqtu": "مقطوع", "Mursal": "مرسل", "Sanad Daif": "ضعيف السند", "Isnaad Malool": "معلول الإسناد",
}


def _rating(label: str) -> str:
    """ok / warn / bad from the scholar's own grading words."""
    l = label.lower()
    if any(k in l for k in ("daif", "mawdu", "munkar", "shadh", "malool")):
        return "bad"
    if any(k in l for k in ("maqtu", "muquf", "mauquf", "mursal")):
        return "warn"   # not the Prophet's words, or chain cut
    if "sahih" in l or "hasan" in l:
        return "ok"
    return "warn"

# Very common words carry no signal for finding a hadith.
_STOP = set(normalize(
    "في من على الى عن ان لا ما قال الله عليه وسلم رسول النبي و ثم او هو هي "
    "كان قد لم لن اذا هذا ذلك التي الذي حدثنا اخبرنا حدثني عن ابي ابن بن"
).split())


def _bare(w: str) -> str:
    """Strip attached particles so «والكلمه» and «الكلمه» compare equal:
    و/ف (and), ب/ل/ك (prepositions), ال (the)."""
    if len(w) > 3 and w[0] in "وف":
        w = w[1:]
    if len(w) > 4 and w[0] in "بلك" and w[1:3] == "ال":
        w = w[1:]
    if len(w) > 4 and w.startswith("ال"):
        w = w[2:]
    elif len(w) > 4 and w.startswith("لل"):
        w = w[2:]
    return w


def _with_particles(w: str) -> list[str]:
    """«الكلمه» -> also «والكلمه», «فالكلمه», «بالكلمه», «للكلمه»… for retrieval."""
    forms = [w, "و" + w, "ف" + w]
    if w.startswith("ال"):
        forms += ["ب" + w, "وب" + w, "ك" + w, "لل" + w[2:], "ولل" + w[2:]]
    else:
        forms += ["ب" + w, "ل" + w]
    return forms


@dataclass
class HadithMatch:
    book: str
    number: str
    score: int
    text: str            # the matn (wording) as it appears in the book
    coverage: float = 1.0
    semantic: bool = False  # found by meaning (embeddings), not by wording
    grade: str = ""      # the scholar's grading, in Arabic
    grader: str = ""     # who gave it
    rating: str = "ok"   # ok / warn / bad, derived from the grading words
    english: str = ""     # English translation of the hadith, if available


@dataclass
class HadithResult:
    status: str                       # "found", "similar", "not_found", "too_short"
    input_text: str
    matches: list[HadithMatch] = field(default_factory=list)


def _matn(text: str) -> str:
    """Best-effort extraction of the Prophet's words from a full narration."""
    quoted = re.findall(r'"‏?\s*(.+?)\s*‏?"', text.replace("“", '"').replace("”", '"'))
    quoted = [q for q in quoted if len(q) > 15]
    if quoted:
        return " … ".join(quoted)
    return text


class HadithIndex:
    def __init__(self, data_dir: Path = DATA_DIR):
        self.docs: list[dict] = []
        for key, title in BOOKS.items():
            data = json.loads((data_dir / f"{key}.json").read_text("utf-8"))
            for h in data["hadiths"]:
                if not h["text"].strip():
                    continue
                number = str(h.get("arabicnumber") or h["hadithnumber"])
                number = number.split(".")[0]  # 52.10 -> 52
                if key in SAHIHAIN:
                    grade, grader, rating = "صحيح", "بإجماع الأمة على صحة ما في الصحيحين", "ok"
                else:
                    albani = next((g["grade"] for g in h["grades"] if g["name"] == "Al-Albani"), None)
                    if albani:
                        grade, grader, rating = _GRADE_AR.get(albani, albani), "الألباني", _rating(albani)
                    else:
                        grade, grader, rating = "غير محكوم عليه في البيانات", "", "warn"
                self.docs.append({
                    "book": title, "number": number, "text": h["text"], "norm": normalize(h["text"]),
                    "key": key, "hnum": str(h["hadithnumber"]),
                    "sahihain": key in SAHIHAIN, "grade": grade, "grader": grader, "rating": rating,
                })

        # Inverted index + document frequencies for BM25.
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for i, d in enumerate(self.docs):
            ws = d["norm"].split()
            self.lengths.append(len(ws))
            for w, tf in Counter(ws).items():
                self.postings[w].append((i, tf))
        self.avg_len = sum(self.lengths) / len(self.lengths)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for w, p in self.postings.items()}

        self.kv = None
        if MODEL_PATH.exists():
            self.kv = WordVectors(MODEL_PATH)

    def _semantic(self, q: str, content: list[str], weights: dict, total: float):
        """Find a hadith that says the same thing in other words.

        The query is expanded with the nearest words in meaning, candidates
        are retrieved with BM25, and each is scored by how much of the query's
        meaning it covers: a word counts if it appears, or if a word of the
        hadith is close to it in the embedding space.
        """
        kv = self.kv
        boost = {}
        for w in content:
            # the word itself, and the same word with an attached particle
            for form in _with_particles(w):
                boost[form] = max(boost.get(form, 0), 1.0 if form == w else 0.9)
            if w in kv.key_to_index:
                for nb, sim in kv.most_similar(w, topn=SEM_EXPAND_TOPN):
                    if sim >= SEM_WORD_SIM and nb not in _STOP:
                        for form in _with_particles(nb):
                            boost[form] = max(boost.get(form, 0), 0.6 * sim)
        candidates = dict.fromkeys(self._bm25(content) + self._bm25(list(boost), boost))
        best = None
        for i in candidates:
            doc = self.docs[i]["norm"]
            doc_words = set(doc.split())
            doc_bare = {_bare(d) for d in doc_words}
            in_vocab = [d for d in doc_words if d in kv.key_to_index]
            credit = {}
            for w, wt in weights.items():
                if w in doc_words or _bare(w) in doc_bare:
                    credit[w] = wt
                elif w in kv.key_to_index and in_vocab:
                    sim = float(kv.cosine_similarities(kv[w], kv[in_vocab]).max())
                    credit[w] = wt * sim if sim >= SEM_WORD_SIM else 0.0
                else:
                    credit[w] = 0.0
            cov = sum(credit.values()) / total
            # Posts often add a word ("في جماعة"). With 4+ meaningful words,
            # one uncovered word may be set aside if everything else matches well.
            missing = [w for w, c in credit.items() if c == 0.0]
            if len(weights) >= 4 and len(missing) == 1 and weights[missing[0]] <= 0.35 * total:
                rest_cov = sum(credit.values()) / (total - weights[missing[0]])
                if rest_cov >= 0.9:
                    cov = max(cov, rest_cov * 0.9)
            if cov >= SEM_COVERAGE and fuzz.partial_ratio(q, doc) >= SEM_MIN_PARTIAL:
                if best is None or cov > best[0]:
                    best = (cov, i)
        return best

    def _bm25(self, query_words, boost: dict | None = None) -> list[int]:
        k1, b = 1.5, 0.75
        scores: Counter[int] = Counter()
        for w in set(query_words):
            if w in _STOP or w not in self.postings:
                continue
            idf = self.idf[w] * (boost.get(w, 1.0) if boost else 1.0)
            for i, tf in self.postings[w]:
                denom = tf + k1 * (1 - b + b * self.lengths[i] / self.avg_len)
                scores[i] += idf * tf * (k1 + 1) / denom
        return [i for i, _ in scores.most_common(TOP_CANDIDATES)]

    def search(self, text: str, top: int = 3, by_meaning: bool = True) -> HadithResult:
        q = normalize(text)
        qw = q.split()
        if len(qw) < MIN_WORDS:
            return HadithResult(status="too_short", input_text=text)

        content = [w for w in dict.fromkeys(qw) if w not in _STOP] or qw
        max_idf = max(self.idf.values())
        weights = {w: self.idf.get(w, max_idf) for w in content}
        total = sum(weights.values())

        ranked = []
        for i in self._bm25(qw):
            doc_words = set(self.docs[i]["norm"].split())
            doc_bare = {_bare(d) for d in doc_words}
            covered = sum(wt for w, wt in weights.items()
                          if w in doc_words or _bare(w) in doc_bare
                          or any(fuzz.ratio(w, d) >= 85 for d in doc_words if abs(len(d) - len(w)) <= 2))
            coverage = covered / total
            score = fuzz.partial_ratio(q, self.docs[i]["norm"])
            ranked.append((round(score), round(coverage, 2), i))
        # Best match first; on a tie prefer the Sahihs.
        ranked.sort(key=lambda r: (r[1] >= SIMILAR_COVERAGE, r[0] * r[1], self.docs[r[2]]["sahihain"]), reverse=True)

        matches = []
        seen = set()
        for score, coverage, i in ranked:
            d = self.docs[i]
            key = (d["book"], d["number"])
            if key in seen:
                continue
            seen.add(key)
            matches.append(HadithMatch(d["book"], d["number"], score, _matn(d["text"]), coverage,
                                       False, d["grade"], d["grader"], d["rating"],
                                       english=translation.hadith(d["key"], d["hnum"])))
            if len(matches) == top:
                break

        def is_found(m): return m.score >= FOUND_SCORE and m.coverage >= FOUND_COVERAGE
        def is_similar(m): return m.score >= SIMILAR_SCORE and m.coverage >= SIMILAR_COVERAGE

        if matches and is_found(matches[0]):
            status = "found"
            matches = [m for m in matches if is_found(m)]
        elif matches and is_similar(matches[0]):
            status = "similar"
            matches = [m for m in matches if is_similar(m)]
        else:
            status = "not_found"
            matches = []

        # Not found by wording: try by meaning. A meaning match is never
        # reported as "found"; it is shown as a close text to compare with.
        if by_meaning and status != "found" and self.kv is not None and len(content) >= 2:
            sem = self._semantic(q, content, weights, total)
            if sem and (status == "not_found" or sem[0] > matches[0].coverage + 0.05):
                cov, i = sem
                d = self.docs[i]
                m = HadithMatch(d["book"], d["number"], round(fuzz.partial_ratio(q, d["norm"])), _matn(d["text"]),
                                round(cov, 2), True, d["grade"], d["grader"], d["rating"],
                                english=translation.hadith(d["key"], d["hnum"]))
                status, matches = "similar", [m] + [x for x in matches if (x.book, x.number) != (m.book, m.number)]
        return HadithResult(status=status, input_text=text, matches=matches)


_index: HadithIndex | None = None


def get_index() -> HadithIndex:
    global _index
    if _index is None:
        _index = HadithIndex()
    return _index
