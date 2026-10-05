"""Quran verse checker.

Given a piece of text that is presented as Quran, find where it comes from
and report any word that differs from the reference Mushaf text.

How it works (plain version):
1. The whole Quran is flattened into one long list of normalized words, and
   every word remembers which surah and ayah it belongs to.
2. For the input text we look up every pair of consecutive words in an
   index. Each hit "votes" for a place in the Quran where the quote could
   start. This finds the right location in milliseconds, even when the
   quote spans several ayat (a whole surah, for example).
3. At the best few locations we align the input against the Quran word by
   word, and keep the location with the most matching words.
4. Every difference inside the aligned region is reported as an
   alteration: a replaced word, a missing word or an added word.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein
from pathlib import Path

from .normalize import _OPEN_TANWEEN_GAP, words as to_words

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# A quote shorter than this is too ambiguous to judge (e.g. "الله أكبر").
MIN_WORDS = 3
# Share of input words that must match the Quran to call it a quote at all.
MIN_MATCH_RATIO = 0.5
# How many candidate locations we align in detail.
TOP_CANDIDATES = 8
# Extra words taken around a candidate window, to absorb insertions.
SLACK = 4


@dataclass
class Difference:
    kind: str            # "replace", "missing" or "extra"
    written: str         # what the post says
    correct: str         # what the Mushaf says
    surah: int
    ayah: int


@dataclass
class QuranResult:
    status: str                      # "verified", "altered", "not_found", "too_short"
    input_text: str
    match_ratio: float = 0.0
    longest_run: int = 0                 # longest run of consecutive matching words
    surah: int | None = None
    ayah_from: int | None = None
    ayah_to: int | None = None
    surah_name: str | None = None
    correct_text: str | None = None  # Uthmani text of the matched ayat
    differences: list[Difference] = field(default_factory=list)

    @property
    def reference(self) -> str | None:
        if self.surah is None:
            return None
        if self.ayah_from == self.ayah_to:
            return f"{self.surah_name} {self.ayah_from}"
        return f"{self.surah_name} {self.ayah_from}–{self.ayah_to}"


class _Layer:
    """The whole Quran as one flat word list, in one spelling, with indexes."""

    def __init__(self, verses: list[dict], uthmani: bool):
        self.uthmani = uthmani
        self.words: list[str] = []
        self.display: list[str] = []   # same word with its diacritics, for showing
        self.loc: list[tuple[int, int]] = []
        for v in verses:
            for token in _OPEN_TANWEEN_GAP.sub(r"\1", v["text"]).split():
                for w in to_words(token, uthmani):
                    self.words.append(w)
                    self.display.append(token)
                    self.loc.append((v["chapter"], v["verse"]))

        # Bigram index: (word, next word) -> positions in the flat list.
        self.bigrams: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i in range(len(self.words) - 1):
            self.bigrams[(self.words[i], self.words[i + 1])].append(i)
        # Unigram fallback for very short or heavily altered quotes.
        self.unigrams: dict[str, list[int]] = defaultdict(list)
        for i, w in enumerate(self.words):
            self.unigrams[w].append(i)

    def candidate_starts(self, q: list[str]) -> list[int]:
        votes: Counter[int] = Counter()
        for i in range(len(q) - 1):
            for p in self.bigrams.get((q[i], q[i + 1]), ()):
                votes[p - i] += 2
        if not votes:
            # No shared word pair: fall back to rare single words.
            for i, w in enumerate(q):
                hits = self.unigrams.get(w, ())
                if 0 < len(hits) <= 50:
                    for p in hits:
                        votes[p - i] += 1
        return [s for s, _ in votes.most_common(TOP_CANDIDATES)]

    def align(self, q: list[str], start: int):
        lo = max(0, start - SLACK)
        hi = min(len(self.words), start + len(q) + SLACK)
        sm = SequenceMatcher(None, q, self.words[lo:hi], autojunk=False)
        matched = sum(b.size for b in sm.get_matching_blocks())
        return matched, lo, sm


class QuranIndex:
    def __init__(self, data_dir: Path = DATA_DIR):
        simple = json.loads((data_dir / "quran_simple.json").read_text("utf-8"))["quran"]
        uthmani = json.loads((data_dir / "quran_uthmani.json").read_text("utf-8"))["quran"]
        names = json.loads((data_dir / "surah_names.json").read_text("utf-8"))

        self.surah_names: dict[int, str] = {int(k): v for k, v in names.items()}
        self.uthmani: dict[tuple[int, int], str] = {
            (v["chapter"], v["verse"]): v["text"] for v in uthmani
        }
        self.simple: dict[tuple[int, int], str] = {(v["chapter"], v["verse"]): v["text"] for v in simple}

        # The Quran is indexed twice: in plain (imla'i) spelling, which is how
        # most posts write it, and in the Uthmani script, which is what people
        # copy from Mushaf apps. A quote is checked against both.
        self.layers = [_Layer(simple, uthmani=False), _Layer(uthmani, uthmani=True)]

    def check(self, text: str) -> QuranResult:
        best = None
        for layer in self.layers:
            q = to_words(text, layer.uthmani)
            if len(q) < MIN_WORDS:
                return QuranResult(status="too_short", input_text=text)
            for start in layer.candidate_starts(q):
                matched, lo, sm = layer.align(q, start)
                if best is None or matched / len(q) > best[0] / len(best[4]):
                    best = (matched, lo, sm, layer, q)

        if best is None:
            return QuranResult(status="not_found", input_text=text)

        matched, lo, sm, L, q = best
        ratio = matched / len(q)
        if ratio < MIN_MATCH_RATIO:
            return QuranResult(status="not_found", input_text=text, match_ratio=round(ratio, 2))

        blocks = [b for b in sm.get_matching_blocks() if b.size]
        q_first, q_last = blocks[0].a, blocks[-1].a + blocks[-1].size   # input span
        w_first, w_last = blocks[0].b, blocks[-1].b + blocks[-1].size   # window span

        diffs: list[Difference] = []
        for op, i1, i2, j1, j2 in sm.get_opcodes():
            if op == "equal":
                continue
            # Before the first / after the last matching word: Mushaf words
            # the quote does not include just mean it starts or ends mid-ayah,
            # so they are not errors. Words the quote adds there ARE errors
            # (e.g. a word appended to the end of an ayah).
            at_start = i2 <= q_first and j2 <= w_first
            at_end = i1 >= q_last and j1 >= w_last
            if at_start or at_end:
                if i1 < i2:
                    # Pair input words with the neighbouring Mushaf words
                    # (from the side touching the match). A close look-alike
                    # is a replaced/misspelt word; anything else was added.
                    qs, ws = list(range(i1, i2)), list(range(j1, j2))
                    if at_start:
                        qs.reverse()
                        ws.reverse()
                    paired = set()
                    for qi, wj in zip(qs, ws):
                        a, b = q[qi], L.words[lo + wj]
                        close = fuzz.ratio(a, b) >= 70 or \
                            Levenshtein.distance(a, b) <= (1 if len(b) <= 5 else 2)
                        if not close:
                            break
                        surah, ayah = L.loc[lo + wj]
                        diffs.append(Difference("replace", q[qi], L.display[lo + wj], surah, ayah))
                        paired.add(qi)
                    rest = [q[k] for k in range(i1, i2) if k not in paired]
                    if rest:
                        edge = w_first if at_start else w_last - 1
                        surah, ayah = L.loc[lo + edge]
                        diffs.append(Difference("extra", " ".join(rest), "", surah, ayah))
                continue
            anchor = lo + (j1 if j1 < j2 else max(j1 - 1, w_first))
            surah, ayah = L.loc[min(anchor, len(L.loc) - 1)]
            written = " ".join(q[i1:i2])
            correct = " ".join(L.display[lo + j1: lo + j2])
            kind = {"replace": "replace", "delete": "extra", "insert": "missing"}[op]
            diffs.append(Difference(kind, written, correct, surah, ayah))

        first_loc = L.loc[lo + w_first]
        last_loc = L.loc[lo + w_last - 1]
        surah = first_loc[0]
        ayat = range(first_loc[1], last_loc[1] + 1) if last_loc[0] == surah else [first_loc[1]]
        # (the data splits the tanween alif off its word: "عِلۡمࣰ ا"; joined for display)
        correct_text = " ".join(_OPEN_TANWEEN_GAP.sub(r"\1", self.uthmani[(surah, a)]) + f" ﴿{a}﴾" for a in ayat)

        return QuranResult(
            status="altered" if diffs else "verified",
            input_text=text,
            match_ratio=round(ratio, 2),
            longest_run=max(b.size for b in blocks),
            surah=surah,
            ayah_from=first_loc[1],
            ayah_to=last_loc[1] if last_loc[0] == surah else first_loc[1],
            surah_name=self.surah_names.get(surah, str(surah)),
            correct_text=correct_text,
            differences=diffs,
        )

    def search_fragment(self, text: str, top: int = 5) -> list[dict]:
        """Ayat that contain the words a user could make out of an image
        (decorative calligraphy, a blurred photo). One or two words are
        enough; a word may be slightly misspelt. Returns the best ayat with
        their reference and text."""
        from rapidfuzz.distance import Levenshtein

        L = self.layers[0]
        q = list(dict.fromkeys(to_words(text)))
        if not q:
            return []
        vocab = set(L.unigrams)
        quality: dict[tuple[int, int], float] = defaultdict(float)
        for w in q:
            # A misread letter or two: up to 1 edit for short words, 2 for long.
            max_edits = 0 if len(w) < 3 else 1 if len(w) <= 5 else 2
            best: dict[tuple[int, int], float] = {}
            for v in ([w] if w in vocab else []) + [
                v for v in vocab
                if max_edits and v != w and abs(len(v) - len(w)) <= max_edits
                and Levenshtein.distance(v, w, score_cutoff=max_edits) <= max_edits
            ]:
                score = 1.0 if v == w else 0.8
                for p in L.unigrams[v]:
                    a = L.loc[p]
                    best[a] = max(best.get(a, 0), score)
            for a, sc in best.items():
                quality[a] += sc
        if not quality:
            return []
        query = " ".join(q)
        top_quality = max(quality.values())
        cands = [a for a, sc in quality.items() if sc >= top_quality - 0.5]

        def rank(a):
            ayah = " ".join(to_words(self.simple[a]))
            # words that occur close together and in the same order rank first
            return (-quality[a], -fuzz.partial_ratio(query, ayah), len(ayah))

        cands.sort(key=rank)
        return [
            {"surah": a[0], "ayah": a[1], "reference": f"{self.surah_names.get(a[0], a[0])} {a[1]}",
             "text": self.simple[a], "uthmani": self.uthmani[a],
             "words_found": round(quality[a], 1), "words_asked": len(q)}
            for a in cands[:top]
        ]

_index: QuranIndex | None = None


def get_index() -> QuranIndex:
    """Load the Quran once and reuse it."""
    global _index
    if _index is None:
        _index = QuranIndex()
    return _index
