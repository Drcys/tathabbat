"""Analyze a whole social-media post.

A post mixes ayat, hadiths and ordinary commentary. This module splits the
post into sentences, removes introductory phrases ("قال الله تعالى",
"قال رسول الله ﷺ" …), and sends each piece to the right checker:

* anything that matches the Quran is checked against the Mushaf;
* anything presented as a hadith (or that matches one) is looked up in the
  six books (with the scholar's grading), then in Dorar;
* everything else is ordinary text and is left alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein

from . import dorar
from .hadith import HadithResult, get_index as hadith_index
from .normalize import normalize
from .quran import QuranResult, get_index as quran_index

# Quran quote must match at least this share of its words to be treated as Quran.
QURAN_MIN_RATIO = 0.6
QURAN_MIN_WORDS = 4
# Text NOT presented as Quran (no "قال تعالى", no ﴿﴾) must be closer to an ayah
# to be treated as one: "الحمد لله على نعمة الإسلام" shares 3 of 5 words with
# Ibrahim 39 but is an ordinary dua, not a misquoted ayah.
QURAN_UNMARKED_RATIO = 0.75
QURAN_UNMARKED_RUN = 4

_GLORY = r"(?:تعال[ىيلا]{0,2}|سبحانه(?:\s+وتعالى)?|عز\s+وجل|جل\s+وعلا|جل\s+جلاله)"
_QURAN_INTRO = re.compile(
    r"^(?:و)?(?:قال|يقول|قوله)\s+(?:(?:الله|ربنا|الحق|المولى)(?:\s+" + _GLORY + r")?|" + _GLORY + r")\s*[:：]?\s*"
)
_HADITH_INTRO = re.compile(
    r"^(?:و)?(?:عن\s+\S+(?:\s+\S+)?\s+(?:رضي\s+الله\s+عنه(?:ا|ما|م)?\s+)?(?:قال|أن|ان)\s+)?"
    r"(?:قال|يقول|أن|ان)\s+(?:رسول\s+الله|النبي|المصطفى|الحبيب)\s*"
    r"(?:ﷺ|صلى\s+الله\s+عليه\s+وسلم|عليه\s+الصلاة\s+والسلام)?\s*[:：]?\s*"
)
# Everything up to the Prophet's name/salutation (and a following "قال") is
# the chain or the introduction, not the claimed words.
_HADITH_LEAD = re.compile(
    r"^.*(?:ﷺ|صلى\s+الله\s+عليه\s+وسلم|عليه\s+الصلاة\s+والسلام|رسول\s+الله|النبي)\s*(?:أنه\s+)?(?:قال|يقول)?\s*:?\s*"
)
_PROPHET = r"(?:ﷺ|صلى\s+الله\s+عليه\s+وسلم|رسول\s+الله|النبي|المصطفى|عليه\s+الصلاة\s+والسلام)"
# The text is *attributed* to the Prophet (said / narrated / "hadith"), not
# just mentions him ("لا تنسوا الصلاة على النبي ﷺ" is not a hadith).
_HADITH_MARKERS = re.compile(
    r"(?:قال|يقول|قوله|عن|حديث|ان|أن)\b[^.\n]{0,40}?" + _PROPHET + r"|" + _PROPHET + r"\s*(?:قال|:)"
    r"|(?:ال)?حديث\s+(?:ال)?قدسي"
)
_SPLIT = re.compile(r"[\n.!؟?؛]+")
_QUOTED = re.compile(r"[«\"“﴿{(]\s*(.+?)\s*[»\"”﴾})]")
# Words of the salutations and formulas that appear in brackets ("(صلى الله
# عليه وسلم)", "(رضي الله عنه)"): a bracket holding only these is not a quote.
_FORMULA = set("صلي الله عليه وسلم رضي عنه عنها عنهما عنهم تعالي سبحانه وتعالي عز وجل الصلاه والسلام".split())
# A reference line under the text ("رواه البخاري"، "صحيح مسلم: 780"، the
# Dorar card fields): it names the source, it is not a saying to check.
_SOURCE_LINE = re.compile(
    r"^(?:رواه|أخرجه|اخرجه|متفق عليه|الراوي|المحدث|المصدر|الصفحة أو الرقم|خلاصة حكم المحدث|"
    r"(?:صحيح|سنن|جامع|مسند|موطأ)\s+\S+\s*[:،]?\s*(?:رقم\s*)?[0-9٠-٩]*\s*$)"
)
_STRIP = "«»\"“”﴿﴾{}()[]:-–— \t*•"


@dataclass
class Segment:
    kind: str                 # "quran", "hadith" or "text"
    text: str
    quran: QuranResult | None = None
    hadith: HadithResult | None = None
    dorar: list | None = None
    dorar_error: bool = False
    from_image: bool = False
    uncertain: frozenset = frozenset()    # words the OCR read with low confidence
    nearest: dict | None = None           # for text attributed to the Quran but not in it

    def _unsure(self, text: str) -> bool:
        return bool(set(normalize(text).split()) & self.uncertain)

    @property
    def maybe_ocr_error(self) -> bool:
        """Text read from an image, and every difference can be explained by
        the reading: a near-identical word, or a word the OCR itself was not
        sure of. More likely a reading error than an altered ayah."""
        if not (self.from_image and self.kind == "quran" and self.quran.differences):
            return False

        ayah_words = set(normalize(self.quran.correct_text or "", uthmani=True).split())
        read_words = set(normalize(self.text).split()) | set(normalize(self.text, uthmani=True).split())

        def near(a, b):
            # A misplaced dot or a letter misread (يتق/يثق, الصبر/الصير).
            a, b = normalize(a), normalize(b, uthmani=True)
            return Levenshtein.distance(a, b) <= (1 if len(b) <= 5 else 2)

        def word_explained(written, correct):
            return normalize(written) in self.uncertain or near(written, correct)

        def explained(d):
            written, correct = d.written.split(), d.correct.split()
            if d.kind == "replace":
                if len(written) == len(correct):
                    return all(word_explained(w, c) for w, c in zip(written, correct))
                return all(normalize(w) in self.uncertain or normalize(w) in ayah_words for w in written)
            if d.kind == "extra":
                # unsure words, or words of this same ayah read in the wrong place
                return all(normalize(w) in self.uncertain or normalize(w) in ayah_words for w in written)
            # a missing word: dropped by an unsure reading, or read elsewhere in the line
            return bool(self.uncertain) or all(normalize(c, uthmani=True) in read_words for c in correct)
        return all(explained(d) for d in self.quran.differences)

    @property
    def reading_doubtful(self) -> bool:
        """A hadith read from an image where many words were uncertain: a
        "not found" may be the reading's fault, not the text's."""
        if not (self.from_image and self.uncertain):
            return False
        ws = normalize(self.text).split()
        return bool(ws) and sum(w in self.uncertain for w in ws) / len(ws) >= 0.2

    @property
    def verdict(self) -> str:
        """One of: ok, warn, bad, neutral."""
        if self.kind == "quran":
            if self.quran.status == "verified":
                return "ok"
            if self.quran.status == "not_found":
                # from an image, the reading itself may be the problem
                return "warn" if self.from_image else "bad"
            return "warn" if self.maybe_ocr_error else "bad"
        if self.kind == "hadith":
            if self.hadith.status == "found":
                return self.hadith.matches[0].rating
            if self.reading_doubtful:
                return "warn"
            if self.hadith.status == "similar":
                # Not the exact wording; if the closest hadith is itself weak,
                # the claim is not established either.
                return "bad" if self.hadith.matches[0].rating == "bad" else "warn"
            # Not in the six books: show what Dorar's scholars said, if
            # reachable. The colour follows the scholar's grading words.
            if not self.dorar:
                return "bad"
            grade = normalize(self.dorar[0].grade)
            if any(k in grade for k in ("موضوع", "باطل", "لا اصل", "ضعيف", "منكر", "كذب")):
                return "bad"
            if any(k in grade for k in ("صحيح", "حسن")):
                return "ok"
            return "warn"
        return "neutral"


@dataclass
class Report:
    segments: list[Segment] = field(default_factory=list)

    @property
    def checked(self) -> list[Segment]:
        return [s for s in self.segments if s.kind != "text"]


def _pieces(post: str) -> list[tuple[str, bool, bool]]:
    """Split a post into (text, presented_as_hadith, presented_as_quran) pieces."""
    pieces = []
    for raw in _SPLIT.split(post):
        if not raw.strip():
            continue
        if _SOURCE_LINE.match(raw.strip(_STRIP)):
            continue
        marked = bool(_HADITH_MARKERS.search(raw))
        as_quran = "﴿" in raw or bool(_QURAN_INTRO.match(raw.strip(_STRIP)))
        # Ayah numbers like ﴿45﴾ or (45) are not quotes.
        raw = re.sub(r"[﴿(\[{]\s*[0-9٠-٩۰-۹]+\s*[﴾)\]}]", " ", raw)
        quoted = [q for q in _QUOTED.findall(raw)
                  if len(q.split()) >= 2 and not set(normalize(q).split()) <= _FORMULA
                  and not _SOURCE_LINE.match(q.strip(_STRIP))]   # "(رواه البخاري ومسلم)"
        if quoted:
            # Text inside quotation marks / Quran brackets is the claimed quote.
            for q in quoted:
                pieces.append((q.strip(_STRIP), marked, as_quran))
            continue
        text = raw.strip(_STRIP)
        if ":" in text and (marked or _QURAN_INTRO.match(text)):
            text = text.rsplit(":", 1)[1]
        text = _QURAN_INTRO.sub("", text.strip(_STRIP))
        text = _HADITH_INTRO.sub("", text).strip(_STRIP)
        if marked:
            rest = _HADITH_LEAD.sub("", text).strip(_STRIP)
            if len(rest.split()) >= 2:
                text = rest
        if text:
            pieces.append((text, marked, as_quran))
        elif marked or as_quran:
            pieces.append(("", marked, as_quran))   # an intro on its own: marks the next piece
    return pieces


_INTRO_WORD = re.compile(r"((?:قال|يقول)(?:\s+الله)?\s+)(\S{4,6})(?=\s*[:：؛]|\s+[﴿«(\[])")


def _fix_misread_intro(post: str) -> str:
    """An image reading may garble "تعالى" in "قال الله تعالى:" (تقالى، تحالى).
    The intro is then not recognised and its words look like additions to
    the ayah. A word one letter away from تعالى in that place is restored."""
    def fix(m):
        word = normalize(m.group(2))
        return m.group(1) + "تعالى" if Levenshtein.distance(word, "تعالي") <= 1 else m.group(0)
    return _INTRO_WORD.sub(fix, post)


def analyze(post: str, use_dorar: bool = True, from_image: bool = False,
            uncertain: set[str] | frozenset = frozenset()) -> Report:
    """Check every ayah and hadith in a post. For text read from an image,
    `uncertain` holds the (normalized) words the OCR was not sure of."""
    uncertain = frozenset(uncertain) if from_image else frozenset()
    if from_image:
        post = _fix_misread_intro(post)
    q_idx, h_idx = quran_index(), hadith_index()
    report = Report()
    pending_marker = False  # "قال رسول الله ﷺ:" on its own line marks the next piece
    pending_quran = False   # so does "قال الله تعالى:"

    for text, marked, as_quran in _pieces(post):
        if not text:
            pending_marker, pending_quran = pending_marker or marked, pending_quran or as_quran
            continue
        marked = marked or pending_marker
        as_quran = as_quran or pending_quran
        pending_quran = False
        words = normalize(text).split()

        # A bare intro line like "قال ﷺ" -> the next piece is the hadith.
        if marked and len(words) <= 4 and _HADITH_MARKERS.search(text):
            pending_marker = True
            report.segments.append(Segment("text", text))
            continue
        pending_marker = False

        # Text presented as Quran (﴿…﴾ or "قال تعالى") is checked from 3 words.
        if len(words) >= (3 if as_quran else QURAN_MIN_WORDS):
            q = q_idx.check(text)
            # Text presented as Quran may be matched more loosely (an image
            # reading can garble a word or two of it).
            min_ratio = 0.5 if (as_quran and from_image) else QURAN_MIN_RATIO
            if q.status in ("verified", "altered") and q.match_ratio >= min_ratio and (
                    as_quran or q.status == "verified" or q.match_ratio >= QURAN_UNMARKED_RATIO
                    or q.longest_run >= QURAN_UNMARKED_RUN):
                report.segments.append(Segment("quran", text, quran=q, from_image=from_image, uncertain=uncertain))
                continue

        # Search by meaning only for text attributed to the Prophet ﷺ: ordinary
        # sentences must not be matched to a hadith just because they are similar.
        h = h_idx.search(text, by_meaning=marked)
        if h.status in ("found", "similar") or (marked and h.status == "not_found"):
            seg = Segment("hadith", text, hadith=h, from_image=from_image, uncertain=uncertain)
            if h.status != "found" and use_dorar:
                entries = dorar.lookup(text)
                seg.dorar_error = entries is None
                seg.dorar = entries or []
            report.segments.append(seg)
            continue

        # Presented as Quran ("قال تعالى" / ﴿…﴾) but not in the Mushaf: a
        # saying wrongly attributed to the Quran. Say so, and show the
        # nearest ayah if there is one.
        if as_quran and len(words) >= 3:
            q = q_idx.check(text)
            seg = Segment("quran", text, quran=q, from_image=from_image, uncertain=uncertain)
            if q.status not in ("verified", "altered"):
                q.status = "not_found"
                near = q_idx.search_fragment(text, top=1)
                if near and near[0]["words_found"] >= 2:
                    seg.nearest = near[0]
            # (a loose match is shown with its differences: part of an ayah
            # with words added to it)
            report.segments.append(seg)
            continue

        report.segments.append(Segment("text", text))
    return report
