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
# The same intro anywhere before a quote on its line.
_QURAN_SAID = re.compile(
    r"(?<![ء-ي])و?(?:قال|يقول|قوله)\s+(?:(?:الله|ربنا|الحق|المولى)(?![ء-ي])|" + _GLORY + r")"
)
_HADITH_INTRO = re.compile(
    r"^(?:و)?(?:عن\s+\S+(?:\s+\S+)?\s+(?:رضي\s+الله\s+عنه(?:ا|ما|م)?\s+)?(?:قال|أن|ان)\s+)?"
    r"(?:قال|يقول|أن|ان)\s+(?:رسول\s+الله|النبي(?![\u0621-\u064A])|المصطفى|الحبيب)\s*"
    r"(?:ﷺ|صلى\s+الله\s+عليه\s+وسلم|عليه\s+الصلاة\s+والسلام)?\s*[:：]?\s*"
)
# Everything up to the Prophet's name/salutation (and a following "قال") is
# the chain or the introduction, not the claimed words.
_HADITH_LEAD = re.compile(
    r"^.*(?:ﷺ|صلى\s+الله\s+عليه\s+وسلم|عليه\s+الصلاة\s+والسلام|رسول\s+الله|النبي(?![\u0621-\u064A]))\s*(?:أنه\s+)?(?:قال|يقول)?\s*:?\s*"
)
_PROPHET = r"(?:ﷺ|صلى\s+الله\s+عليه\s+وسلم|رسول\s+الله|النبي(?![\u0621-\u064A])|المصطفى|عليه\s+الصلاة\s+والسلام)"
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
    misattributed: str = ""               # "quran_as_hadith" or "hadith_as_quran"

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
                # a space lost or added by the reading: "أوما" for "أو ما"
                if normalize("".join(written)) == normalize("".join(correct), uthmani=True):
                    return True
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
        if not self.from_image:
            return False
        ws = normalize(self.text).split()
        if self.uncertain and ws and sum(w in self.uncertain for w in ws) / len(ws) >= 0.2:
            return True
        # A word one letter away from a word of the closest hadith ("تدركيم"
        # for "تدركهم"), or two of its words read as one ("غيربينة"): a
        # misreading, so the reading is in doubt, not the hadith.
        if self.hadith and self.hadith.matches and self.hadith.matches[0].score >= 80:
            top = normalize(self.hadith.matches[0].text).split()
            top_set, glued = set(top), {a + b for a, b in zip(top, top[1:])}
            for w in ws:
                if w in top_set or len(w) < 3:
                    continue
                if w in glued or any(abs(len(t) - len(w)) <= 1 and Levenshtein.distance(w, t) == 1
                                     for t in top_set):
                    return True
        return False

    @property
    def dorar_matches(self) -> list:
        """The Dorar narrations with the post's wording (all of them: several
        scholars may have graded the same words)."""
        q = normalize(self.text)
        if not self.dorar or len(q.split()) < 2:
            return []
        return [e for e in self.dorar if fuzz.partial_ratio(q, normalize(e.text)) >= 90]

    @property
    def dorar_match(self):
        found = self.dorar_matches
        return found[0] if found else None

    @property
    def scholars_differ(self) -> bool:
        """Scholars in Dorar graded these same words differently."""
        return len({dorar_rating(e.grade) for e in self.dorar_matches} - {"warn"}) > 1

    @property
    def verdict(self) -> str:
        """One of: ok, warn, bad, neutral."""
        if self.misattributed == "hadith_as_quran":
            return "bad"          # presenting a saying as the Quran is the error
        if self.misattributed == "quran_as_hadith":
            return "warn"         # the words are right, the attribution is not
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
            # reachable. The colour follows the scholar's grading words, and
            # only for a narration that has the same wording as the post:
            # Dorar also returns related hadiths ("الطهور شطر الإيمان" for
            # "النظافة من الإيمان"), whose grading says nothing about this text.
            found = self.dorar_matches
            if not found:
                return "bad"
            ratings = {dorar_rating(e.grade) for e in found}
            if ratings == {"ok"}:
                return "ok"
            if ratings <= {"bad", "warn"} and "bad" in ratings:
                return "bad"
            # the scholars differ: show their words, the tool does not choose
            return "warn"
        return "neutral"


# A grading that denies the text is checked BEFORE the words "صحيح/حسن",
# because the denial contains them: "ليس بصحيح"، "لا يصح"، "ليس بحديث، لكن معناه صحيح".
_DENIED = ("ليس بصحيح", "ليس صحيح", "غير صحيح", "لا يصح", "لم يصح", "لا يثبت", "لم يثبت", "ليس بحديث",
           "ليس حديث", "لا اصل", "ليس له اصل", "موضوع", "باطل", "ضعيف", "منكر", "كذب", "مكذوب", "لا يعرف")


def dorar_rating(grade: str) -> str:
    """Colour of a scholar's grading in Dorar: bad / ok / warn."""
    g = normalize(grade)
    if any(normalize(k) in g for k in _DENIED):
        return "bad"
    if any(k in g for k in ("صحيح", "حسن")):
        return "ok"
    return "warn"


@dataclass
class Report:
    segments: list[Segment] = field(default_factory=list)

    @property
    def checked(self) -> list[Segment]:
        return [s for s in self.segments if s.kind != "text"]


# A salutation or "رضي الله عنه" in brackets, also as an image reads it
# ("(صل الله عليه وسل)"، "(صلى انه عليه وسلم)"، "الله عنه)").
_FORMULA_IN_BRACKETS = re.compile(
    r"[(\[]?\s*(?:صل[ىي]?|رضي?)\s+\S+\s+(?:عليه(?:\s+وس\S*)?|عنه(?:ما|م|ا)?)\s*[)\]]"
    r"|[(\[]\s*(?:صل[ىي]?|رضي?)\s+\S+\s+(?:عليه(?:\s+وس\S*)?|عنه(?:ما|م|ا)?)\s*[)\]]?"
    r"|(?<=\s)الله\s+عنه(?:ما|م|ا)?\s*[)\]]"
)
# "… رواه مسلم" / "(متفق عليه)" after an unquoted hadith names its source.
_TRAILING_SOURCE = re.compile(r"\s+[(\[]?\s*(?:رواه|أخرجه|اخرجه|متفق\s+عليه)(?:\s+[^\s()]+){0,5}\s*[)\]]?\s*$")
# A quote cut right after an open tanween (…إِذࣰ ا) can start with its lone alif.
_LONE_ALIF = re.compile("^[اى][ً-ْۖ-ۭ]*\\s+")


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
        quoted = [m for m in _QUOTED.finditer(raw)
                  if len(m.group(1).split()) >= 2 and not set(normalize(m.group(1)).split()) <= _FORMULA
                  and not _FORMULA_IN_BRACKETS.fullmatch(m.group(0).strip())
                  and not _SOURCE_LINE.match(m.group(1).strip(_STRIP))]   # "(رواه البخاري ومسلم)"
        if quoted:
            # Text inside quotation marks / Quran brackets is the claimed quote.
            # With several quotes on one line, each takes how it is presented
            # from the words right before it, not from the whole line:
            # «قال تعالى: ﴿…﴾ وقال ﷺ: «…»» holds an ayah and then a hadith.
            start = 0
            q_marked, q_quran = (marked, as_quran) if len(quoted) == 1 else (False, False)
            for m in quoted:
                inner = m.group(1)
                q_begin = m.start()
                # "«من رسول الله ﷺ: «من حسن إسلام…»": a quote opened twice
                # (a line read out of order); the quote is after the last «.
                if "«" in inner:
                    cut = inner.rfind("«")
                    q_begin = m.start(1) + cut
                    inner = inner[cut + 1:]
                context = raw[start:q_begin]
                start = m.end()
                if re.search("[ء-يﷺ]", context):
                    q_marked = bool(_HADITH_MARKERS.search(context))
                    q_quran = bool(_QURAN_SAID.search(context))
                # else: nothing before it on the line, it is presented like the quote before
                quote = _LONE_ALIF.sub("", inner.strip(_STRIP))
                pieces.append((quote, q_marked, q_quran or m.group(0).startswith("﴿")))
            continue
        text = _TRAILING_SOURCE.sub("", raw.strip(_STRIP)).strip(_STRIP)
        if ":" in text and (marked or _QURAN_INTRO.match(text)):
            text = text.rsplit(":", 1)[1]
        text = _QURAN_INTRO.sub("", text.strip(_STRIP))
        text = _HADITH_INTRO.sub("", text).strip(_STRIP)
        if marked:
            # "(رضي الله عنه)"، "(صلى الله عليه وسلم)" in the chain are not words of the hadith
            bare = _FORMULA_IN_BRACKETS.sub(" ", text)
            rest = _HADITH_LEAD.sub("", bare).strip(_STRIP)
            if len(rest.split()) >= 2:
                text = rest
            elif not rest.split() or rest.split() in (["قال"], ["يقول"], ["أنه", "قال"]):
                # only the chain ("عن أبي هريرة (رضي الله عنه) أن النبي ﷺ قال:"),
                # the hadith is on the next line
                text = ""
        text = _LONE_ALIF.sub("", text)
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
        for right in ("تعالى", "سبحانه"):
            if Levenshtein.distance(word, normalize(right)) <= 1:
                return m.group(1) + right
        return m.group(0)
    return _INTRO_WORD.sub(fix, post)


# The ligature ﷺ is often read by the OCR as a short run of letters ("كل"،
# "ل4"، "كَلَةٌ"، "كَمٌ") and "وسلم" as "وسل": the hadith then loses its intro.
_SAW_MISREAD = re.compile(
    r"((?:^|\s)(?:قال|رسول\s+الل[هة]?|النبي|الني|الي))\s+(?!(?:له|لهم|لها|لي|لنا|لكم|لك|كلا|كلها|كله)\s*[:«\"(])"
    r"[كل](?:[ً-ْ]*[ء-ي0-9]){0,2}[ً-ْ]*(?=\s*[:«\"(])"
)
_SALLAM_MISREAD = re.compile(r"(?<=الله عليه )وس[\u0621-\u064A]{0,2}(?![\u0621-\u064A])")
_NABI_MISREAD = re.compile(r"(?<![\u0621-\u064A])((?:قال|عن|أن|ان) )ال[ن]?ي(?=\s)")


def _fix_misread_salutation(post: str) -> str:
    post = _SALLAM_MISREAD.sub("وسلم", post)
    post = _NABI_MISREAD.sub(r"\1النبي", post)
    return _SAW_MISREAD.sub(r"\1 ﷺ", post)


# A hadith qudsi says so in its text: "قال الله عز وجل"، "يقول ربكم". A mere
# mention of Allah ("إن الله عز وجل حرم عليكم…") does not make it one.
_QUDSI = re.compile(r"(?:قال|يقول) (?:الله|ربكم|ربك|ربه|ربنا)")

_COMMON = {"الله", "الذي", "التي", "الذين", "على", "الي", "عن", "في", "من", "ما", "لا", "ان"}


def _shares_most_words(text: str, ayah: str) -> bool:
    words = [w for w in normalize(text).split() if len(w) >= 3 and w not in _COMMON]
    ayah_words = set(normalize(ayah).split())
    return len(words) >= 2 and sum(w in ayah_words for w in words) >= 0.6 * len(words)


# A piece made only of these words is an intro, never the text of a hadith
# ("قال رسول الله صلى الله عليه وسلم" cut off from its quote).
_INTRO_ONLY = _FORMULA | set("قال يقول رسول النبي عن ان انه وقال".split())

_PERSIAN = str.maketrans({"ی": "ي", "ې": "ي", "ک": "ك", "ہ": "ه", "ۃ": "ة", "ە": "ه"})
_MARKS = "[ً-ْ]*"
_ALLAH_MARKED = re.compile("ا" + _MARKS + "ل" + _MARKS + "ل" + _MARKS + "ه")
_SALLA = re.compile(r"\bصل[يى]?(?=\s+الله\s+عليه)")
_WA_SALLAM = re.compile(r"(?<=عليه)\s+و\s+سلم")
_NEXT_INTRO = re.compile(
    r"([»﴾”\"]|\(\s*متفق\s+عليه\s*\)|(?<![ء-ي])(?:رواه|أخرجه)\s+(?:الإمام\s+)?[ء-ي]+(?:\s+و[ء-ي]+)?)"
    r"(?:[^\S\n]|[^\w\s«»﴿﴾\"“”()])+(?=و?(?:قال|يقول|عن)\s)"
)


def _canon(post: str) -> str:
    """Typing variants that do not change what a post says but would hide
    its intros: Persian letters (ی ک ہ), tatweel (عـن), "اللّه" with a shadda,
    "صلي/صل الله عليه وسلم" and "و سلم". The diacritics of a quote are kept
    (the Uthmani check needs them)."""
    post = post.translate(_PERSIAN).replace("ـ", "")
    post = _ALLAH_MARKED.sub(lambda m: "الله" if m.group(0) != "الله" else m.group(0), post)
    post = _SALLA.sub("صلى", post)
    post = _WA_SALLAM.sub(" وسلم", post)
    # Two items run together on one line: "… (متفق عليه) قال تعالى: ﴿…﴾",
    # "«…» رواه مسلم وقال ﷺ: «…»". A new intro after a closed quote or a
    # source starts a new line.
    return _NEXT_INTRO.sub(lambda m: m.group(1) + "\n", post)


def analyze(post: str, use_dorar: bool = True, from_image: bool = False,
            uncertain: set[str] | frozenset = frozenset()) -> Report:
    """Check every ayah and hadith in a post. For text read from an image,
    `uncertain` holds the (normalized) words the OCR was not sure of."""
    uncertain = frozenset(uncertain) if from_image else frozenset()
    post = _canon(post)
    if from_image:
        post = _fix_misread_intro(post)
        post = _fix_misread_salutation(post)
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
        if marked and (len(words) <= 4 and _HADITH_MARKERS.search(text) or set(words) <= _INTRO_ONLY):
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
                    or q.longest_run >= QURAN_UNMARKED_RUN) and not (
                    # A hadith that quotes part of an ayah ("لن يلج النار أحد صلى
                    # قبل طلوع الشمس وقبل غروبها") is the hadith, not a changed ayah.
                    marked and not as_quran and q.status != "verified"
                    and h_idx.search(text, by_meaning=True, ocr=from_image).status == "found"):
                seg = Segment("quran", text, quran=q, from_image=from_image, uncertain=uncertain)
                # An ayah presented as the Prophet's ﷺ saying ("قال رسول الله ﷺ: إن الله مع الصابرين")
                if marked and not as_quran and q.status == "verified":
                    seg.misattributed = "quran_as_hadith"
                report.segments.append(seg)
                continue

        # Search by meaning only for text attributed to the Prophet ﷺ: ordinary
        # sentences must not be matched to a hadith just because they are similar.
        h = h_idx.search(text, by_meaning=marked, ocr=from_image)
        # Text NOT attributed to the Prophet ﷺ is shown as a hadith only when
        # its exact wording is in the books: a dua like "اللهم اجعلنا من أهلها"
        # merely resembles some hadith and must not get a hadith's grading.
        if marked or as_quran:
            is_hadith = h.status in ("found", "similar") or marked
        else:
            is_hadith = h.status == "found"
        if is_hadith:
            seg = Segment("hadith", text, hadith=h, from_image=from_image, uncertain=uncertain)
            # A hadith presented as an ayah ("قال الله تعالى: ﴿إنما الأعمال بالنيات﴾").
            # Not for a hadith qudsi, whose words are Allah's and say so.
            if as_quran and not marked and h.status in ("found", "similar") and \
                    not _QUDSI.search(normalize(h.matches[0].text)):
                seg.misattributed = "hadith_as_quran"
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
                # only an ayah that shares most of the meaningful words, not
                # just "الله" and "في" (counted exactly, not by similarity)
                if near and _shares_most_words(text, near[0]["text"]):
                    seg.nearest = near[0]
            # (a loose match is shown with its differences: part of an ayah
            # with words added to it)
            report.segments.append(seg)
            continue

        report.segments.append(Segment("text", text))
    return report
