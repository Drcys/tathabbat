"""Read the text of a post from an image, in any Arabic script.

Engines, chosen by the kind of writing in the image:
1. Tesseract OCR with the Arabic model (local, no key). It reads printed
   text letter by letter and never "corrects" a word, which matters here: a
   tampered ayah must be read as written. Used first for every image.
2. A vision-language model (Claude or Gemini) when Tesseract's reading is not
   reliable, which is what happens with decorative calligraphy (ثلث، ديواني،
   كوفي), handwriting, stacked letters or text over a busy photo. These models
   read any script, and are told to copy the letters as written, not from memory.

OCR makes mistakes (a dot moved, a letter misread). The app therefore shows
the extracted text so the user can correct it before checking, and the
report treats one-letter differences as a possible reading error rather than
an alteration of the Quran.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field

from PIL import Image, ImageOps

# Several model names per provider: if one is retired or not enabled for the
# key, the next is tried. An environment variable puts a chosen model first.
GEMINI_MODELS = [m for m in (os.environ.get("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-flash-latest",
                             "gemini-2.5-flash-lite", "gemini-2.0-flash") if m]
CLAUDE_MODELS = [m for m in (os.environ.get("CLAUDE_MODEL"), "claude-sonnet-5", "claude-sonnet-4-5",
                             "claude-haiku-4-5-20251001") if m]
# The Arabic Tesseract model ships with the project (tessdata_best, Apache-2.0),
# so no language pack has to be installed separately.
TESSDATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "tessdata")
_PROMPT = (
    "اقرأ كل النص العربي في هذه الصورة. قد يكون مكتوباً بخط زخرفي (الثلث، الديواني، الكوفي، "
    "الفارسي، الرقعة) أو بخط اليد أو فوق صورة، وقد تتراكب الحروف فوق بعضها. "
    "اقرأ الحروف كما رُسمت فعلاً، ولا تكمل النص من حفظك ولا تصحح أي كلمة حتى لو بدت آية أو "
    "حديثاً فيه خطأ، فالمطلوب كشف الأخطاء لا إخفاؤها. إن لم تتبيّن كلمة فاكتب مكانها [؟]. "
    "تجاهل أي كتابة غير عربية مثل أسماء المواقع والعلامات المائية. "
    "حافظ على ترتيب الأسطر من الأعلى إلى الأسفل، وأخرج النص فقط دون أي شرح أو علامات تنسيق."
)
VISION_ENGINES = ("claude", "gemini")


def _tesseract_path() -> str | None:
    """tesseract on PATH, or the default Windows install location."""
    found = shutil.which("tesseract")
    if found:
        return found
    for path in (r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                 r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"):
        if os.path.exists(path):
            return path
    return None


def available_engines(gemini_key: str | None = None, claude_key: str | None = None) -> list[str]:
    """Engines in the order they are tried."""
    engines = []
    if _tesseract_path():
        engines.append("tesseract")
    if claude_key:
        engines.append("claude")
    if gemini_key:
        engines.append("gemini")
    return engines


def _for_vision(image_bytes: bytes, mime: str) -> tuple[bytes, str]:
    """PNG/JPEG/WebP within the size vision APIs accept; small images are
    enlarged so thin calligraphy strokes stay visible."""
    img = Image.open(io.BytesIO(image_bytes))
    longest = max(img.size)
    if mime in ("image/png", "image/jpeg", "image/webp") and 400 <= longest <= 1568 and len(image_bytes) < 4_000_000:
        return image_bytes, mime
    img = img.convert("RGB")
    scale = 1568 / longest if longest > 1568 else min(3, 800 / longest) if longest < 400 else 1
    if scale != 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue(), "image/png"


def _claude(image_bytes: bytes, mime: str, api_key: str) -> str:
    import base64

    import requests

    data, mime = _for_vision(image_bytes, mime)
    payload = base64.b64encode(data).decode()
    last = None
    for model in CLAUDE_MODELS:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": model,
                "max_tokens": 1500,
                "temperature": 0,
                "messages": [{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": mime, "data": payload}},
                    {"type": "text", "text": _PROMPT},
                ]}],
            },
            timeout=60,
        )
        if resp.status_code == 404:          # model name not available: try the next
            last = resp
            continue
        resp.raise_for_status()
        return "".join(b.get("text", "") for b in resp.json().get("content", []) if b.get("type") == "text")
    last.raise_for_status()
    return ""


def _gemini(image_bytes: bytes, mime: str, api_key: str) -> str:
    from google import genai
    from google.genai import types

    data, mime = _for_vision(image_bytes, mime)
    client = genai.Client(api_key=api_key)
    error = None
    for model in GEMINI_MODELS:
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[types.Part.from_bytes(data=data, mime_type=mime), _PROMPT],
                config=types.GenerateContentConfig(temperature=0),
            )
            return resp.text or ""
        except Exception as exc:
            # Only a missing model moves on; a bad key or quota stops here.
            if "404" not in str(exc) and "not found" not in str(exc).lower():
                raise
            error = exc
    raise error


def _arabic_words(text: str) -> int:
    return len(re.findall(r"[\u0621-\u064A]{2,}", text))


def _variants(img: Image.Image):
    """Versions of the image to try. Screenshots read best as they are;
    photos and small images need enlarging, contrast and black/white."""
    gray = ImageOps.grayscale(img)
    yield gray
    scale = max(1, min(4, round(1200 / max(gray.width, 1))))
    big = gray.resize((gray.width * scale, gray.height * scale), Image.LANCZOS) if scale > 1 else gray
    contrast = ImageOps.autocontrast(big, cutoff=2)
    if scale > 1:
        yield contrast
    for cut in (140, 200):
        bw = contrast.point(lambda p, c=cut: 255 if p > c else 0)
        yield bw
        yield ImageOps.invert(bw)


_VOCAB: set[str] | None = None


def _vocabulary() -> set[str]:
    """Every word of the Quran and the six hadith books (normalized)."""
    global _VOCAB
    if _VOCAB is None:
        from .hadith import get_index as hadith_index
        from .quran import get_index as quran_index
        _VOCAB = set(hadith_index().postings)
        for layer in quran_index().layers:
            _VOCAB.update(layer.words)
    return _VOCAB


_ARABIC = re.compile(r"[\u0621-\u064A]")
_ALNUM_JUNK = re.compile(r"[0-9A-Za-z\u0660-\u0669\[\]<>|{}#*+=%$@]")


def _line_counts(line: str) -> tuple[int, int]:
    """(real words, debris) in one OCR line.
    Real = a corpus word of 3+ letters (short words are too easy to hit by
    accident). Debris = unknown Arabic-looking words, digits, Latin, symbols."""
    from .normalize import normalize
    vocab = _vocabulary()
    real = junk = 0
    for tok in line.split():
        if _ARABIC.search(tok):
            w = normalize(tok)
            if len(w) >= 3 and w in vocab:
                real += 1
            elif len(w) >= 2 and w not in vocab:
                junk += 1
            elif _ALNUM_JUNK.search(tok) or len(w) == 1:
                junk += 1          # a lone letter is a broken fragment of a word
        elif _ALNUM_JUNK.search(tok):
            junk += 1
    return real, junk


_BIGRAMS = None


def _bigrams():
    """Hashes of every pair of consecutive words in the Quran and the six books.
    A reading whose words follow each other as they do in the sources is in
    the right order; lines read out of order break these pairs."""
    global _BIGRAMS
    if _BIGRAMS is None:
        import numpy as np

        from .hadith import get_index as hadith_index
        from .quran import get_index as quran_index
        hashes = set()
        for layer in quran_index().layers:
            ws = layer.words
            hashes.update(hash((a, b)) for a, b in zip(ws, ws[1:]))
        for d in hadith_index().docs:
            ws = d["norm"].split()
            hashes.update(hash((a, b)) for a, b in zip(ws, ws[1:]))
        _BIGRAMS = np.array(sorted(hashes), dtype=np.int64)
    return _BIGRAMS


def _pairs_in_sources(text: str) -> tuple[int, int]:
    """(word pairs that occur in the sources, longest unbroken chain of them)."""
    import numpy as np

    from .normalize import normalize
    ws = normalize(text).split()
    if len(ws) < 2:
        return 0, 0
    table = _bigrams()
    keys = np.array([hash(p) for p in zip(ws, ws[1:])], dtype=np.int64)
    pos = np.clip(np.searchsorted(table, keys), 0, len(table) - 1)
    hits = table[pos] == keys
    longest = run = 0
    for h in hits:
        run = run + 1 if h else 0
        longest = max(longest, run)
    return int(hits.sum()), longest


def _real_words(text: str) -> float:
    """Quality of a reading: real words, plus word pairs in the right order
    (and the longest such chain: lines read out of order break it), minus
    half the debris."""
    real = junk = 0
    for line in text.splitlines():
        r, j = _line_counts(line)
        real += r
        junk += j
    pairs, longest = _pairs_in_sources(text)
    # A line read twice (once whole, once in pieces) repeats word pairs.
    # Both readings of one image share the image's real repetitions, so only
    # the extra ones count against a reading.
    from .normalize import normalize
    ws = normalize(text).split()
    seen, repeats = set(), 0
    for p in zip(ws, ws[1:]):
        repeats += p in seen
        seen.add(p)
    return real + pairs + longest - 0.5 * junk - 2 * repeats


def _drop_noise_lines(text: str) -> str:
    """Remove lines that are mostly misread fragments (e.g. English text in the
    image read as Arabic). Blank lines are kept: they separate paragraphs."""
    kept = []
    for line in text.splitlines():
        if not line.strip():
            kept.append("")
            continue
        real, junk = _line_counts(line)
        if real == 0:
            # a short wrapped line like «الصين”» is kept, debris is not
            words_ = [t for t in line.split() if _ARABIC.search(t)]
            if words_ and len(words_) <= 2 and not _ALNUM_JUNK.search(line):
                kept.append(line)
            continue
        if real / (real + junk) >= 0.5:
            kept.append(line)
    return "\n".join(kept)


# A word Tesseract reads with less confidence than this is "uncertain": a
# verdict of "wrong" or "not found" that rests on it is shown as "check the
# reading" instead, so a misread letter never becomes an accusation.
WORD_CONFIDENCE = 70


def _run_tesseract(path: str, psm: int = 6) -> tuple[str, float, set[str]]:
    """(text, mean confidence 0-100 of the Arabic words, uncertain words) for one image file."""
    from .normalize import normalize
    base = path[:-4]
    subprocess.run(
        [_tesseract_path(), path, base, "--tessdata-dir", TESSDATA, "-l", "ara", "--psm", str(psm),
         "-c", "tessedit_create_txt=1", "-c", "tessedit_create_tsv=1"],
        capture_output=True, timeout=60,
    )
    with open(base + ".txt", encoding="utf-8") as fh:
        text = fh.read()
    confs, uncertain = [], set()
    with open(base + ".tsv", encoding="utf-8") as fh:
        for row in fh.read().splitlines()[1:]:
            cols = row.split("\t")
            if len(cols) == 12 and _ARABIC.search(cols[11]) and float(cols[10]) >= 0:
                confs.append(float(cols[10]))
                if float(cols[10]) < WORD_CONFIDENCE:
                    uncertain.update(normalize(cols[11]).split())
    return text, (sum(confs) / len(confs) if confs else 0.0), uncertain


def _tesseract_with_confidence(image_bytes: bytes) -> tuple[str, float, set[str]]:
    """Try a few preprocessed versions and keep the reading with the most real
    words in the right order. Also returns Tesseract's own confidence in that
    reading and the words it was unsure of."""
    from rapidfuzz import fuzz

    from .normalize import normalize
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    best, best_score, best_conf, best_unsure = "", -1, 0.0, set()
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "post.png")
        for n, variant in enumerate(_variants(img)):
            variant.save(path)
            # psm 6 reads a block of text; psm 11 finds text anywhere, which
            # keeps the line order right for tall naskh fonts (Amiri) and
            # short wrapped lines, where psm 6 can mix the lines up.
            readings = []
            for psm in (6, 11):
                out, conf, unsure = _run_tesseract(path, psm)
                if psm == 11:
                    # psm 11 puts every line in its own block; a wrapped
                    # sentence must be joined back into one paragraph.
                    out = "\n".join(line for line in out.splitlines() if line.strip())
                out = _drop_noise_lines(out)
                score = _real_words(out)
                readings.append(normalize(out))
                if score > best_score:
                    best, best_score, best_conf, best_unsure = out, score, conf, unsure
            # A clean screenshot reads well as is: when both ways of reading
            # the original image agree, stop early to stay fast.
            if n == 0 and len(readings[0].split()) >= 5 and fuzz.ratio(*readings) >= 95:
                break
    return _clean(_join_lines(best)), best_conf, best_unsure


def _tesseract(image_bytes: bytes) -> str:
    return _tesseract_with_confidence(image_bytes)[0]


def _join_lines(text: str) -> str:
    """Tesseract breaks long lines; keep blank lines as paragraph breaks."""
    paragraphs = re.split(r"\n\s*\n", text)
    return "\n".join(" ".join(p.split()) for p in paragraphs if p.strip())


_KEEP = set("«»\"“”﴿﴾():.،؟!ﷺ")


def _clean(text: str) -> str:
    """Drop OCR debris (stray marks, Latin noise, 1-2 letter fragments that are
    not real words) while keeping every real word and the punctuation the
    post analysis relies on."""
    from .normalize import normalize
    vocab = _vocabulary()
    lines = []
    for line in text.splitlines():
        kept = []
        for tok in line.split():
            # An opening bracket ﴿ or « is often read as a letter glued to
            # the word (إواستعينوا, ؤومن): drop it when that leaves a real word.
            m = re.match(r"^([#(\[«﴿]*[إؤ]|[#(\[«﴿]+)(?=[\u0621-\u064A])", tok)
            if m and normalize(tok) not in vocab and normalize(tok[m.end():]) in vocab:
                tok = tok[m.end():]
            core = normalize(tok)
            if core and (len(core) >= 3 or core in vocab):
                kept.append(tok)
            elif tok in _KEEP or (tok and all(c in _KEEP for c in tok)):
                kept.append(tok)
        if kept:
            lines.append(" ".join(kept))
    return "\n".join(lines)


# Measured: printed text and screenshots read at 58-84 mean confidence;
# ثلث calligraphy at 19-33, even when a few of its fragments look like words.
MIN_CONFIDENCE = 45


# Measured on 16 fonts: decorative scripts (Ruqaa, Kufi display, Nastaliq) leave
# 40-85% of the words uncertain; naskh and plain fonts 0-30%.
MAX_UNCERTAIN_SHARE = 0.4


def reading_is_reliable(text: str, confidence: float = 100.0, uncertain: set[str] | None = None) -> bool:
    """Tesseract's reading is trusted when Tesseract itself is confident, few
    words were uncertain, at least 3 real words were read, and real words are
    the majority. Decorative calligraphy fails this test."""
    from .normalize import normalize
    if confidence < MIN_CONFIDENCE:
        return False
    ws = normalize(text).split()
    if uncertain and ws and sum(w in uncertain for w in ws) / len(ws) >= MAX_UNCERTAIN_SHARE:
        return False
    real = junk = 0
    for line in text.splitlines():
        r, j = _line_counts(line)
        real += r
        junk += j
    return real >= 3 and real >= junk


# A line that ends like this closes a sentence; any other line was wrapped by
# the image layout and continues on the next one.
_LINE_END = re.compile(r"[.!؟?»)\]﴾}:؛]\s*$")


def _tidy_vision(text: str) -> str:
    """Remove formatting a model may add (code fences), and join lines that
    the image wrapped in the middle of a sentence: «(لا» / «يؤمن أحدكم…»."""
    text = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    # Website names and watermarks in Latin letters (ALBETAQA.SITE, @page) are
    # not part of the post.
    text = re.sub(r"(?<!\S)\S*[A-Za-z]\S*(?!\S)", "", text)
    lines = [" ".join(line.split()) for line in text.splitlines() if line.strip()]
    out = []
    for line in lines:
        if out and not _LINE_END.search(out[-1]):
            out[-1] += " " + line
        else:
            out.append(line)
    return "\n".join(out)


@dataclass
class Reading:
    text: str
    engine: str                                   # tesseract, claude, gemini or tesseract-low
    uncertain: set[str] = field(default_factory=set)   # normalized words read with low confidence


def _same_reading(local: str, vision: str) -> bool:
    """Tesseract's reading is kept only if it has (almost) every word the
    vision model read: a skipped line ("ولا تشرب الخمر فإنها مفتاح كل شر")
    or a missing word must not pass, only a letter or two of difference."""
    from rapidfuzz import fuzz
    a, b = local.split(), vision.split()
    if not b or abs(len(a) - len(b)) > max(1, len(b) // 20):
        return False
    return fuzz.ratio(local, vision) >= 90


def read_image(image_bytes: bytes, mime: str = "image/png", gemini_key: str | None = None,
               claude_key: str | None = None) -> Reading:
    """Read the text of an image.

    Without a vision model: Tesseract's reading, or "tesseract-low" when it
    is not reliable (decorative calligraphy…).

    With a vision model, both read the image. Tesseract copies letters as
    they are and never "corrects" a tampered ayah, so its reading is kept
    when it is reliable AND says the same as the vision model. Otherwise
    (calligraphy, diacritised or decorated text, lines Tesseract skipped)
    the vision model's reading is used."""
    from rapidfuzz import fuzz

    from .normalize import normalize

    engines = available_engines(gemini_key, claude_key)
    errors, local = [], None
    if "tesseract" in engines:
        try:
            text, conf, unsure = _tesseract_with_confidence(image_bytes)
            ok = _arabic_words(text) >= 2 and reading_is_reliable(text, conf, unsure)
            local = Reading(text.strip(), "tesseract" if ok else "tesseract-low", unsure)
        except Exception as exc:
            errors.append(f"tesseract: {exc}")

    for engine in (e for e in engines if e in VISION_ENGINES):
        try:
            raw = (_claude(image_bytes, mime, claude_key) if engine == "claude"
                   else _gemini(image_bytes, mime, gemini_key))
            text = _tidy_vision(raw)
            if _arabic_words(text) < 1:
                continue
            if local and local.engine == "tesseract" and _same_reading(normalize(local.text), normalize(text)):
                return local
            return Reading(text, engine)
        except Exception as exc:  # fall through to the next engine
            errors.append(f"{engine}: {exc}")

    if local and local.text:
        return local
    raise RuntimeError("; ".join(errors) or "no OCR engine available")


def extract_text(image_bytes: bytes, mime: str = "image/png", api_key: str | None = None,
                 claude_key: str | None = None) -> tuple[str, str]:
    """(text, engine name); `api_key` is the Gemini key. See read_image."""
    r = read_image(image_bytes, mime, api_key, claude_key)
    return r.text, r.engine
