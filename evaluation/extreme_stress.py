"""1,000 very hard tests: 600 noisy text posts + 400 post images (fixed seed).

Text posts (600): 6-9 items each, with the noise real posts have:
  typing variants of the intros ("قال تعالي"، "صلي الله عليه وسلم")، Persian
  letters (ی ک)، tatweel, stray spaces and repeated punctuation, Arabic-Indic
  ayah numbers, two items on one line, emojis inside quotes, partial
  diacritics, and every trap: tampered ayat, weak and fabricated hadiths,
  sayings presented as Quran, hadiths presented as Quran, ayat presented as
  the Prophet's ﷺ words.

Images (400): 2-4 items drawn as a post image in 16 fonts and 5 looks
(screenshot, dark mode, chat bubble, text over a photo, small and compressed),
read with the local OCR only (no vision model), then checked.

Each built item is paired with the card shown for it by its words, not by
its position, so one extra or missing card does not shift the others.

Outcomes per item:
  correct    the right card
  cautious   a softer card ("check the reading" / "needs checking"), never wrong
  vision     (images) the reading was not reliable: the app asks for the vision
             model or a typed word instead of judging
  missed     no card for this item
  WRONG      a wrong claim: an error shown as authentic, or a correct text
             called wrong

Run: python evaluation/extreme_stress.py <fonts folder>
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import post_stress as ps  # noqa: E402
from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

SEED = 777

INTRO_NOISE = [
    ("قال تعالى", "قال تعالي"), ("قال الله تعالى", "قال الله تعالي"), ("قال الله تعالى", "قال اللّه تعالى"),
    ("صلى الله عليه وسلم", "صلي الله عليه وسلم"), ("صلى الله عليه وسلم", "صلى الله عليه و سلم"),
    ("رسول الله", "رسول اللہ"), ("قال ﷺ", "قال ﷺ"), ("قال رسول الله ﷺ", "قال رسول الله صل الله عليه وسلم"),
]
EXTRA_KINDS = ["ayah_ok", "ayah_tampered", "hadith_sahih", "hadith_weak", "fabricated", "fake_quran",
               "hadith_as_quran", "quran_as_hadith", "ayah_ok", "hadith_sahih"]


def noisy(line: str, rng: random.Random) -> str:
    """Typing noise that does not change what the line says."""
    for a, b in INTRO_NOISE:
        if a in line and rng.random() < 0.35:
            line = line.replace(a, b, 1)
    if rng.random() < 0.25:
        line = line.replace("ي", "ی", 1)             # Persian ya from some keyboards
    if rng.random() < 0.15:
        line = line.replace("ك", "ک", 1)             # Persian kaf
    if rng.random() < 0.2:
        line = re.sub(r"(?<=[ب-ي])(?=[ب-ي])", "ـ", line, count=1)  # tatweel
    if rng.random() < 0.2:
        line = line.replace(":", " :  ", 1)
    if rng.random() < 0.15:
        line = line.replace("»", " 🤍»", 1)
    if rng.random() < 0.15:
        line = line + rng.choice([" !!", " ..", " ✅", " 👇"])
    return line


def extreme_post(pools, rng):
    """6-9 items with noise; sometimes two items share a line."""
    lines, expected, kinds, texts = [], [], [], []
    if (o := rng.choice(ps.OPENERS)):
        lines.append(o)
    n = rng.randint(6, 9)
    pending = None
    for _ in range(n):
        kind = rng.choice(EXTRA_KINDS)
        text, exp = ps.item(kind, pools, rng, "hard")
        text = noisy(text, rng)
        expected.append(exp)
        kinds.append(kind)
        texts.append(text)
        if pending is not None and "\n" not in text and "\n" not in pending and rng.random() < 0.25 \
                and _has_brackets(pending) and _has_brackets(text):
            lines[-1] = pending + " " + text          # two quoted items on one line
            pending = None
            continue
        lines.append(text)
        pending = text
        if rng.random() < 0.3:
            lines.append(rng.choice(ps.NOISE))
            pending = None
    if (c := rng.choice(ps.CLOSERS)):
        lines.append(c)
    return "\n".join(lines), expected, kinds, texts


def _has_brackets(t):
    return bool(re.search(r"[«﴿\"(].+[»﴾\")]", t))


# ---------------------------------------------------------------- images

LOOKS = ["screenshot", "dark", "bubble", "photo", "small_jpeg"]


def render_post(text: str, font_path: str, look: str, seed: int) -> tuple[bytes, str]:
    rng = random.Random(seed)
    size = 24 if look == "small_jpeg" else 34
    width = 560 if look == "small_jpeg" else 900
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    lines = []
    for para in text.split("\n"):
        lines += _wrap(probe, para, font, width - 90) or [""]
    line_h = int(size * 1.9)
    height = line_h * len(lines) + 80
    if look == "dark":
        img, fg = Image.new("RGB", (width, height), (17, 27, 33)), (233, 237, 239)
    elif look == "bubble":
        img, fg = Image.new("RGB", (width, height), (236, 229, 221)), (17, 17, 17)
        ImageDraw.Draw(img).rounded_rectangle((30, 20, width - 20, height - 20), 18, fill=(220, 248, 198))
    elif look == "photo":
        img = Image.new("RGB", (width, height))
        px = img.load()
        c1 = [rng.randint(30, 120) for _ in range(3)]
        c2 = [rng.randint(60, 180) for _ in range(3)]
        for y in range(height):
            for x in range(0, width):
                t = (x / width + y / height) / 2
                px[x, y] = tuple(max(0, min(255, int(a + (b - a) * t) + rng.randint(-20, 20))) for a, b in zip(c1, c2))
        img = img.filter(ImageFilter.GaussianBlur(1.2))
        fg = (255, 255, 255)
    else:
        img, fg = Image.new("RGB", (width, height), "white"), (20, 20, 20)
    d = ImageDraw.Draw(img)
    y = 40
    for line in lines:
        w = d.textlength(line, font=font, direction="rtl") if line else 0
        x = width - 45 - w
        if look == "photo":
            d.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0), direction="rtl")
        d.text((x, y), line, font=font, fill=fg, direction="rtl")
        y += line_h
    buf = io.BytesIO()
    if look == "small_jpeg":
        img = img.resize((int(width * 0.75), int(height * 0.75)), Image.BILINEAR)
        img.save(buf, "JPEG", quality=40)
        return buf.getvalue(), "image/jpeg"
    img.save(buf, "PNG")
    return buf.getvalue(), "image/png"


def _wrap(draw, text, font, width):
    out, line = [], ""
    for w in text.split():
        trial = (line + " " + w).strip()
        if draw.textlength(trial, font=font, direction="rtl") > width and line:
            out.append(line)
            line = w
        else:
            line = trial
    if line:
        out.append(line)
    return out


# ---------------------------------------------------------------- scoring

def align(texts, segs):
    """Pair each built item with the card shown for it, by its words (not by
    position): an extra or a missing card does not shift the others."""
    from core.normalize import normalize
    got, used = [None] * len(texts), set()
    for i, t in enumerate(texts):
        item_words = set(normalize(t).split())
        for j, s in enumerate(segs):
            w = normalize(s.text).split()
            if j not in used and w and sum(x in item_words for x in w) >= 0.6 * len(w):
                got[i] = (s.kind, s.verdict)
                used.add(j)
                break
    return got, len(segs) - len(used)


def score_aligned(expected, got, kinds):
    res = []
    for exp, g, kind in zip(expected, got, kinds):
        if g is None:
            o = "missed"
        elif g == exp:
            o = "correct"
        elif (exp[1] in ("bad", "warn") and g[1] == "ok") or (exp[1] == "ok" and g[1] == "bad"):
            o = "WRONG"
        else:
            o = "cautious"
        res.append((kind, o))
    return res


def score(expected, got, kinds, from_image):
    out, extra = ps.compare(expected, got, kinds)
    res = []
    for kind, o in out:
        if o == "softer":
            o = "cautious"
        elif o in ("error_called_correct", "false_alarm"):
            o = "WRONG"
        res.append((kind, o))
    return res, extra


def image_post(level, pools, rng):
    """A post image: an opener, 2-4 items, maybe a closer."""
    lines, expected, kinds, texts = [], [], [], []
    if (o := rng.choice(ps.OPENERS)):
        lines.append(o)
    for _ in range(rng.randint(2, 4)):
        kind = rng.choice(ps.KINDS[level])
        text, exp = ps.item(kind, pools, rng, level)
        lines.append(text)
        expected.append(exp)
        kinds.append(kind)
        texts.append(text)
    if (c := rng.choice(ps.CLOSERS)):
        lines.append(c)
    return "\n".join(lines), expected, kinds, texts


def run_image(job):
    from core.analyze import analyze
    from core.ocr import read_image
    post, expected, kinds, texts, font, look, seed = job
    data, mime = render_post(post, font, look, seed)
    try:
        r = read_image(data, mime)
    except RuntimeError:
        return [(k, "vision") for k in kinds], 0, Path(font).stem, look, []
    if r.engine == "tesseract-low":
        return [(k, "vision") for k in kinds], 0, Path(font).stem, look, []
    segs = analyze(r.text, use_dorar=False, from_image=True, uncertain=r.uncertain).checked
    got, extra = align(texts, segs)
    res = score_aligned(expected, got, kinds)
    fails = [(Path(font).stem, look, kind, o, t, g, r.text) for (kind, o), t, g in zip(res, texts, got) if o == "WRONG"]
    return res, extra, Path(font).stem, look, fails


def main(font_dir: str):
    from core.analyze import analyze
    rng = random.Random(SEED)
    pools = ps.Pools(rng)

    # ---- 600 extreme text posts
    text_by_kind = defaultdict(Counter)
    text_total = Counter()
    text_posts_ok = 0
    text_fail = []
    for _ in range(600):
        post, expected, kinds, texts = extreme_post(pools, rng)
        segs = analyze(post, use_dorar=False).checked
        got, extra = align(texts, segs)
        res = score_aligned(expected, got, kinds)
        text_posts_ok += all(o == "correct" for _, o in res) and not extra
        for i, (kind, o) in enumerate(res):
            text_by_kind[kind][o] += 1
            text_total[o] += 1
            if o in ("WRONG", "missed") and len(text_fail) < 60:
                text_fail.append((kind, o, texts[i], got[i]))
        text_total["extra_cards"] += extra

    if os.environ.get("TEXT_ONLY"):
        print("TEXT 600 posts | fully correct:", text_posts_ok, "|", dict(text_total))
        for k, v in text_by_kind.items():
            n = sum(v.values())
            print("  ", k.ljust(16), f"{v['correct']}/{n}", {a: b for a, b in v.items() if a != "correct"})
        for f in text_fail[:60]:
            print("TEXTFAIL", f)
        return

    # ---- 400 images
    fonts = sorted(str(p) for p in Path(font_dir).glob("*.ttf"))
    jobs = []
    for i in range(400):
        level = "medium" if i % 2 else "hard"
        jobs.append((*image_post(level, pools, rng), fonts[i % len(fonts)], LOOKS[i % len(LOOKS)], i))
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(run_image, jobs, chunksize=4))
    img_total = Counter()
    img_by_font = defaultdict(Counter)
    img_by_look = defaultdict(Counter)
    img_fail = []
    for res, extra, font, look, fails in results:
        img_fail += fails
        for _, o in res:
            img_total[o] += 1
            img_by_font[font][o] += 1
            img_by_look[look][o] += 1
        img_total["extra_cards"] += extra

    out = {
        "text": {"posts": 600, "posts_fully_correct": text_posts_ok, "items": dict(text_total),
                 "by_kind": {k: dict(v) for k, v in text_by_kind.items()}, "failures": text_fail},
        "images": {"images": 400, "items": dict(img_total), "by_font": {k: dict(v) for k, v in img_by_font.items()},
                   "by_look": {k: dict(v) for k, v in img_by_look.items()}, "wrong": img_fail},
    }
    (ROOT / "evaluation" / "extreme_stress_results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), "utf-8")

    print("TEXT 600 posts | fully correct:", text_posts_ok, "|", dict(text_total))
    for k, v in text_by_kind.items():
        n = sum(v.values())
        print("  ", k.ljust(16), f"{v['correct']}/{n}", {a: b for a, b in v.items() if a != "correct"})
    print("IMAGES 400 |", dict(img_total))
    for k, v in img_by_look.items():
        print("  ", k.ljust(12), dict(v))
    for k, v in sorted(img_by_font.items()):
        print("  ", k.ljust(16), dict(v))
    for f in text_fail[:15]:
        print("TEXTFAIL", f)
    for f in img_fail:
        print("IMGWRONG", f[:6], "| OCR:", f[6].replace("\n", " ⏎ ")[:300])


if __name__ == "__main__":
    main(sys.argv[1])
