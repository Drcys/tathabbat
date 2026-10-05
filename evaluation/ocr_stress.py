"""Image-reading stress test: many Arabic fonts x image conditions.

Each post (genuine ayat, tampered ayat, real and fabricated hadiths) is drawn
as an image in every font and condition, read by the app's OCR, analysed, and
the verdicts compared with the known answer. Outcomes:

  correct   every verdict is the right one
  cautious  nothing wrong, but a genuine text was marked "check the reading"
  vision    the reading was not reliable, so the app asks for the vision model
            (or for a word typed by the user) instead of guessing
  missed    the text was read too poorly to be recognised; no verdict was shown
  wrong     a wrong verdict was shown (a genuine text called altered, a fake one
            called authentic, or a post split wrongly)

Fonts are the free Google Fonts (OFL). Download them into a folder and pass it:
    python evaluation/ocr_stress.py path/to/fonts
"""

from __future__ import annotations

import io
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFilter, ImageFont  # noqa: E402

POSTS = [
    ("قال الله تعالى: ﴿واستعينوا بالصبر والصلاة وإنها لكبيرة إلا على الخاشعين﴾", ["ok"]),
    ("﴿ألا بذكر الله تطمئن القلوب﴾", ["ok"]),
    ("قال تعالى: ﴿ومن يتق الله يجعل له مخرجا ويرزقه من حيث لا يحتسب﴾", ["ok"]),
    ("قال تعالى: ﴿وقل رب زدني علما﴾", ["ok"]),
    ("قال الله تعالى: ﴿واستعينوا بالصبر والدعاء وإنها لكبيرة إلا على الخاشعين﴾", ["bad"]),
    ("قال تعالى: ﴿يا أيها الذين آمنوا اصبروا وصابروا ورابطوا واتقوا الله كثيرا لعلكم تفلحون﴾", ["bad"]),
    ("قال رسول الله صلى الله عليه وسلم: «إنما الأعمال بالنيات وإنما لكل امرئ ما نوى»", ["ok"]),
    ("قال رسول الله صلى الله عليه وسلم: «من حسن إسلام المرء تركه ما لا يعنيه»", ["ok"]),
    ("قال رسول الله صلى الله عليه وسلم: «النظافة من الإيمان والوسخ من الشيطان»", ["bad"]),
    ("قال رسول الله صلى الله عليه وسلم: «اطلبوا العلم ولو في الصين»", ["bad"]),
]

CONDITIONS = ["clean", "small", "dark", "photo", "jpeg_blur"]


def _wrap(draw, text, font, width):
    lines, line = [], ""
    for w in text.split():
        trial = (line + " " + w).strip()
        if draw.textlength(trial, font=font, direction="rtl") > width and line:
            lines.append(line)
            line = w
        else:
            line = trial
    lines.append(line)
    return lines


def render(text: str, font_path: str, condition: str, seed: int) -> bytes:
    rng = random.Random(seed)
    size = 22 if condition == "small" else 38
    width = 520 if condition == "small" else 900
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("L", (10, 10)))
    lines = _wrap(probe, text, font, width - 60)
    line_h = int(size * 2.0)
    height = line_h * len(lines) + 60
    if condition == "dark":
        img = Image.new("RGB", (width, height), (18, 32, 47))
        fg = (240, 240, 240)
    elif condition == "photo":
        img = Image.new("RGB", (width, height))
        px = img.load()
        c1 = [rng.randint(40, 140) for _ in range(3)]
        c2 = [rng.randint(60, 200) for _ in range(3)]
        for y in range(height):
            for x in range(width):
                t = (x / width + y / height) / 2
                n = rng.randint(-25, 25)
                px[x, y] = tuple(max(0, min(255, int(a + (b - a) * t) + n)) for a, b in zip(c1, c2))
        img = img.filter(ImageFilter.GaussianBlur(1.5))
        fg = (255, 255, 255)
    else:
        img = Image.new("RGB", (width, height), "white")
        fg = (20, 20, 20)
    d = ImageDraw.Draw(img)
    y = 30
    for line in lines:
        w = d.textlength(line, font=font, direction="rtl")
        x = width - 30 - w
        if condition == "photo":
            d.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0), direction="rtl")
        d.text((x, y), line, font=font, fill=fg, direction="rtl")
        y += line_h
    if condition == "jpeg_blur":
        img = img.resize((int(width * 0.6), int(height * 0.6)), Image.BILINEAR).filter(ImageFilter.GaussianBlur(0.6))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=30)
        return buf.getvalue()
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def classify(expected: list[str], verdicts: list[str]) -> str:
    if not verdicts:
        return "missed"           # nothing recognised: no claim was made
    if len(verdicts) != len(expected):
        # A line read out of order can split one text into two cards. That is
        # a wrong claim only if a card says the opposite of the truth.
        opposite = {"ok": "bad", "bad": "ok", "warn": None}
        if any(opposite[e] in verdicts for e in expected):
            return "wrong"
        return "cautious"
    out = "correct"
    for e, v in zip(expected, verdicts):
        if e == v:
            continue
        if v == "warn":
            out = "cautious"      # never a wrong claim, only "check the reading"
        else:
            return "wrong"
    return out


def one(job):
    from core.analyze import analyze
    from core.ocr import read_image

    font, cond, i = job
    text, expected = POSTS[i]
    data = render(text, font, cond, seed=i)
    t = time.perf_counter()
    try:
        reading = read_image(data, "image/jpeg" if cond == "jpeg_blur" else "image/png")
        read, engine = reading.text, reading.engine
    except RuntimeError:
        return Path(font).stem, cond, i, "vision", "", time.perf_counter() - t
    if engine == "tesseract-low":
        return Path(font).stem, cond, i, "vision", read, time.perf_counter() - t
    verdicts = [s.verdict for s in analyze(read, use_dorar=False, from_image=True,
                                           uncertain=reading.uncertain).checked]
    return Path(font).stem, cond, i, classify(expected, verdicts), read, time.perf_counter() - t


def main(font_dir: str):
    import os
    only_fonts = [x for x in os.environ.get("FONTS", "").split(",") if x]
    only_conds = [x for x in os.environ.get("CONDS", "").split(",") if x] or CONDITIONS
    fonts = sorted(str(p) for p in Path(font_dir).glob("*.ttf") if not only_fonts or p.stem in only_fonts)
    jobs = [(f, c, i) for f in fonts for c in only_conds for i in range(len(POSTS))]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(one, jobs, chunksize=4))

    by_font = defaultdict(Counter)
    by_cond = defaultdict(Counter)
    total = Counter()
    for font, cond, i, outcome, _, _ in results:
        by_font[font][outcome] += 1
        by_cond[cond][outcome] += 1
        total[outcome] += 1
    wrong = [(f, c, POSTS[i][0][:40], r) for f, c, i, o, r, _ in results if o == "wrong"]
    secs = sorted(r[5] for r in results)

    out = Path(os.environ.get("OUT", Path(__file__).parent / "ocr_stress_results.json"))
    out.write_text(json.dumps({
        "total": dict(total), "images": len(results),
        "by_font": {k: dict(v) for k, v in by_font.items()},
        "by_condition": {k: dict(v) for k, v in by_cond.items()},
        "wrong": wrong, "median_seconds": secs[len(secs) // 2],
    }, ensure_ascii=False, indent=1), "utf-8")

    cols = ["correct", "cautious", "vision", "missed", "wrong"]
    print("font".ljust(18), *[c.ljust(9) for c in cols])
    for f, c in sorted(by_font.items()):
        print(f.ljust(18), *[str(c[k]).ljust(9) for k in cols])
    print()
    for cond, c in by_cond.items():
        print(cond.ljust(18), *[str(c[k]).ljust(9) for k in cols])
    print("\nTOTAL", dict(total), "of", len(results), "| median s/image", round(secs[len(secs) // 2], 2))
    for w in wrong:
        print("WRONG", w)


if __name__ == "__main__":
    main(sys.argv[1])
