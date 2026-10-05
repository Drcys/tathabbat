"""600 realistic posts at three levels of difficulty (fixed seed, reproducible).

Each post is built like a real WhatsApp/X post: greetings, emojis, several
items, different ways of introducing an ayah or a hadith, brackets or none,
the Uthmani script, source lines ("رواه البخاري"), salutations in brackets,
and traps: tampered ayat, weak and fabricated hadiths, sayings presented as
Quran, hadiths presented as Quran, ayat presented as the Prophet's ﷺ words.

The expected verdict of every item is known from how it was built. The test
compares the cards the tool shows with the expected ones, in order.

  easy    200 posts, 1 item, the usual wording
  medium  200 posts, 2-3 items, varied wording and the Uthmani script
  hard    200 posts, 4-6 items, with misattributions, fakes and noise

Run: python evaluation/post_stress.py      (Dorar is off: no network needed)
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import build_testset as bt  # noqa: E402
from core.analyze import analyze  # noqa: E402
from core.normalize import normalize  # noqa: E402

SEED = 4242

QURAN_INTROS_EASY = ["قال الله تعالى: ﴿{}﴾", "قال تعالى: ﴿{}﴾"]
QURAN_INTROS = QURAN_INTROS_EASY + ["﴿{}﴾", "يقول الله عز وجل: {}", "قال سبحانه: ({})",
                                    "وقال جل وعلا: «{}»", "قال الله تعالى في كتابه الكريم: ﴿{}﴾", "{}"]
HADITH_INTROS_EASY = ["قال رسول الله ﷺ: «{}»", "قال النبي صلى الله عليه وسلم: «{}»"]
HADITH_INTROS = HADITH_INTROS_EASY + [
    "قال ﷺ: «{}»", "عن النبي ﷺ أنه قال: {}", "قال رسول الله صلى الله عليه وسلم: \"{}\"",
    "عن أبي هريرة (رضي الله عنه) أن النبي (صلى الله عليه وسلم) قال: «{}»",
    "عن ابن عمر رضي الله عنهما قال: قال رسول الله ﷺ: ({})", "وقال عليه الصلاة والسلام: «{}»",
]
SOURCES = ["", "", " رواه البخاري", " (متفق عليه)", " رواه مسلم", "\nرواه الترمذي"]
OPENERS = ["", "صباح الخير 🌸", "جمعة مباركة 🤍", "تذكير جميل 🌿", "🌙 فوائد اليوم 🌙", "رسالة اليوم:"]
CLOSERS = ["", "انشرها تؤجر 🤍", "لا تجعلها تقف عندك", "اللهم صل وسلم على نبينا محمد",
           "Share this post 🙏", "اللهم اجعلنا من أهلها", "لا تنسوا سورة الكهف"]
NOISE = ["الله يجزاكم خير على النشر", "جزاكم الله خيرا وبارك فيكم", "تذكروا الصلاة على النبي ﷺ",
         "اللهم اغفر لنا ولوالدينا", "شاركوها مع أحبابكم ❤️", "اللهم إنا نسألك الجنة"]
FAKE_QURAN = ["العلم نور والجهل ظلام", "الصبر مفتاح الفرج", "القناعة كنز لا يفنى", "الدين المعاملة والأخلاق",
              "من جد وجد ومن زرع حصد", "العقل السليم في الجسم السليم", "اتق شر من أحسنت إليه",
              "الوقت كالسيف إن لم تقطعه قطعك"]
FABRICATED = bt.NOT_IN_SIX_BOOKS + ["صوموا تصحوا", "من علمني حرفا صرت له عبدا", "الأقربون أولى بالمعروف",
                                    "خير الأمور أوسطها", "الجنة تحت أقدام الأمهات"]


def plain(text):
    return bt.strip_marks(text)


class Pools:
    def __init__(self, rng):
        random.seed(rng.randint(0, 10 ** 9))  # build_testset uses the global random
        self.sahih = bt.distinctive_phrase(lambda d: d["sahihain"], 220)
        self.weak = bt.distinctive_phrase(lambda d: d["grader"] == "الألباني" and d["rating"] == "bad", 120)
        self.rng = rng
        print(f"pools: {len(self.sahih)} sahih phrases, {len(self.weak)} weak phrases")

    def ayah(self, uthmani_ok):
        while True:
            k, toks, uth = bt.ayah_sample((5, 12))
            if uth and not uthmani_ok:
                continue
            return k, toks


def item(kind, pools, rng, level):
    """Return (text line, expected (kind, verdict), label)."""
    q_intros = QURAN_INTROS_EASY if level == "easy" else QURAN_INTROS
    h_intros = HADITH_INTROS_EASY if level == "easy" else HADITH_INTROS
    src = "" if level == "easy" else rng.choice(SOURCES)
    if kind == "ayah_ok":
        _, toks = pools.ayah(level != "easy")
        return rng.choice(q_intros).format(" ".join(toks)), ("quran", "ok")
    if kind == "ayah_tampered":
        _, toks = pools.ayah(level != "easy")
        bad, _ = bt.alter(toks)
        intro = rng.choice([i for i in q_intros if i != "{}"])   # an altered ayah is presented as Quran
        return intro.format(" ".join(bad)), ("quran", "bad")
    if kind == "hadith_sahih":
        phrase, _ = rng.choice(pools.sahih)
        return rng.choice(h_intros).format(phrase) + src, ("hadith", "ok")
    if kind == "hadith_weak":
        phrase, _ = rng.choice(pools.weak)
        return rng.choice(h_intros).format(phrase) + src, ("hadith", "bad")
    if kind == "fabricated":
        return rng.choice(h_intros).format(rng.choice(FABRICATED)) + src, ("hadith", "bad")
    if kind == "fake_quran":
        intro = rng.choice(["قال الله تعالى: «{}»", "قال تعالى: ﴿{}﴾", "يقول الله عز وجل: {}"])
        return intro.format(rng.choice(FAKE_QURAN)), ("quran", "bad")
    if kind == "hadith_as_quran":
        phrase, _ = rng.choice(pools.sahih)
        intro = rng.choice(["قال الله تعالى: ﴿{}﴾", "قال تعالى: ﴿{}﴾"])
        return intro.format(phrase), ("hadith", "bad")
    if kind == "quran_as_hadith":
        _, toks = pools.ayah(False)
        intro = rng.choice(["قال رسول الله ﷺ: «{}»", "قال النبي صلى الله عليه وسلم: «{}»"])
        return intro.format(" ".join(toks)), ("quran", "warn")
    raise ValueError(kind)


KINDS = {
    "easy": ["ayah_ok", "ayah_tampered", "hadith_sahih", "hadith_weak", "fabricated"],
    "medium": ["ayah_ok", "ayah_ok", "ayah_tampered", "hadith_sahih", "hadith_sahih", "hadith_weak",
               "fabricated", "fake_quran"],
    "hard": ["ayah_ok", "ayah_tampered", "hadith_sahih", "hadith_weak", "fabricated", "fake_quran",
             "hadith_as_quran", "quran_as_hadith"],
}
SIZE = {"easy": (1, 1), "medium": (2, 3), "hard": (4, 6)}


def make_post(level, pools, rng):
    lines, expected, kinds = [], [], []
    if level != "easy" or rng.random() < 0.5:
        if (o := rng.choice(OPENERS)):
            lines.append(o)
    for _ in range(rng.randint(*SIZE[level])):
        kind = rng.choice(KINDS[level])
        text, exp = item(kind, pools, rng, level)
        lines.append(text)
        expected.append(exp)
        kinds.append(kind)
        if level == "hard" and rng.random() < 0.3:
            lines.append(rng.choice(NOISE))
    if (c := rng.choice(CLOSERS)):
        lines.append(c)
    return "\n".join(lines), expected, kinds


def compare(expected, got, kinds):
    """Per-item outcome, aligning the expected cards with the cards shown."""
    out = []
    sm = SequenceMatcher(None, [e[0] for e in expected], [g[0] for g in got], autojunk=False)
    matched = {}
    for blk in sm.get_matching_blocks():
        for k in range(blk.size):
            matched[blk.a + k] = blk.b + k
    for i, (exp, kind) in enumerate(zip(expected, kinds)):
        if i not in matched:
            out.append((kind, "missed"))
            continue
        g = got[matched[i]]
        if g == exp:
            out.append((kind, "correct"))
        elif exp[1] in ("bad", "warn") and g[1] == "ok":
            out.append((kind, "error_called_correct"))      # the worst failure
        elif exp[1] == "ok" and g[1] == "bad":
            out.append((kind, "false_alarm"))
        else:
            out.append((kind, "softer"))                     # e.g. bad shown as "check"
    extra = len(got) - len(matched)
    return out, extra


def main():
    rng = random.Random(SEED)
    pools = Pools(rng)
    by_level = defaultdict(Counter)
    by_kind = defaultdict(Counter)
    posts_ok = Counter()
    extras = Counter()
    failures = []
    for level in ("easy", "medium", "hard"):
        for _ in range(200):
            post, expected, kinds = make_post(level, pools, rng)
            segs = analyze(post, use_dorar=False).checked
            got = [(s.kind, s.verdict) for s in segs]
            outcome, extra = compare(expected, got, kinds)
            extras[level] += extra
            if all(o == "correct" for _, o in outcome) and not extra:
                posts_ok[level] += 1
            for kind, o in outcome:
                by_level[level][o] += 1
                by_kind[kind][o] += 1
                if o != "correct" and len(failures) < 40:
                    failures.append((level, kind, o, post.replace("\n", " ⏎ ")[:160], got))
    total_items = sum(sum(c.values()) for c in by_level.values())
    result = {
        "posts": 600, "items": total_items,
        "posts_fully_correct": dict(posts_ok),
        "by_level": {k: dict(v) for k, v in by_level.items()},
        "by_kind": {k: dict(v) for k, v in by_kind.items()},
        "extra_cards": dict(extras),
        "failures_sample": failures,
    }
    (ROOT / "evaluation" / "post_stress_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), "utf-8")

    cols = ["correct", "softer", "false_alarm", "error_called_correct", "missed"]
    print("level".ljust(8), "posts✓".ljust(8), *[c[:12].ljust(13) for c in cols], "extra")
    for level in ("easy", "medium", "hard"):
        c = by_level[level]
        print(level.ljust(8), f"{posts_ok[level]}/200".ljust(8), *[str(c[k]).ljust(13) for k in cols], extras[level])
    print()
    for kind, c in by_kind.items():
        n = sum(c.values())
        print(kind.ljust(18), f"{c['correct']}/{n}".ljust(9), {k: v for k, v in c.items() if k != "correct"})
    print("\nitems:", total_items)
    for f in failures[:25]:
        print("FAIL", f)


if __name__ == "__main__":
    main()
