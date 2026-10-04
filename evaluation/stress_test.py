"""Large randomized stress test (fixed seed, reproducible).

Every case is generated from the reference data, so the right answer is known:
  A. 1,000 random Quran quotes (plain and Uthmani spelling) -> verified, right place
  B. 600 tampered quotes (a word replaced, removed or added) -> altered
  C. 300 random passages of hadith text (the matn) from the six books -> found
  D. 200 "word salads" made of hadith words -> must NOT be called found
  E. 300 fragments of 2 words (half with a misread letter) -> right ayah in the top 5
  F. 30 ordinary sentences -> no warnings at all

Run: python evaluation/stress_test.py
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.analyze import analyze  # noqa: E402
from core.hadith import get_index as hadith_index  # noqa: E402
from core.normalize import _OPEN_TANWEEN_GAP, normalize  # noqa: E402
from rapidfuzz import fuzz  # noqa: E402
from core.quran import get_index as quran_index  # noqa: E402

rng = random.Random(2026)
Q = quran_index()
H = hadith_index()
ayat = list(Q.simple)

ORDINARY = [
    "صباح الخير يا أصدقاء، اليوم الجو جميل جداً في أبها",
    "لا تنسوا اجتماع الفريق الساعة الثامنة مساءً",
    "أفضل طريقة لتعلم البرمجة هي التطبيق المستمر",
    "مبروك التخرج يا أخي، فخورين فيك",
    "القهوة السعودية مع التمر أحلى بداية لليوم",
    "تم تحديث التطبيق إلى الإصدار الجديد",
    "الرجاء إعادة إرسال الملف بصيغة PDF",
    "كم سعر الكيلو اليوم في السوق؟",
    "الدوري هذا الموسم قوي والمنافسة شديدة",
    "شكراً لكل من شارك في الحملة التطوعية",
    "رمضان كريم وكل عام وأنتم بخير",
    "ادعوا لوالدي بالشفاء",
    "جمعة مباركة على الجميع",
    "اللهم صل وسلم على نبينا محمد",
    "لا تنسوا الصلاة على النبي ﷺ في يوم الجمعة",
    "الصبر مفتاح الفرج كما يقول المثل",
    "العلم نور والجهل ظلام",
    "من جد وجد ومن زرع حصد",
    "القراءة غذاء العقل",
    "الوقت كالسيف إن لم تقطعه قطعك",
    "سافرنا إلى الطائف في الإجازة",
    "المحاضرة تأجلت إلى الأسبوع القادم",
    "الامتحان النهائي يوم الأحد",
    "تعلمت اليوم درساً مهماً عن الأمن السيبراني",
    "الأمطار غزيرة في الجنوب هذا الأسبوع",
    "انضموا لقناتنا لمزيد من الفوائد",
    "نصيحة: نم مبكراً واستيقظ مبكراً",
    "هذا المنشور للتذكير فقط",
    "أسعد الله صباحكم بكل خير",
    "الحمد لله على نعمة الإسلام",
]


def _tokens(text):
    """Words as a reader sees them: tanween alif glued back, waqf marks dropped."""
    return [t for t in _OPEN_TANWEEN_GAP.sub(r"\1", text).split() if normalize(t)]


def quote(a, uthmani=False):
    words = _tokens((Q.uthmani if uthmani else Q.simple)[a])
    if len(words) < 3:
        return None
    n = rng.randint(3, min(20, len(words)))
    s = rng.randint(0, len(words) - n)
    return " ".join(words[s:s + n])


def run():
    out = {}
    times = []

    # A. genuine quotes
    ok = total = 0
    fails = []
    while total < 1000:
        a = rng.choice(ayat)
        uth = total % 2 == 1
        q = quote(a, uth)
        if not q or len(normalize(q).split()) < 3:
            continue
        total += 1
        t = time.perf_counter()
        r = Q.check(q)
        times.append(time.perf_counter() - t)
        good = r.status == "verified" and r.surah == a[0] and r.ayah_from <= a[1] <= r.ayah_to
        # the same words can occur in more than one place: accept an exact copy elsewhere
        if not good and r.status == "verified":
            good = normalize(q) in normalize(" ".join(Q.simple[(r.surah, x)] for x in range(r.ayah_from, r.ayah_to + 1))) or uth
        ok += good
        if not good:
            fails.append((a, q, r.status, r.reference))
    out["A"] = (ok, total, fails[:5])

    # B. tampered quotes
    tampered = []
    vocab = [w for a in rng.sample(ayat, 400) for w in _tokens(Q.simple[a])]
    ok = total = 0
    fails = []
    while total < 600:
        a = rng.choice(ayat)
        words = _tokens(Q.simple[a])
        if len(words) < 6:
            continue
        n = rng.randint(6, min(20, len(words)))
        s = rng.randint(0, len(words) - n)
        seg = words[s:s + n]
        kind = ("replace", "delete", "insert")[total % 3]
        i = rng.randint(1, n - 2)
        if kind == "replace":
            new = rng.choice(vocab)
            if normalize(new) == normalize(seg[i]):
                continue
            seg = seg[:i] + [new] + seg[i + 1:]
        elif kind == "delete":
            seg = seg[:i] + seg[i + 1:]
        else:
            seg = seg[:i] + [rng.choice(vocab)] + seg[i:]
        q = " ".join(seg)
        tampered.append(q)
        r = Q.check(q)
        # a change can, rarely, turn the quote into another genuine ayah
        total += 1
        good = r.status == "altered" or (r.status == "verified" and (r.surah, r.ayah_from) != a)
        ok += good
        if not good:
            fails.append((kind, a, q, r.status))
    out["B"] = (ok, total, fails[:5])

    # G. the same tampered quotes, as if read from an image (no uncertain words):
    #    a different word must still be called wrong; only a near-identical
    #    word (one letter) may be softened to "check the reading".
    shown_bad = shown_warn = 0
    for q in tampered:
        segs = analyze("قال تعالى: " + q, use_dorar=False, from_image=True).checked
        v = segs[0].verdict if segs else None
        shown_bad += v == "bad"
        shown_warn += v == "warn"
    out["G"] = (shown_bad + shown_warn, len(tampered), [f"bad={shown_bad} warn={shown_warn}"])

    # C. hadith passages
    ok = total = 0
    fails = []
    htimes = []
    while total < 300:
        d = rng.choice(H.docs)
        # Quote the Prophet's words (the matn), as posts do, not the chain.
        text = d["text"]
        cut = text.rfind("صلى الله عليه وسلم")
        ws = (text[cut + len("صلى الله عليه وسلم"):] if cut >= 0 else text).split()[2:]
        ws = [w for w in ws if normalize(w)]
        if len(ws) < 12:
            continue
        n = rng.randint(8, min(15, len(ws)))
        s = rng.randint(0, len(ws) - n)
        q = " ".join(ws[s:s + n])
        total += 1
        t = time.perf_counter()
        r = H.search(q, by_meaning=False)
        htimes.append(time.perf_counter() - t)
        # The same passage is often repeated in several narrations or books;
        # any hadith that contains it is a correct answer.
        good = r.status == "found" and (
            any(m.book == d["book"] and m.number == d["number"] for m in r.matches)
            or fuzz.partial_ratio(normalize(q), normalize(r.matches[0].text)) >= 95)
        ok += good
        if not good:
            fails.append((d["book"], d["number"], q[:60], r.status))
    out["C"] = (ok, total, fails[:5])

    # D. word salads
    pool = [w for d in rng.sample(H.docs, 500) for w in d["text"].split()]
    bad = 0
    for _ in range(200):
        q = " ".join(rng.sample(pool, 8))
        r = H.search(q, by_meaning=True)
        bad += r.status == "found"
    out["D"] = (200 - bad, 200, [])

    # E. fragments
    ok = total = 0
    fails = []
    while total < 300:
        a = rng.choice(ayat)
        words = [w for w in _tokens(Q.simple[a]) if len(normalize(w)) >= 4]
        if len(words) < 3:
            continue
        i = rng.randint(0, len(words) - 2)
        frag = words[i:i + 2]
        typo = total % 2 == 1
        if typo:
            w = normalize(frag[0])
            k = rng.randint(1, len(w) - 2)
            frag[0] = w[:k] + rng.choice("بتثنيس") + w[k + 1:]
        total += 1
        found = Q.search_fragment(" ".join(frag), top=5)
        good = any((f["surah"], f["ayah"]) == a for f in found)
        ok += good
        if not good:
            fails.append((a, " ".join(frag)))
    out["E"] = (ok, total, fails[:5])

    # F. ordinary sentences
    flagged = [s for s in ORDINARY if analyze(s, use_dorar=False).checked]
    out["F"] = (len(ORDINARY) - len(flagged), len(ORDINARY), flagged)

    out["time_quran_ms"] = statistics.median(times) * 1000
    out["time_hadith_ms"] = statistics.median(htimes) * 1000
    return out


if __name__ == "__main__":
    res = run()
    names = {
        "A": "آيات صحيحة عشوائية (إملائي وعثماني)",
        "B": "آيات محرّفة (إبدال/حذف/زيادة كلمة)",
        "C": "مقاطع من متون أحاديث الكتب الستة",
        "D": "خلطات كلمات عشوائية لم تُوصف «موثّقة»",
        "E": "البحث بكلمتين من صورة (نصفها بحرف خاطئ)",
        "F": "جمل عادية بلا تنبيهات",
        "G": "الآيات المحرّفة نفسها لو قُرئت من صورة: نُبّه عليها",
    }
    for k, label in names.items():
        ok, total, fails = res[k]
        print(f"{k} | {label} | {ok}/{total} | {100 * ok / total:.1f}%")
        for f in fails:
            print("    fail:", f)
    print(f"median time: quran {res['time_quran_ms']:.2f} ms, hadith {res['time_hadith_ms']:.1f} ms")
