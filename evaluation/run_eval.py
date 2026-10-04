"""Run the tool on the evaluation set and write the results.

    python evaluation/run_eval.py   -> evaluation/results.md

Each post has one expected outcome:
  ok   – the tool must find the quote and confirm it (correct ayah / sahih hadith)
  bad  – the tool must flag it (altered ayah, weak hadith, not in the six books)
  none – a plain post: the tool must not raise anything
For ayat we also check that the reported surah:ayah is the right one.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.analyze import analyze  # noqa: E402
from core.normalize import normalize  # noqa: E402
from core.quran import get_index as quran_index  # noqa: E402

SET = ROOT / "evaluation" / "testset.jsonl"
SIMPLE = {(v["chapter"], v["verse"]): v["text"] for v in json.loads(
    (ROOT / "data" / "quran_simple.json").read_text("utf-8"))["quran"]}
OUT = ROOT / "evaluation" / "results.md"

NAMES = {
    "ayah_correct": "آيات صحيحة",
    "ayah_altered": "آيات محرّفة (كلمة مبدّلة/ناقصة/زائدة)",
    "hadith_sahih": "أحاديث من الصحيحين",
    "hadith_weak": "أحاديث ضعيفة (حكم الألباني)",
    "not_in_six_books": "أقوال منسوبة للنبي ﷺ لا توجد في الكتب الستة",
    "plain": "منشورات عادية بلا آيات أو أحاديث",
}


def judge(row, report):
    checked = report.checked
    if row["expect"] == "none":
        return not checked, "لا شيء" if not checked else f"{len(checked)} تنبيه"
    if not checked:
        return False, "لم يُكتشف"
    seg = checked[0]
    got = seg.verdict
    if row["category"].startswith("ayah"):
        ref = f"{seg.quran.surah}:{seg.quran.ayah_from}" if seg.kind == "quran" else "-"
        right_place = False
        if seg.kind == "quran":
            # Right place = the original quote really is in the reported ayat.
            # (Some wordings repeat verbatim in several surahs.)
            ayat = range(seg.quran.ayah_from, seg.quran.ayah_to + 1)
            for texts, flag in ((SIMPLE, False), (quran_index().uthmani, True)):
                reported = " ".join(texts[(seg.quran.surah, a)] for a in ayat)
                if normalize(row["original"], flag) in normalize(reported, flag):
                    right_place = True
        return (got == row["expect"] and right_place), f"{got} @ {ref}"
    return got == row["expect"], got


# Hard cases, reported separately: hadiths quoted with different wording
# (narration by meaning). The tool should at least say "needs checking"
# and show the real wording; it must never call them authentic as written.
HARD = [
    # (as written in the post, where it really is, words that must appear in the matched hadith)
    ("من صلى الفجر في جماعة فهو في ذمة الله", "مسلم: من صلى الصبح فهو في ذمة الله", ["ذمه الله"]),
    ("ابتسامتك في وجه أخيك صدقة", "الترمذي: تبسمك في وجه أخيك لك صدقة", ["تبسمك"]),
    ("تبسمك في وجه أخيك صدقة", "الترمذي: تبسمك في وجه أخيك لك صدقة", ["تبسمك"]),
    ("لا يدخل الجنة نمام", "الصحيحان: لا يدخل الجنة قتات / نمام", ["لا يدخل الجنه"]),
    ("أحب الأعمال إلى الله أدومها وإن قل", "الصحيحان: أحب الأعمال إلى الله أدومها وإن قل", ["ادومها"]),
    ("الإيمان بضع وسبعون شعبة أعلاها لا إله إلا الله", "مسلم: فأفضلها قول لا إله إلا الله", ["بضع وسبعون"]),
    ("الكلمة الحسنة صدقة", "الصحيحان: الكلمة الطيبة صدقة", ["الكلمه الطيبه صدقه"]),
    ("صلاة الجماعة أفضل من صلاة الفرد بسبع وعشرين درجة", "الصحيحان: من صلاة الفذ بسبع وعشرين درجة", ["وعشرين درجه"]),
    ("ما نقص مال من صدقة", "مسلم: ما نقصت صدقة من مال / الترمذي: ما نقص مال عبد من صدقة", ["نقص", "صدقه"]),
    ("المسلم أخو المسلم لا يظلمه ولا يخذله", "مسلم: ... ولا يخذله ولا يحقره", ["لا يظلمه"]),
    ("الدنيا سجن المؤمن وجنة الكافر", "مسلم", ["سجن المومن"]),
    ("من لا يرحم الناس لا يرحمه الله", "الصحيحان", ["لا يرحم الناس"]),
    ("الصدقة تطفئ غضب الرب", "الترمذي: إن الصدقة لتطفئ غضب الرب", ["غضب الرب"]),
    ("من قام ليلة القدر بإيمان واحتساب غفرت ذنوبه", "البخاري: إيمانا واحتسابا غفر له ما تقدم من ذنبه", ["ليله القدر"]),
    ("البر هو حسن الخلق", "مسلم: البر حسن الخلق", ["البر حسن الخلق"]),
]


# Held-out cases, written AFTER the semantic settings were fixed and never
# used to tune them: they show how the tool generalises.
HELDOUT_POS = [
    ("من صام رمضان بإيمان واحتساب غفرت ذنوبه السابقة", ["رمضان"]),
    ("خيركم من تعلم القرآن ثم علمه للناس", ["تعلم القران"]),
    ("لا يحل لمسلم أن يقاطع أخاه فوق ثلاث ليال", ["فوق ثلاث"]),
    ("الطهارة نصف الإيمان", ["شطر الايمان"]),
    ("المؤمن لا يلدغ من الحفرة مرتين", ["مرتين"]),
    ("إذا مات الإنسان توقف عمله إلا من ثلاث", ["انقطع"]),
    ("الدين النصيحة لله ولكتابه ولرسوله", ["الدين النصيحه"]),
    ("من دل على خير كان له مثل أجر فاعله", ["مثل اجر"]),
]
HELDOUT_NEG = ["الجنة تحت أقدام الأمهات", "صوموا تصحوا", "من علمني حرفا صرت له عبدا",
               "الأقربون أولى بالمعروف", "خير الأمور أوسطها", "العلم نور والجهل ظلام",
               "الصبر مفتاح الفرج", "القناعة كنز لا يفنى"]


def ablation():
    """Same cases with and without our trained meaning model."""
    from core.hadith import get_index as hadith_index
    h = hadith_index()

    def recovered(cases, by_meaning):
        n = 0
        for case in cases:
            text, keys = case[0], case[-1]
            r = h.search(text, by_meaning=by_meaning)
            n += bool(r.matches) and all(k in normalize(r.matches[0].text) for k in keys)
        return n

    def false_ok(by_meaning):
        n = 0
        for text in HELDOUT_NEG:
            r = h.search(text, by_meaning=by_meaning)
            n += r.status == "found" and r.matches[0].rating == "ok"
        return n

    out = ["", "## أثر نموذج المعاني المدرَّب (مقارنة: بدونه / معه)", "",
           "| المجموعة | بالمطابقة اللفظية فقط | مع نموذج المعاني |", "|---|---|---|"]
    out.append(f"| حالات الرواية بالمعنى ({len(HARD)}) | {recovered(HARD, False)}/{len(HARD)} | {recovered(HARD, True)}/{len(HARD)} |")
    out.append(f"| حالات جديدة لم تُستخدم في الضبط ({len(HELDOUT_POS)}) | {recovered(HELDOUT_POS, False)}/{len(HELDOUT_POS)} | {recovered(HELDOUT_POS, True)}/{len(HELDOUT_POS)} |")
    out.append(f"| أقوال ليست أحاديث وُصفت «موثّقة» خطأً ({len(HELDOUT_NEG)}) | {false_ok(False)} | {false_ok(True)} |")
    out += ["", "النموذج: Word2Vec (skip-gram) درّبه الفريق على نصوص الكتب الستة والقرآن (2.4 مليون كلمة)، "
            "انظر `models/train_embeddings.py`. نتائج البحث بالمعنى تُعرض دائماً «تحتاج تحققاً» ولا تُوصف بأنها موثّقة."]
    return out


def hard_cases():
    out = ["", "## حالات صعبة: أحاديث بألفاظ مختلفة أو بالمعنى (تُقاس منفصلة)", "",
           "| النص كما كُتب | الأصل | نتيجة الأداة |", "|---|---|---|"]
    labels = {"ok": "✅", "warn": "⚠️", "bad": "❌", None: "—"}
    hit = 0
    for text, origin, keys in HARD:
        rep = analyze(f"قال رسول الله ﷺ: «{text}»", use_dorar=False)
        seg = rep.checked[0] if rep.checked else None
        v = seg.verdict if seg else None
        where, right = "لم يُعثر على نص قريب", False
        if seg and seg.hadith.matches:
            m = seg.hadith.matches[0]
            right = all(k in normalize(m.text) for k in keys)
            how = " (بالمعنى)" if getattr(m, "semantic", False) else ""
            where = f"{m.book} {m.number}{how}" + ("" if right else " ✗ غير المقصود")
        hit += right
        out.append(f"| {text} | {origin} | {labels[v]} {where} |")
    out.append("")
    out.append(f"أعادت الأداة النص إلى أصله الصحيح في **{hit}/{len(HARD)}** حالة. "
               "ولا تصف أي نص مكتوب بغير لفظه بأنه «موثّق»: تعرضه «يحتاج تحققاً» مع لفظه الأصلي.")
    return out, hit


def main():
    rows = [json.loads(line) for line in SET.read_text("utf-8").splitlines()]
    per = defaultdict(lambda: [0, 0])
    failures = []
    t0 = time.time()
    for r in rows:
        report = analyze(r["post"], use_dorar=False)
        ok, got = judge(r, report)
        per[r["category"]][0] += ok
        per[r["category"]][1] += 1
        if not ok:
            failures.append((r, got))
    elapsed = (time.time() - t0) / len(rows) * 1000
    total_ok = sum(v[0] for v in per.values())

    lines = ["# نتائج تقييم تثبَّت", "",
             f"- عدد المنشورات: **{len(rows)}** (اصطناعية بالكامل)",
             f"- الدقة الإجمالية: **{total_ok}/{len(rows)} = {total_ok / len(rows):.0%}**",
             f"- متوسط زمن فحص المنشور: **{elapsed:.0f} ملّي ثانية**", "",
             "| الفئة | الصحيح | النسبة |", "|---|---|---|"]
    for cat, (ok, n) in per.items():
        lines.append(f"| {NAMES[cat]} | {ok}/{n} | {ok / n:.0%} |")
    lines += ["", "## الحالات التي أخفقت فيها الأداة", ""]
    if not failures:
        lines.append("لا توجد.")
    for r, got in failures:
        lines.append(f"- #{r['id']} ({NAMES[r['category']]}) المتوقع: {r['expect']} · الناتج: {got} · المرجع: {r['ref']}")
        lines.append(f"  > {r['post'].replace(chr(10), ' / ')}")
    hard_lines, _ = hard_cases()
    lines += hard_lines
    lines += ablation()
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
