"""Manual check of the hadith finder (+ Dorar lookup).

Reads hadith_input.txt (one text per line), writes hadith_result.txt.
Open both in Notepad.

    python try_hadith.py
"""
from pathlib import Path

from core import dorar
from core.hadith import get_index

src, dst = Path("hadith_input.txt"), Path("hadith_result.txt")
if not src.exists():
    src.write_text("من غشنا فليس منا\nاطلبوا العلم ولو في الصين\n", encoding="utf-8")

labels = {"found": "✅ في الصحيحين", "similar": "⚠️ مشابه، يحتاج تحقق",
          "not_found": "❌ لم يُعثر عليه في الصحيحين", "too_short": "— النص قصير جداً"}

out = []
idx = get_index()
for line in src.read_text(encoding="utf-8-sig").splitlines():
    if not line.strip():
        continue
    r = idx.search(line)
    out.append(f"النص: {line}")
    out.append(f"النتيجة: {labels[r.status]}")
    for m in r.matches:
        out.append(f"  - {m.book} رقم {m.number} (تطابق {m.score}%)")
        out.append(f"    {m.text[:200]}")
    if r.status != "found":
        out.append("  الدرر السنية:")
        entries = dorar.lookup(line)
        if entries is None:
            out.append("    تعذر الوصول إلى الدرر السنية")
        elif not entries:
            out.append("    لا نتائج ← يُحال إلى مختص")
        for e in (entries or [])[:3]:
            out.append(f"    - {e.text[:150]}")
            out.append(f"      المحدث: {e.scholar} | المصدر: {e.source} {e.number} | الحكم: {e.grade}")
    out.append("")

dst.write_text("\n".join(out), encoding="utf-8-sig")
print(f"Done. Open {dst.resolve()}")
