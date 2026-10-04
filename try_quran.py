"""Manual check of the Quran checker.

Windows cmd garbles Arabic typed on the command line, so the text is read
from input.txt and the result is written to result.txt (open both in
Notepad). Each line of input.txt is checked separately.

    python try_quran.py
"""
from pathlib import Path

from core.quran import get_index

src, dst = Path("input.txt"), Path("result.txt")
if not src.exists():
    src.write_text("إن الله مع الصابرين والمتقين\n", encoding="utf-8")

labels = {"verified": "✅ صحيحة", "altered": "❌ فيها خطأ",
          "not_found": "— ليست آية", "too_short": "— النص قصير جداً"}
names = {"replace": "كلمة مبدّلة", "missing": "كلمة ناقصة", "extra": "كلمة زائدة"}

out = []
idx = get_index()
for line in src.read_text(encoding="utf-8-sig").splitlines():
    if not line.strip():
        continue
    r = idx.check(line)
    out.append(f"النص: {line}")
    out.append(f"النتيجة: {labels[r.status]}  (نسبة التطابق {r.match_ratio})")
    if r.reference:
        out.append(f"الموضع: {r.reference}")
        out.append(f"النص الصحيح: {r.correct_text}")
    for d in r.differences:
        if d.kind == "replace":
            out.append(f"  - {names[d.kind]}: كُتب «{d.written}» والصحيح «{d.correct}»")
        elif d.kind == "missing":
            out.append(f"  - {names[d.kind]}: «{d.correct}»")
        else:
            out.append(f"  - {names[d.kind]}: «{d.written}»")
    out.append("")

dst.write_text("\n".join(out), encoding="utf-8-sig")
print(f"Done. Open {dst.resolve()}")
