"""Build the synthetic evaluation set (100 posts) with known answers.

All posts are synthetic: no real user messages are used (challenge rule).
Ground truth comes from the reference data itself, so it is not a matter of
opinion:

* correct ayat   – copied from the Mushaf text (plain and Uthmani spelling)
* altered ayat   – the same, with one word replaced, removed or added
* sahih hadith   – a distinctive phrase from Sahih al-Bukhari / Muslim
* weak hadith    – a distinctive phrase from the Sunan graded weak by al-Albani
* not in the six books – well-known sayings wrongly attributed to the Prophet ﷺ
  (scholars state they have no basis; to be reviewed with a Sharia mentor)
* plain posts    – ordinary messages with no Quran or hadith

    python evaluation/build_testset.py      -> evaluation/testset.jsonl
"""

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.hadith import get_index as hadith_index  # noqa: E402
from core.normalize import normalize  # noqa: E402
from core.quran import get_index as quran_index  # noqa: E402

random.seed(2026)
OUT = ROOT / "evaluation" / "testset.jsonl"

AYAH_INTROS = ["قال الله تعالى: ﴿{}﴾", "قال تعالى: {}", "﴿{}﴾", "{}", "يقول الله عز وجل: {}"]
HADITH_INTROS = ["قال رسول الله ﷺ: «{}»", "قال النبي صلى الله عليه وسلم: \"{}\"",
                 "عن النبي ﷺ أنه قال: {}", "قال ﷺ: «{}»"]
FILLERS = ["", "صباح الخير 🌸\n", "جمعة مباركة\n", "انشرها ولك الأجر\n", "تذكير جميل 🤍\n"]
ENDINGS = ["", "\nانشرها تؤجر", "\nلا تجعلها تقف عندك", "\nاللهم اجعلنا من أهلها"]

NOT_IN_SIX_BOOKS = [
    "اطلبوا العلم ولو في الصين",
    "حب الوطن من الإيمان",
    "اختلاف أمتي رحمة",
    "النظافة من الإيمان",
    "تفاءلوا بالخير تجدوه",
    "أدبني ربي فأحسن تأديبي",
    "الدين المعاملة",
    "من عرف نفسه فقد عرف ربه",
    "خير الأسماء ما حمد وعبد",
    "الكلام في المسجد يأكل الحسنات كما تأكل النار الحطب",
    "نحن قوم لا نأكل حتى نجوع وإذا أكلنا لا نشبع",
    "اعمل لدنياك كأنك تعيش أبدا واعمل لآخرتك كأنك تموت غدا",
]

PLAIN = [
    "صباح الخير يا أحلى جروب ☀️ لا تنسون اجتماع بكرة الساعة ٨",
    "مبروك التخرج يا أبو فهد، الله يوفقك في حياتك العملية 🎓",
    "تذكير: آخر موعد لتسليم الواجب يوم الخميس",
    "جمعة مباركة على الجميع، لا تنسوا قراءة سورة الكهف والصلاة على النبي ﷺ",
    "اللهم ارحم والدينا واغفر لهم 🤍",
    "الجو اليوم حلو مرة، مين يطلع نزهة العصر؟",
    "رمضان قرب، جهزوا أنفسكم بالعبادة والطاعة",
    "شكراً لكل من شارك في الحملة، أثركم وصل 🌿",
]


def make_post(intro_list, quote):
    return random.choice(FILLERS) + random.choice(intro_list).format(quote) + random.choice(ENDINGS)


def strip_marks(text):
    """Plain spelling without diacritics, like most posts."""
    return re.sub("[ً-ٰٟۖ-ۭ]", "", text)


def ayah_sample(n_words=(5, 14)):
    q = quran_index()
    keys = list(q.uthmani)
    simple = {(v["chapter"], v["verse"]): v["text"] for v in json.loads(
        (ROOT / "data" / "quran_simple.json").read_text("utf-8"))["quran"]}
    while True:
        k = random.choice(keys)
        uthmani = random.random() < 0.3
        toks = (q.uthmani[k] if uthmani else strip_marks(simple[k])).split()
        if len(toks) < n_words[0]:
            continue
        length = min(len(toks), random.randint(*n_words))
        start = random.randint(0, len(toks) - length)
        return k, toks[start:start + length], uthmani


def alter(toks):
    vocab = ["والدعاء", "الكريم", "دائما", "والمتقين", "الناس", "رحمة", "العظيم", "قلوبهم"]
    toks = list(toks)
    i = random.randrange(1, len(toks) - 1)
    op = random.choice(["replace", "delete", "insert"])
    if op == "replace":
        new = random.choice([w for w in vocab if normalize(w) != normalize(toks[i])])
        toks[i] = new
    elif op == "delete":
        toks.pop(i)
    else:
        toks.insert(i, random.choice(vocab))
    return toks, op


def distinctive_phrase(doc_filter, count):
    """Phrases from the hadith texts that appear in exactly one hadith."""
    h = hadith_index()
    docs = [d for d in h.docs if doc_filter(d)]
    random.shuffle(docs)
    out = []
    for d in docs:
        m = re.findall(r'"‏?\s*(.+?)\s*‏?"', d["text"])
        matn = max(m, key=len) if m else ""
        words = matn.split()
        if not (5 <= len(words) <= 40):
            continue
        if any(c in matn for c in "\u200f.:،؟") or "قال" in matn[:12]:
            continue  # keep clean spoken phrases only
        length = random.randint(5, min(12, len(words)))
        start = random.randint(0, len(words) - length)
        phrase = " ".join(words[start:start + length])
        norm = normalize(phrase)
        if len(norm.split()) < 5:
            continue
        hits = [x for x in h.docs if norm in x["norm"]]
        if len(hits) == 1:
            out.append((strip_marks(phrase), d))
        if len(out) == count:
            return out
    return out


def main():
    rows = []
    for _ in range(25):
        k, toks, uth = ayah_sample()
        rows.append({"category": "ayah_correct", "post": make_post(AYAH_INTROS, " ".join(toks)),
                     "expect": "ok", "ref": f"{k[0]}:{k[1]}", "original": " ".join(toks), "uthmani": uth})
    for _ in range(25):
        k, toks, uth = ayah_sample()
        bad, op = alter(toks)
        rows.append({"category": "ayah_altered", "post": make_post(AYAH_INTROS, " ".join(bad)),
                     "expect": "bad", "ref": f"{k[0]}:{k[1]}", "original": " ".join(toks), "uthmani": uth,
                     "alteration": op})
    for phrase, d in distinctive_phrase(lambda d: d["sahihain"], 20):
        rows.append({"category": "hadith_sahih", "post": make_post(HADITH_INTROS, phrase),
                     "expect": "ok", "ref": f'{d["book"]} {d["number"]}'})
    for phrase, d in distinctive_phrase(lambda d: d["grader"] == "الألباني" and d["rating"] == "bad", 10):
        rows.append({"category": "hadith_weak", "post": make_post(HADITH_INTROS, phrase),
                     "expect": "bad", "ref": f'{d["book"]} {d["number"]} ({d["grade"]})'})
    for saying in NOT_IN_SIX_BOOKS:
        rows.append({"category": "not_in_six_books", "post": make_post(HADITH_INTROS, saying),
                     "expect": "bad", "ref": "لا يوجد في الكتب الستة"})
    for text in PLAIN:
        rows.append({"category": "plain", "post": text, "expect": "none", "ref": ""})

    with OUT.open("w", encoding="utf-8") as f:
        for i, r in enumerate(rows, 1):
            f.write(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n")
    print(f"{len(rows)} posts -> {OUT}")


if __name__ == "__main__":
    main()
