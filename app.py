"""تثبَّت — web app.

Run locally:   streamlit run app.py
"""

from html import escape
from urllib.parse import quote

import streamlit as st

import hashlib
import os

from core import translation
from core.analyze import analyze
from core.ocr import available_engines, read_image
from core.hadith import get_index as hadith_index
from core.quran import get_index as quran_index

st.set_page_config(page_title="تثبَّت | التحقق من النصوص الشرعية", page_icon="🔍", layout="centered")

SAMPLES = {
    "منشور فيه آية محرّفة": "صباح الخير 🌸\nقال الله تعالى: ﴿واستعينوا بالصبر والدعاء وإنها لكبيرة إلا على الخاشعين﴾\nانشرها ولك الأجر",
    "منشور فيه حديث لا يصح": "وقال رسول الله صلى الله عليه وسلم: \"اطلبوا العلم ولو في الصين\"\nالعلم نور 📚",
    "منشور سليم": "قال ﷺ: «من غشنا فليس منا»\nقل هو الله أحد الله الصمد لم يلد ولم يولد ولم يكن له كفوا أحد",
}

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;600;700&family=Amiri+Quran&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextArea textarea, .stButton button, p, li, label {
  font-family: 'IBM Plex Sans Arabic', sans-serif !important;
}
.stApp { direction: rtl; }
.ayah-pick { font-family: 'Amiri Quran', serif !important; font-size: 1.15rem; line-height: 2; padding: 4px 0; }
.stTextArea textarea { direction: rtl; text-align: right; font-size: 1.05rem; line-height: 1.8; }
h1, h2, h3, p, li, label { text-align: right; }
.hero h1 { font-size: 2.6rem; margin: 0; color: #0b3d5c; }
.hero p { color: #5b6180; margin: .2rem 0 1.2rem; font-size: 1.05rem; }
.card { border: 1px solid #e3e6ef; border-right: 6px solid var(--c); border-radius: 12px;
        padding: 14px 18px; margin: 12px 0; background: #fff; }
.card.ok { --c: #12805c; } .card.warn { --c: #b7791f; } .card.bad { --c: #b83232; }
.badge { display: inline-block; padding: 2px 12px; border-radius: 20px; color: #fff;
         font-weight: 700; font-size: .85rem; background: var(--c); }
.kind { color: #5b6180; font-size: .85rem; margin-inline-start: 8px; }
.quote { font-size: 1.1rem; margin: 10px 0 6px; line-height: 1.9; }
.en { direction: ltr; text-align: left; font-family: Georgia, 'Times New Roman', serif !important; font-size: .98rem;
      line-height: 1.6; background: #f7f9fc; border-left: 4px solid #0f8b7a; border-radius: 8px; padding: 8px 12px; margin-top: 8px; }
.en small { display: block; color: #6b7280; font-family: 'IBM Plex Sans Arabic', sans-serif !important; font-size: .75rem; margin-top: 4px; }
.mushaf { font-family: 'Amiri Quran', serif !important; font-size: 1.35rem; line-height: 2.3;
          background: #f6f7fb; border-radius: 10px; padding: 8px 14px; margin-top: 6px; }
.ref { color: #0f8b7d; font-weight: 700; }
.diff { margin: 4px 0; }
.diff s { color: #b83232; } .diff b { color: #12805c; }
.src { background: #f6f7fb; border-radius: 10px; padding: 8px 14px; margin-top: 8px; font-size: .95rem; }
.src small { color: #5b6180; }
.summary { display: flex; gap: 10px; margin: 8px 0 4px; flex-wrap: wrap; }
.pill { border-radius: 10px; padding: 8px 14px; background: #f6f7fb; font-weight: 600; }
.note { color: #5b6180; font-size: .85rem; border-top: 1px solid #e3e6ef; margin-top: 28px; padding-top: 10px; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="جارٍ تحميل نص المصحف وكتب الحديث…")
def load():
    return quran_index(), hadith_index()


load()

st.markdown(
    '<div class="hero"><h1>تثبَّت</h1>'
    "<p>الصق منشوراً فيه آيات أو أحاديث، وسنتحقق من كل نص ونعيده إلى مصدره.</p></div>",
    unsafe_allow_html=True,
)

if "post" not in st.session_state:
    st.session_state.post = ""
    st.session_state.from_image = False
    st.session_state.image_hash = None


def secret(name):
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:
        return os.environ.get(name)


ENGINE_NAMES = {"tesseract": "Tesseract", "claude": "نموذج الرؤية Claude", "gemini": "نموذج الرؤية Gemini"}


tab_text, tab_image = st.tabs(["✍️ لصق نص", "🖼️ رفع صورة المنشور"])
with tab_text:
    cols = st.columns(len(SAMPLES))
    for col, (label, text) in zip(cols, SAMPLES.items()):
        if col.button(label, use_container_width=True):
            st.session_state.post = text
            st.session_state.from_image = False
with tab_image:
    g_key, c_key = secret("GEMINI_API_KEY"), secret("ANTHROPIC_API_KEY")
    if not (g_key or c_key):
        # No key on this server: the visitor may use their own, for this session only.
        with st.expander("🔑 تفعيل قراءة الخطوط الزخرفية (الثلث، الديواني، الكوفي، خط اليد)"):
            st.markdown("أنشئ مفتاحاً مجانياً من [Google AI Studio](https://aistudio.google.com/app/apikey) "
                        "والصقه هنا. يبقى في هذه الجلسة فقط ولا يُحفظ.")
            g_key = st.text_input("مفتاح Gemini", type="password", key="user_gemini_key") or None
    engines = available_engines(g_key, c_key)
    vision = any(e in engines for e in ("claude", "gemini"))
    if not engines:
        st.info("قراءة الصور غير مفعّلة على هذا الجهاز (تحتاج Tesseract أو مفتاح نموذج رؤية). تعمل في النسخة المنشورة.")
    else:
        st.caption("يقرأ الخط المطبوع ولقطات الشاشة، والخطوط الزخرفية (الثلث، الديواني، الكوفي) وخط اليد"
                   + (" عبر نموذج رؤية." if vision else "، والأخيرة تحتاج تفعيل نموذج الرؤية."))
        up = st.file_uploader("ارفع صورة المنشور", type=["png", "jpg", "jpeg", "webp"])
        if up is not None:
            data = up.getvalue()
            digest = hashlib.sha1(data + (b"v" if vision else b"")).hexdigest()
            st.image(data, width=320)
            if st.session_state.image_hash != digest:
                st.session_state.image_hash = digest
                st.session_state.engine = None
                with st.spinner("جارٍ قراءة النص من الصورة…"):
                    try:
                        reading = read_image(data, up.type or "image/png", g_key, c_key)
                        text, engine = reading.text, reading.engine
                        st.session_state.uncertain = reading.uncertain
                        st.session_state.engine = engine
                        st.session_state.from_image = engine != "tesseract-low"
                        if engine != "tesseract-low":
                            st.session_state.post = text
                    except RuntimeError:
                        st.session_state.from_image = False
                        st.error("تعذّرت قراءة النص من الصورة. جرّب صورة أوضح أو الصق النص يدوياً.")
            engine = st.session_state.get("engine")
            if engine == "tesseract-low":
                st.warning("يبدو أن النص مكتوب بخط زخرفي أو بخط اليد، وقراءته الكاملة تحتاج نموذج الرؤية "
                           "(فعّله من «🔑 تفعيل قراءة الخطوط الزخرفية» أعلاه). أو اكتب أي كلمة تتبيّنها من الصورة "
                           "وسنجد الآية:")
                frag = st.text_input("كلمة أو كلمتان من الصورة", placeholder="مثال: تطمئن القلوب", key="fragment")
                if frag.strip():
                    found = quran_index().search_fragment(frag)
                    if not found:
                        st.caption("لم نجد آية فيها هذه الكلمات. جرّب كلمة أخرى.")
                    elif len(found) > 1:
                        st.caption("كلما كتبت كلمات أكثر من الصورة كانت النتيجة أدق.")
                    for n, a in enumerate(found):
                        c1, c2 = st.columns([5, 1])
                        c1.markdown(f"<div class='ayah-pick'><b>{escape(a['reference'])}</b> · {escape(a['uthmani'])}</div>",
                                    unsafe_allow_html=True)
                        if c2.button("هي هذه", key=f"pick{n}"):
                            st.session_state.post = a["text"]
                            st.session_state.from_image = False
                            st.session_state.engine = "picked"
                            st.rerun()
            elif engine == "picked":
                st.success("وضعنا نص الآية المختارة في المربع أدناه. قارنه بالصورة ثم اضغط «تحقّق».")
            elif engine and st.session_state.from_image:
                st.success(f"استخرجنا النص ({ENGINE_NAMES.get(engine, engine)}) ووضعناه في المربع أدناه. "
                           "راجعه وصحّح أي كلمة قُرئت خطأً، ثم اضغط «تحقّق».")

post = st.text_area("نص المنشور", key="post", height=180, placeholder="الصق نص المنشور هنا…")
use_dorar = st.toggle("البحث الإضافي في الدرر السنية للأحاديث غير الموجودة في الكتب الستة", value=True)
show_en = st.toggle("إظهار الترجمة الإنجليزية (للمعرّفين بالإسلام)", value=True)
go = st.button("تحقّق", type="primary", use_container_width=True)

LABELS = {
    ("quran", "ok"): "آية صحيحة",
    ("quran", "bad"): "آية فيها خطأ",
    ("quran", "warn"): "تحقق من القراءة",
    ("hadith", "ok"): "حديث موثّق",
    ("hadith", "warn"): "يحتاج تحققاً",
    ("hadith", "bad"): "لا يصح أو لم يثبت",
}
KIND = {"quran": "قرآن كريم", "hadith": "حديث"}


def label(seg) -> str:
    if seg.kind == "quran" and seg.quran.status == "not_found":
        return "ليست آية بهذا اللفظ" if seg.verdict == "bad" else "تحقق من القراءة"
    return LABELS[(seg.kind, seg.verdict)]


def quran_card(seg) -> str:
    q = seg.quran
    if q.status == "not_found":
        parts = [f'<div class="quote">{escape(seg.text)}</div>',
                 "<div>هذا النص منسوب إلى القرآن، لكنه <b>غير موجود في المصحف</b> بهذا اللفظ. "
                 "لا تنشره على أنه آية.</div>"]
        if seg.nearest:
            n = seg.nearest
            parts.append(f'<div style="margin-top:6px">أقرب آية إليه: <span class="ref">{escape(n["reference"])}</span></div>'
                         f'<div class="mushaf">{escape(n["uthmani"])}</div>')
        return "".join(parts)
    parts = [f'<div class="quote">{escape(seg.text)}</div>',
             f'<div>الموضع: <span class="ref">{escape(q.reference)}</span></div>']
    for d in q.differences:
        if d.kind == "replace":
            parts.append(f'<div class="diff">كلمة مبدّلة: <s>{escape(d.written)}</s> ← الصحيح <b>{escape(d.correct)}</b></div>')
        elif d.kind == "missing":
            parts.append(f'<div class="diff">كلمة ناقصة: <b>{escape(d.correct)}</b></div>')
        else:
            parts.append(f'<div class="diff">كلمة زائدة ليست في الآية: <s>{escape(d.written)}</s></div>')
    if seg.maybe_ocr_error:
        parts.append('<div class="src">الفرق في حرف أو حرفين فقط، وقد يكون من قراءة الصورة لا من المنشور نفسه. '
                     "قارن الكلمة بالصورة، وصحّحها في المربع إن كانت قراءة خاطئة.</div>")
    parts.append(f'<div class="mushaf">{escape(q.correct_text)}</div>')
    parts.append(english_block(translation.quran(q.surah, q.ayah_from, q.ayah_to), translation.QURAN_SOURCE))
    return "".join(parts)


def dorar_link(text: str) -> str:
    url = "https://dorar.net/hadith/search?q=" + quote(text)
    return f'<div style="margin-top:8px"><a href="{url}" target="_blank">ابحث عنه في الدرر السنية ↗</a></div>'


def match_block(m) -> str:
    note = "" if m.book.startswith("صحيح") else (
        "<br><small>الحكم على الرواية كاملة كما وردت في الكتاب، وقد يختلف حكم جزء منها.</small>")
    return (f'<div>المصدر: <span class="ref">{escape(m.book)}، رقم {escape(m.number)}</span></div>'
            f'<div>الحكم: <b>{escape(m.grade)}</b>'
            + (f' <small>({escape(m.grader)})</small>' if m.grader else "") + f"{note}</div>"
            f'<div class="src">{escape(m.text[:400])}</div>'
            + english_block(m.english[:600] + ("…" if len(m.english) > 600 else ""), translation.HADITH_SOURCE))


def english_block(text: str, source: str) -> str:
    if not (show_en and text):
        return ""
    return f'<div class="en">{escape(text)}<small>English translation: {escape(source)}</small></div>'



def hadith_card(seg) -> str:
    h = seg.hadith
    parts = [f'<div class="quote">«{escape(seg.text)}»</div>']
    if seg.reading_doubtful and h.status != "found":
        parts.append("<div class='diff'>بعض كلمات هذا النص قُرئت من الصورة بثقة منخفضة، فقد يكون عدم العثور عليه "
                     "بسبب القراءة. صحّح النص في المربع أعلاه ثم أعد التحقق.</div>")
    if h.status == "found":
        parts.append(match_block(h.matches[0]))
    elif h.status == "similar":
        if h.matches[0].semantic:
            parts.append("<div>هذا اللفظ غير موجود في كتب الحديث، لكن <b>نموذج المعاني</b> وجد حديثاً بالمعنى نفسه "
                         "بألفاظ أخرى. انشره بلفظه الصحيح أدناه:</div>")
            parts.append(match_block(h.matches[0]))
        else:
            parts.append("<div>لم نجد اللفظ نفسه، لكن وجدنا نصاً قريباً منه في كتب الحديث. قارن بينهما قبل النشر:</div>")
            for m in h.matches[:2]:
                parts.append(match_block(m))
    else:
        parts.append("<div>لم يُعثر على هذا النص في الكتب الستة (البخاري، مسلم، أبو داود، الترمذي، النسائي، ابن ماجه).</div>")
        if seg.dorar:
            parts.append("<div style='margin-top:8px'>ما قاله أهل العلم (من الدرر السنية):</div>")
            for e in seg.dorar[:3]:
                parts.append(
                    f'<div class="src">{escape(e.text[:250])}<br>'
                    f"<small>المحدث: {escape(e.scholar)} · المصدر: {escape(e.source)} {escape(e.number)}</small><br>"
                    f"<b>حكم المحدث: {escape(e.grade)}</b></div>"
                )
        else:
            parts.append('<div class="src">لا تنشره منسوباً إلى النبي ﷺ قبل التحقق منه أو سؤال مختص.</div>')
    parts.append(dorar_link(seg.text))
    return "".join(parts)


if go and post.strip():
    with st.spinner("جارٍ التحقق…"):
        report = analyze(post, use_dorar=use_dorar, from_image=st.session_state.from_image,
                         uncertain=st.session_state.get("uncertain", frozenset()))
    checked = report.checked
    if not checked:
        st.info("لم نجد في النص آيات أو أحاديث للتحقق منها.")
    else:
        counts = {v: sum(s.verdict == v for s in checked) for v in ("ok", "warn", "bad")}
        st.markdown(
            '<div class="summary">'
            f'<div class="pill">✅ موثّق: {counts["ok"]}</div>'
            f'<div class="pill">⚠️ يحتاج تحققاً: {counts["warn"]}</div>'
            f'<div class="pill">❌ خطأ أو لم يثبت: {counts["bad"]}</div></div>',
            unsafe_allow_html=True,
        )
        for seg in checked:
            body = quran_card(seg) if seg.kind == "quran" else hadith_card(seg)
            st.markdown(
                f'<div class="card {seg.verdict}"><span class="badge">{label(seg)}</span>'
                f'<span class="kind">{KIND[seg.kind]}</span>{body}</div>',
                unsafe_allow_html=True,
            )
        plain = [s.text for s in report.segments if s.kind == "text"]
        if plain:
            with st.expander("نصوص عادية لم تُفحص"):
                for t in plain:
                    st.write(t)
    feedback = secret("FEEDBACK_URL")
    if feedback:
        st.link_button("📝 قيّم تجربتك (دقيقة واحدة)", feedback, use_container_width=True)

st.markdown(
    '<div class="note">تثبَّت أداة مدعومة بالذكاء الاصطناعي وليست مفتياً ولا محدّثاً: لا تولّد نصاً شرعياً ولا تُصدر حكماً، '
    "بل تنقل النص من مصدره والحكم من قائله، وتحيل إلى المختص عند غياب المرجع.<br>"
    "المصادر: نص المصحف (رواية حفص)، الكتب الستة مع أحكام الشيخ الألباني على السنن الأربع، الدرر السنية.</div>",
    unsafe_allow_html=True,
)
