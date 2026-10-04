"""Unusual and hostile inputs: the app must never crash, mislabel or render HTML from the user."""

import io
import time
from pathlib import Path

import pytest
from PIL import Image

from core.analyze import analyze
from core.ocr import extract_text


@pytest.mark.parametrize("post", ["", "   \n\n ", "😀🙏🌙", "Good morning everyone!", "12345 678", "؟؟؟ ... !!!",
                                  "https://example.com/abc?x=1", "#جمعة_مباركة @user"])
def test_empty_or_non_arabic_input_has_nothing_to_check(post):
    assert analyze(post, use_dorar=False).checked == []


def test_spelling_variants_of_the_same_ayah_are_all_verified():
    forms = [
        "إن مع العسر يسرا",
        "إِنَّ مَعَ الْعُسْرِ يُسْرًا",
        "ان مع العسر يسرا",
        "إنّ مع العســر يســرا",            # tatweel
        "إِنَّ مَعَ ٱلۡعُسۡرِ يُسۡرࣰا",       # Uthmani
    ]
    for f in forms:
        seg = analyze("قال تعالى: " + f, use_dorar=False).checked[0]
        assert seg.kind == "quran" and seg.verdict == "ok", f


def test_whole_surah_with_ayah_numbers():
    fatiha = ("بسم الله الرحمن الرحيم (1) الحمد لله رب العالمين (2) الرحمن الرحيم (3) مالك يوم الدين (4) "
              "إياك نعبد وإياك نستعين (5) اهدنا الصراط المستقيم (6) صراط الذين أنعمت عليهم غير المغضوب "
              "عليهم ولا الضالين (7)")
    segs = analyze(fatiha, use_dorar=False).checked
    assert segs and all(s.kind == "quran" and s.verdict == "ok" for s in segs)


def test_common_dua_is_not_called_a_misquoted_ayah():
    for dua in ["الحمد لله على نعمة الإسلام", "اللهم إنا نسألك الجنة", "سبحان الله وبحمده سبحان الله العظيم",
                "جزاكم الله خيرا وبارك فيكم"]:
        assert all(s.verdict != "bad" for s in analyze(dua, use_dorar=False).checked), dua


def test_html_in_the_post_is_kept_as_text():
    post = "قال رسول الله ﷺ: «<script>alert(1)</script> إنما الأعمال بالنيات»"
    segs = analyze(post, use_dorar=False).checked
    assert segs and segs[0].kind == "hadith"


def test_app_escapes_user_text():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=120).run()
    at.text_area[0].set_value("قال رسول الله ﷺ: «<img src=x onerror=alert(1)> إنما الأعمال بالنيات»").run()
    [b for b in at.button if b.label == "تحقّق"][0].click().run()
    html = " ".join(m.value for m in at.markdown)
    assert "&lt;img" in html and "<img src=x" not in html


def test_long_post_is_fast():
    block = ("صباح الخير\nقال الله تعالى: ﴿واستعينوا بالصبر والصلاة﴾\n"
             "قال رسول الله ﷺ: «إنما الأعمال بالنيات»\nشاركوا المنشور\n")
    t = time.perf_counter()
    report = analyze(block * 50, use_dorar=False)
    assert len(report.checked) == 100
    assert time.perf_counter() - t < 20


def test_not_an_image_is_a_clean_error():
    with pytest.raises(RuntimeError):
        extract_text(b"this is not an image", "image/png")


def test_blank_image_is_not_read_as_text():
    buf = io.BytesIO()
    Image.new("RGB", (800, 600), "white").save(buf, "PNG")
    try:
        text, engine = extract_text(buf.getvalue())
    except RuntimeError:
        return
    assert engine == "tesseract-low" or not analyze(text, use_dorar=False).checked
