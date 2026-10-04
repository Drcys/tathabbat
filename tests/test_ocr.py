"""OCR tests (the screenshot test is skipped when Tesseract with Arabic is not installed)."""

from pathlib import Path

import pytest

from core.analyze import analyze
from core.ocr import extract_text


def _has_arabic_tesseract():
    # The Arabic model ships in models/tessdata; only the tesseract program is needed.
    from core.ocr import _tesseract_path
    return _tesseract_path() is not None


@pytest.mark.skipif(not _has_arabic_tesseract(), reason="Tesseract (ara) not installed")
def test_screenshot_end_to_end():
    img = (Path(__file__).parent / "sample_post.png").read_bytes()
    text, _ = extract_text(img)
    verdicts = [s.verdict for s in analyze(text, use_dorar=False, from_image=True).checked]
    assert verdicts.count("ok") >= 1 and verdicts.count("bad") >= 2


def test_one_letter_ocr_slip_is_not_called_tampering():
    seg = analyze("قال الله تعالى: (واستعيتوا بالصبر والصلاة وإنها لكبيرة إلا على الخاشعين)",
                  use_dorar=False, from_image=True).checked[0]
    assert seg.verdict == "warn" and seg.maybe_ocr_error


@pytest.mark.skipif(not _has_arabic_tesseract(), reason="Tesseract not installed")
def test_small_photo_with_text_over_landscape():
    # A 320px image of a hadith written over a sunset photo.
    img = (Path(__file__).parent / "sample_photo_small.png").read_bytes()
    text, _ = extract_text(img)
    segs = analyze(text, use_dorar=False, from_image=True).checked
    assert any(s.kind == "hadith" and s.hadith.matches and s.hadith.matches[0].number == "7405" for s in segs)


CALLIGRAPHY = Path(__file__).parent / "sample_calligraphy.png"   # «ألا بذكر الله تطمئن القلوب» in ثلث


@pytest.mark.skipif(not _has_arabic_tesseract(), reason="Tesseract not installed")
def test_calligraphy_goes_to_vision_model(monkeypatch):
    # Tesseract cannot read ثلث; the vision model is asked instead.
    from core import ocr
    monkeypatch.setattr(ocr, "_claude", lambda *a: "```\nألا بذكر الله تطمئن القلوب\n```")
    text, engine = extract_text(CALLIGRAPHY.read_bytes(), "image/png", None, "test-key")
    assert engine == "claude" and text == "ألا بذكر الله تطمئن القلوب"
    seg = analyze(text, use_dorar=False, from_image=True).checked[0]
    assert seg.kind == "quran" and seg.verdict == "ok" and "28" in seg.quran.reference


@pytest.mark.skipif(not _has_arabic_tesseract(), reason="Tesseract not installed")
def test_calligraphy_without_vision_model_is_flagged_not_guessed():
    _, engine = extract_text(CALLIGRAPHY.read_bytes())
    assert engine == "tesseract-low"


def test_vision_falls_back_to_second_model(monkeypatch):
    from core import ocr
    monkeypatch.setattr(ocr, "_tesseract_path", lambda: None)
    monkeypatch.setattr(ocr, "_claude", lambda *a: (_ for _ in ()).throw(RuntimeError("down")))
    monkeypatch.setattr(ocr, "_gemini", lambda *a: "إنما الأعمال بالنيات")
    assert extract_text(CALLIGRAPHY.read_bytes(), "image/png", "g", "c") == ("إنما الأعمال بالنيات", "gemini")


def test_a_misread_word_the_ocr_was_unsure_of_is_not_called_tampering():
    post = "قال الله تعالى: ﴿واستعينوا بالصبر والدعاء وإنها لكبيرة إلا على الخاشعين﴾"
    unsure = analyze(post, use_dorar=False, from_image=True, uncertain={"والدعاء"}).checked[0]
    sure = analyze(post, use_dorar=False, from_image=True, uncertain={"الخاشعين"}).checked[0]
    assert unsure.verdict == "warn"          # "check the reading", never "wrong"
    assert sure.verdict == "bad"             # read with confidence: a real alteration


def test_hadith_not_found_because_of_a_poor_reading_is_not_called_fabricated():
    post = "قال رسول الله ﷺ: «إنما الاعمل بالنيت وإنما لكل امرئ ما نوى»"
    seg = analyze(post, use_dorar=False, from_image=True, uncertain={"الاعمل", "بالنيت"}).checked[0]
    assert seg.verdict != "bad"


def test_misread_opening_bracket_is_removed():
    from core.ocr import _clean
    assert _clean("قال الله تعالى: إواستعينوا بالصبر") == "قال الله تعالى: واستعينوا بالصبر"


def test_reading_with_many_uncertain_words_is_not_reliable():
    from core.ocr import reading_is_reliable
    text = "قال رسول الله صلى الله عليه وسلم"
    assert reading_is_reliable(text, 80, set())
    assert not reading_is_reliable(text, 80, {"قال", "رسول", "صلي"})


@pytest.mark.skipif(not _has_arabic_tesseract(), reason="Tesseract not installed")
def test_vision_reading_wins_when_tesseract_skipped_lines(monkeypatch):
    # Decorated, diacritised hadith card: Tesseract reads only fragments.
    from core import ocr
    full = "قال الرسول صلى الله عليه وسلم: «لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه»"
    monkeypatch.setattr(ocr, "_gemini", lambda *a: full)
    r = ocr.read_image((Path(__file__).parent / "sample_decorated_hadith.png").read_bytes(), "image/png", "k")
    assert r.engine == "gemini" and r.text == full
    seg = analyze(r.text, use_dorar=False, from_image=True).checked[0]
    assert seg.kind == "hadith" and seg.verdict == "ok"


@pytest.mark.skipif(not _has_arabic_tesseract(), reason="Tesseract not installed")
def test_tesseract_kept_when_it_agrees_with_vision(monkeypatch):
    from core import ocr
    img = (Path(__file__).parent / "sample_post.png").read_bytes()
    local = ocr._tesseract(img)
    monkeypatch.setattr(ocr, "_gemini", lambda *a: local)
    assert ocr.read_image(img, "image/png", "k").engine == "tesseract"


def test_vision_text_wrapped_like_the_image_is_one_hadith():
    # Gemini keeps the image's line breaks; a hadith cut across lines must stay whole,
    # and the reference lines under it are not checked as sayings.
    from core.ocr import _tidy_vision
    read = ("فقرة حديث نبوي\nقال الرَّسولُ -صلَّى اللهُ عليه وسلَّم-: (لا\nيُؤْمِنُ أحَدُكُم، حتَّى يُحِبَّ لأخِيهِ ما\n"
            "يُحِبُّ لِنَفْسِهِ)\nرواه البخاري، في صحيح البخاري، عن أنس بن مالك،\nالصفحة أو الرقم: 13، صحيح")
    segs = analyze(_tidy_vision(read), use_dorar=False, from_image=True).checked
    assert len(segs) == 1 and segs[0].verdict == "ok" and segs[0].hadith.matches[0].number == "13"
    read = ("قال رسول الله ﷺ\nلا تَجْعَلُوا بُيُوتَكُمْ مَقابِرَ،\nإنَّ الشَّيْطانَ يَنْفِرُ مِنَ البَيْتِ\n"
            "الَّذِي تُقْرَأُ فيه سُورَةُ البَقَرَةِ.\nصحيح مسلم: 780")
    segs = analyze(_tidy_vision(read), use_dorar=False, from_image=True).checked
    assert len(segs) == 1 and segs[0].hadith.matches[0].number == "780"


def test_salutation_in_brackets_is_not_a_quote():
    segs = analyze("عن أبي هريرة (رضي الله عنه) قال: قال رسول الله (صلى الله عليه وسلم): «من غشنا فليس منا»",
                   use_dorar=False).checked
    assert len(segs) == 1 and segs[0].verdict == "ok"


def test_tesseract_reading_that_skipped_a_line_is_not_kept():
    from core.ocr import _same_reading
    full = "اوصاني خليلي ان لا تشرك بالله شيئا ولا تترك صلاه مكتوبه متعمدا ولا تشرب الخمر فانها مفتاح كل شر"
    skipped = "اوصاني خليلي ان لا تشرك بالله شيئا ولا تترك صلاه مكتوبه متعمدا"
    assert not _same_reading(skipped, full)
    assert _same_reading(full.replace("تشرك", "تشرك"), full)
    assert _same_reading(full.replace("خليلي", "خليلى"), full)


def test_latin_watermark_is_dropped_from_vision_reading():
    from core.ocr import _tidy_vision
    assert _tidy_vision("ALBETAQA.SITE من وصايا الرسول.\nصحيح مسلم: 780") == "من وصايا الرسول.\nصحيح مسلم: 780"
