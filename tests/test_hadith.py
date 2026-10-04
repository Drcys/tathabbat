"""Tests for the hadith finder and the Dorar parser. Run with:  python -m pytest -q"""

from core.dorar import parse
from core.normalize import normalize
from core.hadith import get_index

idx = get_index()


def test_bukhari_first_hadith():
    r = idx.search("إنما الأعمال بالنيات")
    assert r.status == "found"
    assert (r.matches[0].book, r.matches[0].number) == ("صحيح البخاري", "1")


def test_muslim_numbering():
    r = idx.search("الدين النصيحة")
    assert r.status == "found"
    assert (r.matches[0].book, r.matches[0].number) == ("صحيح مسلم", "55")


def test_short_hadith():
    assert idx.search("لا تغضب").status == "found"


def test_common_phrase_does_not_fake_a_match():
    # "من الإيمان" appears in many hadiths; the saying itself is not in the Sahihs.
    assert idx.search("حب الوطن من الإيمان").status == "not_found"
    assert idx.search("النظافة من الإيمان").status == "not_found"


def test_known_fabrications_not_found():
    for text in ["اطلبوا العلم ولو في الصين", "اختلاف أمتي رحمة", "تفاءلوا بالخير تجدوه"]:
        assert idx.search(text).status == "not_found", text


SAMPLE = """
<div class="hadith" style="text-align:justify;">1 - اطلبوا العلمَ ولو بالصينِ</div>
<div class="hadith-info">
<span class="info-subtitle">الراوي:</span> أنس بن مالك | <span class="info-subtitle">المحدث:</span>
<span style="color:blue;">ابن حبان</span> | <span class="info-subtitle">المصدر:</span> المجروحين |
<span class="info-subtitle">الصفحة أو الرقم:</span> 1/382 | <span class="info-subtitle">خلاصة حكم المحدث:</span>
<span>باطل لا أصل له</span></div>
"""


def test_dorar_parser():
    [e] = parse(SAMPLE)
    assert e.text == "اطلبوا العلمَ ولو بالصينِ"
    assert e.narrator == "أنس بن مالك"
    assert e.scholar == "ابن حبان"
    assert e.source == "المجروحين"
    assert e.number == "1/382"
    assert e.grade == "باطل لا أصل له"


def test_sunan_with_scholar_grade():
    r = idx.search("الكيس من دان نفسه وعمل لما بعد الموت")
    assert r.status == "found"
    m = r.matches[0]
    assert m.book == "سنن ابن ماجه" and m.grader == "الألباني" and m.rating == "bad"


def test_sahihain_preferred_on_tie():
    r = idx.search("إنما الأعمال بالنيات")
    assert r.matches[0].book == "صحيح البخاري" and r.matches[0].rating == "ok"


def test_found_by_meaning_is_never_called_authentic():
    # «الحسنة» vs the real wording «الطيبة»: found by the meaning model.
    r = idx.search("الكلمة الحسنة صدقة")
    assert r.status == "similar" and r.matches[0].semantic
    assert "الكلمه الطيبه صدقه" in normalize(r.matches[0].text)


def test_meaning_search_off_for_plain_text():
    assert idx.search("اللهم ارحم والدينا واغفر لهم", by_meaning=False).status != "found"
