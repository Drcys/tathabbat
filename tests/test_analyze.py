"""Tests for whole-post analysis (Dorar disabled so tests run offline)."""

from core.analyze import analyze

POST = """صباح الخير يا جماعة
قال الله تعالى: ﴿واستعينوا بالصبر والدعاء وإنها لكبيرة إلا على الخاشعين﴾
وقال رسول الله صلى الله عليه وسلم: "اطلبوا العلم ولو في الصين"
وقال ﷺ: «من غشنا فليس منا»
قل هو الله أحد الله الصمد لم يلد ولم يولد ولم يكن له كفوا أحد
عن أبي هريرة رضي الله عنه أن رسول الله صلى الله عليه وسلم قال لا تغضب
انشرها ولك الأجر"""


def _by_text():
    return {s.text: s for s in analyze(POST, use_dorar=False).segments}


def test_altered_ayah_flagged():
    seg = _by_text()["واستعينوا بالصبر والدعاء وإنها لكبيرة إلا على الخاشعين"]
    assert seg.kind == "quran" and seg.verdict == "bad"


def test_correct_surah_ok():
    seg = _by_text()["قل هو الله أحد الله الصمد لم يلد ولم يولد ولم يكن له كفوا أحد"]
    assert seg.kind == "quran" and seg.verdict == "ok"


def test_attributed_fabrication_flagged():
    seg = _by_text()["اطلبوا العلم ولو في الصين"]
    assert seg.kind == "hadith" and seg.verdict == "bad"


def test_authentic_hadith_ok():
    assert _by_text()["من غشنا فليس منا"].verdict == "ok"


def test_chain_is_stripped():
    seg = _by_text()["لا تغضب"]
    assert seg.kind == "hadith" and seg.verdict == "ok"


def test_plain_text_left_alone():
    segs = _by_text()
    assert segs["صباح الخير يا جماعة"].kind == "text"
    assert segs["انشرها ولك الأجر"].kind == "text"


def test_ayah_number_in_brackets_is_not_a_quote():
    segs = analyze("وَٱسۡتَعِينُواْ بِٱلصَّبۡرِ وَٱلصَّلَوٰةِۚ وَإِنَّهَا لَكَبِيرَةٌ إِلَّا عَلَى ٱلۡخَٰشِعِينَ ﴿45﴾", use_dorar=False).segments
    assert [s.kind for s in segs] == ["quran"] and segs[0].verdict == "ok"


def test_saying_attributed_to_the_quran_but_not_in_it_is_flagged():
    seg = analyze("قال تعالى: ﴿العلم نور والجهل ظلام﴾", use_dorar=False).checked[0]
    assert seg.kind == "quran" and seg.quran.status == "not_found" and seg.verdict == "bad"
    seg = analyze("قال الله تعالى: ﴿ادعوني أستجب لكم إن الله غفور رحيم﴾", use_dorar=False).checked[0]
    assert seg.verdict == "bad" and seg.quran.surah == 40
