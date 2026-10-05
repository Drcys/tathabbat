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


def test_verified_texts_come_with_published_english_translation():
    from core import translation
    assert translation.quran(13, 28).startswith("those who believe and whose hearts find tranquility")
    seg = analyze("قال ﷺ: «لا يؤمن أحدكم حتى يحب لأخيه ما يحب لنفسه»", use_dorar=False).checked[0]
    assert "None of you will have faith" in seg.hadith.matches[0].english


def test_reference_in_brackets_is_not_checked_as_a_hadith():
    post = 'عن عمر بن الخطاب رضي الله عنه قال: سمعت رسول الله صلى الله عليه وسلم يقول: "إنما الأعمال بالنيات، وإنما لكل امرئ ما نوى" (رواه البخاري ومسلم)'
    segs = analyze(post, use_dorar=False).checked
    assert len(segs) == 1 and segs[0].verdict == "ok"


def test_dorar_grading_that_denies_the_hadith_is_not_called_authentic():
    from core.analyze import dorar_rating
    assert dorar_rating("رفعه إلى النبي صلى الله عليه وسلم ليس بصحيح") == "bad"
    assert dorar_rating("ليس بحديث، لكن معناه صحيح") == "bad"
    assert dorar_rating("ضعيف لا يصح") == "bad"
    assert dorar_rating("لا أصل له") == "bad"
    assert dorar_rating("صحيح") == "ok"
    assert dorar_rating("إسناده حسن") == "ok"


def test_dorar_result_with_other_wording_does_not_decide_the_verdict():
    from core.analyze import Segment
    from core.dorar import DorarEntry
    from core.hadith import HadithResult
    seg = Segment("hadith", "النظافة من الإيمان", hadith=HadithResult(status="not_found", input_text=""))
    seg.dorar = [DorarEntry(text="الطهور شطر الإيمان", grade="صحيح")]
    assert seg.verdict == "bad"
    seg.dorar = [DorarEntry(text="النَّظافةُ مِن الإيمانِ .", grade="رفعه إلى النبي صلى الله عليه وسلم ليس بصحيح")]
    assert seg.verdict == "bad" and seg.dorar_match is not None


def test_scholars_who_differ_on_the_same_words_give_check_not_a_verdict():
    from core.analyze import Segment
    from core.dorar import DorarEntry
    from core.hadith import HadithResult
    seg = Segment("hadith", "الجنة تحت أقدام الأمهات", hadith=HadithResult(status="not_found", input_text=""))
    seg.dorar = [DorarEntry(text="الجنة تحت أقدام الأمهات", grade="منكر"),
                 DorarEntry(text="الجنة تحت أقدام الأمهات", grade="صحيح")]
    assert seg.verdict == "warn" and seg.scholars_differ
    seg.dorar = [DorarEntry(text="الجنة تحت أقدام الأمهات", grade="منكر"),
                 DorarEntry(text="الجنة تحت أقدام الأمهات", grade="موضوع")]
    assert seg.verdict == "bad" and not seg.scholars_differ


def test_nearest_ayah_needs_most_of_the_words():
    seg = analyze("قال الله تعالى: «العلم نور يقذفه الله في القلب»", use_dorar=False).checked[0]
    assert seg.quran.status == "not_found" and seg.nearest is None


def test_ayah_display_keeps_tanween_alif_attached():
    seg = analyze("قال تعالى: ﴿وقل رب زدني علما﴾", use_dorar=False).checked[0]
    assert "ࣰ ا" not in seg.quran.correct_text
