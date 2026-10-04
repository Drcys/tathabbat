"""Tests for the Quran checker. Run with:  python -m pytest -q"""

from core.normalize import normalize
from core.quran import get_index

idx = get_index()


def test_whole_surah_verified():
    r = idx.check("قُلْ هُوَ اللَّهُ أَحَدٌ اللَّهُ الصَّمَدُ لَمْ يَلِدْ وَلَمْ يُولَدْ وَلَمْ يَكُن لَّهُ كُفُوًا أَحَدٌ")
    assert r.status == "verified"
    assert (r.surah, r.ayah_from, r.ayah_to) == (112, 1, 4)


def test_plain_spelling_without_diacritics():
    r = idx.check("الحمد لله رب العالمين الرحمن الرحيم مالك يوم الدين")
    assert r.status == "verified"
    assert (r.surah, r.ayah_from, r.ayah_to) == (1, 2, 4)


def test_replaced_word_is_located():
    r = idx.check("واستعينوا بالصبر والدعاء وإنها لكبيرة إلا على الخاشعين")
    assert r.status == "altered"
    assert (r.surah, r.ayah_from) == (2, 45)
    assert len(r.differences) == 1
    d = r.differences[0]
    assert d.kind == "replace" and d.written == "والدعاء" and normalize(d.correct) == "والصلاه"


def test_added_word_is_located():
    r = idx.check("إن الله مع الصابرين والمتقين")
    assert r.status == "altered"
    assert any(d.kind == "extra" and d.written == "والمتقين" for d in r.differences)


def test_missing_word_is_located():
    # Correct: الله لا إله إلا هو الحي القيوم (2:255) — "الحي" dropped.
    r = idx.check("الله لا إله إلا هو القيوم لا تأخذه سنة ولا نوم")
    assert r.status == "altered"
    assert any(d.kind == "missing" and normalize(d.correct) == "الحي" for d in r.differences)


def test_multi_ayah_span():
    r = idx.check("إنا أعطيناك الكوثر فصل لربك وانحر إن شانئك هو الأبتر")
    assert r.status == "verified"
    assert (r.surah, r.ayah_from, r.ayah_to) == (108, 1, 3)


def test_non_quran_text():
    assert idx.check("النظافة من الإيمان والعلم نور").status == "not_found"


def test_too_short():
    assert idx.check("الله أكبر").status == "too_short"


def test_partial_ayah_is_not_an_error():
    r = idx.check("لا يكلف الله نفسا إلا وسعها")
    assert r.status == "verified"
    assert (r.surah, r.ayah_from) == (2, 286)


def test_uthmani_script_copied_from_mushaf():
    r = idx.check("وَٱسۡتَعِينُواْ بِٱلصَّبۡرِ وَٱلصَّلَوٰةِۚ وَإِنَّهَا لَكَبِيرَةٌ إِلَّا عَلَى ٱلۡخَٰشِعِينَ")
    assert r.status == "verified" and (r.surah, r.ayah_from) == (2, 45)


def test_uthmani_altered():
    r = idx.check("وَٱسۡتَعِينُواْ بِٱلصَّبۡرِ وَٱلدُّعَآءِۚ وَإِنَّهَا لَكَبِيرَةٌ إِلَّا عَلَى ٱلۡخَٰشِعِينَ")
    assert r.status == "altered"


def test_open_tanween_spacing():
    assert idx.check("ذَٰلِكَ ٱلۡكِتَٰبُ لَا رَيۡبَۛ فِيهِۛ هُدٗى لِّلۡمُتَّقِينَ").status == "verified"


def test_fragment_search_finds_ayah_from_words_read_in_an_image():
    assert idx.search_fragment("تطمئن القلوب")[0]["reference"].endswith("28")
    assert idx.search_fragment("تطمين القلوب")[0]["surah"] == 13        # one letter misread
    assert idx.search_fragment("قل هو الله احد")[0]["surah"] == 112
