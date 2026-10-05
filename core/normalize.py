"""Arabic text normalization shared by the Quran and hadith matchers.

Posts on social media are written with inconsistent spelling: some keep the
diacritics, some drop them, and the hamza / alef / ta-marbuta forms vary.
Before comparing anything we reduce every text to one plain form so that
only real wording differences remain.
"""

import re

# Harakat, Quranic annotation marks, superscript alef, tatweel.
_DIACRITICS = re.compile("[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640\u08D3-\u08FF]")
# Some Uthmani texts split the tanween alif off its word after the open
# fathatan (خَيۡرࣰ ا, هُدࣰ ى); glue it back. Only a standalone ا/ى is glued:
# after the other open tanween marks a space is a real word boundary (بَرَآءَةࣱ مِّنَ).
_OPEN_TANWEEN_GAP = re.compile("(\u08F0)\\s+(?=[\u0627\u0649](?:[^\u0621-\u064A\u0671]|$))")
# Anything that is not an Arabic letter or a space (digits, punctuation, ﴿﴾ …).
_ORPHAN_TANWEEN = re.compile("\u08F0(?=[^\u0627\u0649\u0621-\u064A]|$)")
_NON_LETTERS = re.compile("[^\u0621-\u064A\\s]")
_SPACES = re.compile(r"\s+")
# "و" is never a word on its own; a space after it (و استعينوا) is a typing or
# reading slip, so it is joined to the next word.
_LONE_WAW = re.compile("(?:^|(?<= ))و (?=[ء-ي])")

_LETTER_MAP = str.maketrans({
    "إ": "ا", "أ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
    # Persian/Urdu keyboards type these for the Arabic letters
    "ی": "ي", "ې": "ي", "ک": "ك", "ہ": "ه", "ۃ": "ه", "ە": "ه",
})


def normalize(text: str, uthmani: bool = False) -> str:
    """Return a plain, comparable form of an Arabic string.

    uthmani=True is used for text in the Uthmani script of the Mushaf, where
    the small "dagger" alef (ٰ) stands for a written alef: ٱلْخَٰشِعِينَ -> الخاشعين.
    """
    text = _OPEN_TANWEEN_GAP.sub(r"\1", text)
    if uthmani:
        text = text.replace("\u0670", "ا")
        # A quote cut right after the open fathatan (…طُولࣰ) has lost its tanween
        # alif, which the data writes as a separate token: restore it.
        text = _ORPHAN_TANWEEN.sub("\u08F0ا", text)
    text = _DIACRITICS.sub("", text)
    text = text.translate(_LETTER_MAP)
    text = _NON_LETTERS.sub(" ", text)
    text = _SPACES.sub(" ", text).strip()
    return _LONE_WAW.sub("و", text)


def words(text: str, uthmani: bool = False) -> list[str]:
    """Normalize and split into words."""
    norm = normalize(text, uthmani)
    return norm.split() if norm else []
