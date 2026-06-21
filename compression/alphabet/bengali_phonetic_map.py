"""Bengali (বাংলা) phonetic coordinate mapping for the Lexis character layer.

This module mirrors the structure of ``compression/alphabet/phonetic_map.py``
but covers the Bengali script (Unicode block U+0980–U+09FF).

Phonetic class scheme
---------------------
Class 0  — স্বরবর্ণ   (independent vowels: অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ)
Class 1  — ব্যঞ্জনবর্ণ গুচ্ছ ক  (velar stops:  ক খ গ ঘ ঙ)
Class 2  — ব্যঞ্জনবর্ণ গুচ্ছ চ  (palatal stops: চ ছ জ ঝ ঞ)
Class 3  — ব্যঞ্জনবর্ণ গুচ্ছ ট  (retroflex stops: ট ঠ ড ঢ ণ)
Class 4  — ব্যঞ্জনবর্ণ গুচ্ছ ত  (dental stops: ত থ দ ধ ন)
Class 5  — ব্যঞ্জনবর্ণ গুচ্ছ প  (labial stops: প ফ ব ভ ম)
Class 6  — ব্যঞ্জনবর্ণ গুচ্ছ য  (semivowels & sibilants: য র ল শ ষ স হ)
Class 7  — ব্যঞ্জনবর্ণ বিবিধ    (extra consonants: ড় ঢ় য় ৎ ং ঃ ঁ)
Class 8  — কার / মাত্রা         (dependent vowel signs / matras)
Class 9  — বিরাম ও সংখ্যা      (Bengali punctuation & numerals ০–৯)
Class 10 — হসন্ত / যুক্তাক্ষর   (virama ্  and Zero-Width Joiner / ZWNJ)

All other code-points (including ASCII) fall back to (11, 0).

Integration note
----------------
This map is *supplementary*.  The existing ``get_coords()`` in phonetic_map.py
already handles ASCII.  To use Bengali-aware coords in the pipeline, call
``get_bengali_coords()`` from this module for characters in U+0980–U+09FF,
and keep calling the original ``get_coords()`` for everything else.
"""

from __future__ import annotations

BENGALI_PHONETIC_CLASSES: dict[str, tuple[int, int]] = {

    # ------------------------------------------------------------------
    # Class 0 — Independent vowels (স্বরবর্ণ)
    # ------------------------------------------------------------------
    "অ": (0, 0),   # a   — inherent vowel / schwa
    "আ": (0, 1),   # aa
    "ই": (0, 2),   # i   (short)
    "ঈ": (0, 3),   # ii  (long)
    "উ": (0, 4),   # u   (short)
    "ঊ": (0, 5),   # uu  (long)
    "ঋ": (0, 6),   # ri  (vocalic r)
    "এ": (0, 7),   # e
    "ঐ": (0, 8),   # oi  (diphthong)
    "ও": (0, 9),   # o
    "ঔ": (0, 10),  # ou  (diphthong)

    # ------------------------------------------------------------------
    # Class 1 — Velar stops (ক-বর্গ)
    # ------------------------------------------------------------------
    "ক": (1, 0),   # k   (voiceless unaspirated)
    "খ": (1, 1),   # kh  (voiceless aspirated)
    "গ": (1, 2),   # g   (voiced unaspirated)
    "ঘ": (1, 3),   # gh  (voiced aspirated)
    "ঙ": (1, 4),   # ng  (nasal)

    # ------------------------------------------------------------------
    # Class 2 — Palatal stops (চ-বর্গ)
    # ------------------------------------------------------------------
    "চ": (2, 0),   # ch  (voiceless unaspirated)
    "ছ": (2, 1),   # chh (voiceless aspirated)
    "জ": (2, 2),   # j   (voiced unaspirated)
    "ঝ": (2, 3),   # jh  (voiced aspirated)
    "ঞ": (2, 4),   # ny  (palatal nasal)

    # ------------------------------------------------------------------
    # Class 3 — Retroflex stops (ট-বর্গ)
    # ------------------------------------------------------------------
    "ট": (3, 0),   # T   (voiceless unaspirated retroflex)
    "ঠ": (3, 1),   # Th  (voiceless aspirated retroflex)
    "ড": (3, 2),   # D   (voiced unaspirated retroflex)
    "ঢ": (3, 3),   # Dh  (voiced aspirated retroflex)
    "ণ": (3, 4),   # N   (retroflex nasal)

    # ------------------------------------------------------------------
    # Class 4 — Dental stops (ত-বর্গ)
    # ------------------------------------------------------------------
    "ত": (4, 0),   # t   (voiceless unaspirated dental)
    "থ": (4, 1),   # th  (voiceless aspirated dental)
    "দ": (4, 2),   # d   (voiced unaspirated dental)
    "ধ": (4, 3),   # dh  (voiced aspirated dental)
    "ন": (4, 4),   # n   (dental nasal)

    # ------------------------------------------------------------------
    # Class 5 — Labial stops (প-বর্গ)
    # ------------------------------------------------------------------
    "প": (5, 0),   # p   (voiceless unaspirated)
    "ফ": (5, 1),   # ph  (voiceless aspirated / f)
    "ব": (5, 2),   # b   (voiced unaspirated)
    "ভ": (5, 3),   # bh  (voiced aspirated / v)
    "ম": (5, 4),   # m   (labial nasal)

    # ------------------------------------------------------------------
    # Class 6 — Semivowels, liquids & sibilants (য-বর্গ)
    # ------------------------------------------------------------------
    "য": (6, 0),   # y/j (palatal approximant)
    "র": (6, 1),   # r   (flap/trill)
    "ল": (6, 2),   # l   (lateral)
    "শ": (6, 3),   # sh  (palatal sibilant)
    "ষ": (6, 4),   # Sh  (retroflex sibilant)
    "স": (6, 5),   # s   (dental sibilant)
    "হ": (6, 6),   # h   (glottal fricative)

    # ------------------------------------------------------------------
    # Class 7 — Extra / derived consonants
    # ------------------------------------------------------------------
    "ড়": (7, 0),  # Rh  (voiced retroflex flap — derived from ড)
    "ঢ়": (7, 1),  # Rhh (voiced aspirated retroflex flap — derived from ঢ)
    "য়": (7, 2),  # y   (approximant — derived from য, used in word-medial/final)
    "ৎ": (7, 3),  # t'  (final unaspirated stop — khanda ta)
    "ং": (7, 4),  # ng  (anusvara — final nasal)
    "ঃ": (7, 5),  # h   (visarga — breathy release)
    "ঁ": (7, 6),  # ~   (chandrabindu — nasalisation diacritic)

    # ------------------------------------------------------------------
    # Class 8 — Dependent vowel signs / matras (কার)
    # Matras always follow a consonant; they are NOT independent vowels.
    # ------------------------------------------------------------------
    "া": (8, 0),   # -aa matra  ( া)
    "ি": (8, 1),   # -i  matra  (ি)
    "ী": (8, 2),   # -ii matra  (ী)
    "ু": (8, 3),   # -u  matra  (ু)
    "ূ": (8, 4),   # -uu matra  (ূ)
    "ৃ": (8, 5),   # -ri matra  (ৃ)
    "ে": (8, 6),   # -e  matra  (ে)
    "ৈ": (8, 7),   # -oi matra  (ৈ)
    "ো": (8, 8),   # -o  matra  (ো  = ে + া, precomposed)
    "ৌ": (8, 9),   # -ou matra  (ৌ  = ে + ৌ, precomposed)

    # ------------------------------------------------------------------
    # Class 9 — Bengali numerals and dandas (punctuation)
    # ------------------------------------------------------------------
    "০": (9, 0),   # digit 0
    "১": (9, 1),   # digit 1
    "২": (9, 2),   # digit 2
    "৩": (9, 3),   # digit 3
    "৪": (9, 4),   # digit 4
    "৫": (9, 5),   # digit 5
    "৬": (9, 6),   # digit 6
    "৭": (9, 7),   # digit 7
    "৮": (9, 8),   # digit 8
    "৯": (9, 9),   # digit 9
    "।": (9, 10),  # danda      — Bengali full stop (single vertical bar)
    "॥": (9, 11),  # double danda — section / verse end marker

    # ------------------------------------------------------------------
    # Class 10 — Virama (হসন্ত) and zero-width joiners
    # The virama (্) suppresses the inherent vowel of the preceding consonant
    # and is the foundation of all conjunct consonants (যুক্তাক্ষর).
    # ------------------------------------------------------------------
    "্": (10, 0),  # virama / hasanta (্)
    "\u200c": (10, 1),  # ZWNJ  — forces explicit half-form (non-joining)
    "\u200d": (10, 2),  # ZWJ   — forces conjunct / ligature form
}

# Fallback coordinate for any character not in the table above.
_BENGALI_UNKNOWN: tuple[int, int] = (11, 0)


def get_bengali_coords(char: str) -> tuple[int, int]:
    """Return (phonetic_class, position_in_class) for a Bengali character.

    For characters in the Bengali Unicode block (U+0980–U+09FF) that are not
    in the explicit table (e.g. rare archaic letters), returns (11, 0).
    For ASCII and non-Bengali characters, callers should use the original
    ``phonetic_map.get_coords()`` instead.
    """
    if not char:
        return _BENGALI_UNKNOWN
    return BENGALI_PHONETIC_CLASSES.get(char, _BENGALI_UNKNOWN)


def is_bengali(char: str) -> bool:
    """Return True if *char* is in the Bengali Unicode block (U+0980–U+09FF)."""
    return bool(char) and "\u0980" <= char <= "\u09ff"


def bengali_phonetic_class(char: str) -> int:
    """Return just the phonetic class index (0-11) for a Bengali character."""
    return get_bengali_coords(char)[0]


class BengaliPhoneticMap:
    """Utility wrapper around the Bengali phonetic coordinate system.

    Usage::

        from compression.alphabet.bengali_phonetic_map import BengaliPhoneticMap
        bpm = BengaliPhoneticMap()
        print(bpm.get_coords('ক'))   # (1, 0)
        print(bpm.get_coords('আ'))   # (0, 1)
        print(bpm.is_bengali('a'))   # False
    """

    def __init__(self) -> None:
        self.phonetic_classes = BENGALI_PHONETIC_CLASSES

    def get_coords(self, char: str) -> tuple[int, int]:
        """Coords for *char*; falls back to (11, 0) for unknowns."""
        return get_bengali_coords(char)

    def is_bengali(self, char: str) -> bool:
        """True if *char* is in the Bengali Unicode block."""
        return is_bengali(char)

    def phonetic_class(self, char: str) -> int:
        """Return just the class index."""
        return bengali_phonetic_class(char)

    def class_name(self, char: str) -> str:
        """Human-readable class label for *char*."""
        cls = bengali_phonetic_class(char)
        _NAMES = {
            0:  "vowel (স্বরবর্ণ)",
            1:  "velar stop (ক-বর্গ)",
            2:  "palatal stop (চ-বর্গ)",
            3:  "retroflex stop (ট-বর্গ)",
            4:  "dental stop (ত-বর্গ)",
            5:  "labial stop (প-বর্গ)",
            6:  "semivowel/sibilant (য-বর্গ)",
            7:  "extra consonant",
            8:  "matra (কার)",
            9:  "numeral/punctuation",
            10: "virama/joiner",
            11: "unknown",
        }
        return _NAMES.get(cls, "unknown")
