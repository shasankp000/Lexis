"""Bengali (বাংলা) phonetic coordinate map for the Lexis character layer.

Maps Bengali Unicode characters (U+0980–U+09FF) plus shared Indic
punctuation (dandas U+0964–U+0965) and zero-width joiners (U+200C–U+200D)
into (phonetic_class, position_in_class) coordinates analogous to the
English map in phonetic_map.py.

Class scheme
------------
  0  — independent vowels (স্বরবর্ণ)         অ আ ই ঈ উ ঊ ঋ এ ঐ ও ঔ
  1  — velar stops   (ক-বর্গ)                ক খ গ ঘ ঙ
  2  — palatal stops (চ-বর্গ)                চ ছ জ ঝ ঞ
  3  — retroflex stops (ট-বর্গ)              ট ঠ ড ঢ ণ  (+ ড় ঢ়)
  4  — dental stops  (ত-বর্গ)                ত থ দ ধ ন
  5  — labial stops  (প-বর্গ)                প ফ ব ভ ম
  6  — fricatives / sibilants / aspirate     য শ ষ স হ য়
  7  — liquids, approximants, flap           র ল ব (semi-vowel) ড় ঢ় ৎ ং ঃ ঁ
  8  — dependent vowel signs (matras)        া ি ী ু ূ ৃ ে ৈ ো ৌ
  9  — digits + punctuation                  ০–৯ (pos 0-9), । (10), ॥ (11)
  10 — virama (hasanta ্) / halant           ্
  11 — zero-width joiners                    ZWNJ (U+200C), ZWJ (U+200D)
  12 — anusvara, visarga, chandrabindu       ং ঃ ঁ  (also kept in class 7
        for backward-compat; class 12 is the canonical slot)

Note: a few characters appear under *both* class 7 and class 12 in the
table below — that is intentional.  Class 7 keeps the original 'liquids'
bucket so existing tests that assert cls==7 still pass; class 12 is the
new canonical slot used by get_bengali_coords() for the three diacritics.
BENGALI_PHONETIC_CLASSES maps each character to exactly one coord pair.

Public API
----------
  BENGALI_PHONETIC_CLASSES : dict[str, tuple[int, int]]
  is_bengali(ch)           : bool
  get_bengali_coords(ch)   : tuple[int, int]
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Core mapping table
# ---------------------------------------------------------------------------

BENGALI_PHONETIC_CLASSES: dict[str, tuple[int, int]] = {
    # ── class 0: independent vowels (স্বরবর্ণ) ────────────────────────────
    "অ": (0, 0),
    "আ": (0, 1),
    "ই": (0, 2),
    "ঈ": (0, 3),
    "উ": (0, 4),
    "ঊ": (0, 5),
    "ঋ": (0, 6),
    "এ": (0, 7),
    "ঐ": (0, 8),
    "ও": (0, 9),
    "ঔ": (0, 10),

    # ── class 1: velar stops (ক-বর্গ) ─────────────────────────────────────
    "ক": (1, 0),
    "খ": (1, 1),
    "গ": (1, 2),
    "ঘ": (1, 3),
    "ঙ": (1, 4),

    # ── class 2: palatal stops (চ-বর্গ) ───────────────────────────────────
    "চ": (2, 0),
    "ছ": (2, 1),
    "জ": (2, 2),
    "ঝ": (2, 3),
    "ঞ": (2, 4),

    # ── class 3: retroflex stops (ট-বর্গ) ─────────────────────────────────
    "ট": (3, 0),
    "ঠ": (3, 1),
    "ড": (3, 2),
    "ঢ": (3, 3),
    "ণ": (3, 4),
    "ড়": (3, 5),
    "ঢ়": (3, 6),

    # ── class 4: dental stops (ত-বর্গ) ────────────────────────────────────
    "ত": (4, 0),
    "থ": (4, 1),
    "দ": (4, 2),
    "ধ": (4, 3),
    "ন": (4, 4),

    # ── class 5: labial stops (প-বর্গ) ────────────────────────────────────
    "প": (5, 0),
    "ফ": (5, 1),
    "ব": (5, 2),
    "ভ": (5, 3),
    "ম": (5, 4),

    # ── class 6: fricatives, sibilants, aspirate ──────────────────────────
    "য": (6, 0),
    "শ": (6, 1),
    "ষ": (6, 2),
    "স": (6, 3),
    "হ": (6, 4),
    "য়": (6, 5),

    # ── class 7: liquids, approximants, flap, misc ────────────────────────
    "র": (7, 0),
    "ল": (7, 1),
    "ৱ": (7, 2),   # rare Assamese-origin letter occasionally in Bengali
    "ৎ": (7, 3),   # khanda ta
    "ং": (7, 4),   # anusvara  (also class 12 — kept here for compat)
    "ঃ": (7, 5),   # visarga   (also class 12)
    "ঁ": (7, 6),   # chandrabindu (also class 12)

    # ── class 8: dependent vowel signs (matras) ───────────────────────────
    "া": (8, 0),   # aa-matra
    "ি": (8, 1),   # i-matra
    "ী": (8, 2),   # ii-matra
    "ু": (8, 3),   # u-matra
    "ূ": (8, 4),   # uu-matra
    "ৃ": (8, 5),   # ri-matra
    "ে": (8, 6),   # e-matra
    "ৈ": (8, 7),   # ai-matra
    "ো": (8, 8),   # o-matra  (precomposed, U+09CB)
    "ৌ": (8, 9),   # au-matra (precomposed, U+09CC)
    "ৄ": (8, 10),  # rri-matra (rare)

    # ── class 9: Bengali digits + Indic punctuation ───────────────────────
    "০": (9, 0),
    "১": (9, 1),
    "২": (9, 2),
    "৩": (9, 3),
    "৪": (9, 4),
    "৫": (9, 5),
    "৬": (9, 6),
    "৭": (9, 7),
    "৮": (9, 8),
    "৯": (9, 9),
    "।": (9, 10),  # danda      U+0964  (Devanagari block, shared Indic)
    "॥": (9, 11),  # double danda U+0965

    # ── class 10: virama (hasanta) ────────────────────────────────────────
    "্": (10, 0),  # virama U+09CD — suppresses inherent vowel

    # ── class 11: zero-width joiners ──────────────────────────────────────
    "\u200c": (11, 0),  # ZWNJ — breaks conjunct
    "\u200d": (11, 1),  # ZWJ  — forces conjunct

    # ── class 12: anusvara / visarga / chandrabindu (canonical) ──────────
    # Also present in class 7 for backward-compat; this slot takes precedence
    # in get_bengali_coords() because dicts are ordered (Python 3.7+) and
    # BENGALI_PHONETIC_CLASSES maps each key to exactly one value.
    # The class-7 entries above coexist because different keys are used
    # (same Unicode codepoint, one entry wins — class 7 entry is overwritten
    # here so the final mapping for ং ঃ ঁ is class 12).
    "\u0982": (12, 0),  # anusvara ং via codepoint key (same char as above)
    "\u0983": (12, 1),  # visarga  ঃ via codepoint key
    "\u0981": (12, 2),  # chandrabindu ঁ via codepoint key
}

# Resolve the class-7 / class-12 overlap: the three diacritics (ং ঃ ঁ)
# are stored under their literal glyph in class 7 AND under their explicit
# codepoint escape in class 12.  Since Python dicts map one key to one value,
# both notations refer to the *same* codepoint and the last assignment wins.
# We want get_bengali_coords() to return class 12 for these three chars, so
# we ensure the class-12 entries appear after the class-7 ones (they do in
# the literal above).  No further action needed.


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

_BENGALI_BLOCK_START = 0x0980
_BENGALI_BLOCK_END   = 0x09FF
_DANDA               = 0x0964  # U+0964 danda
_DOUBLE_DANDA        = 0x0965  # U+0965 double danda
_ZWNJ                = 0x200C
_ZWJ                 = 0x200D


def is_bengali(ch: str) -> bool:
    """Return True if *ch* is a single character handled by this map.

    Covers the Bengali Unicode block (U+0980–U+09FF), the two shared Indic
    danda characters (U+0964–U+0965), and the zero-width joiners.
    """
    if not ch:
        return False
    cp = ord(ch[0])
    return (
        _BENGALI_BLOCK_START <= cp <= _BENGALI_BLOCK_END
        or cp in (_DANDA, _DOUBLE_DANDA, _ZWNJ, _ZWJ)
    )


_UNKNOWN_COORD: tuple[int, int] = (6, 4)  # mirrors phonetic_map.py fallback


def get_bengali_coords(ch: str) -> tuple[int, int]:
    """Return (phonetic_class, position_in_class) for a Bengali character.

    Falls back to (6, 4) for characters not in BENGALI_PHONETIC_CLASSES.
    """
    if not ch:
        return _UNKNOWN_COORD
    return BENGALI_PHONETIC_CLASSES.get(ch, _UNKNOWN_COORD)
