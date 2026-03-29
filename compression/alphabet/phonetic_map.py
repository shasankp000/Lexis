"""Fixed phonetic coordinate mapping helpers for the character layer."""

from __future__ import annotations

from math import sqrt

PHONETIC_CLASSES: dict[str, tuple[int, int]] = {
    # class 0 — vowels
    "a": (0, 0),
    "e": (0, 1),
    "i": (0, 2),
    "o": (0, 3),
    "u": (0, 4),
    # class 1 — stops (complete air blockage + release)
    "b": (1, 0),
    "d": (1, 1),
    "g": (1, 2),
    "k": (1, 3),
    "p": (1, 4),
    "t": (1, 5),
    # class 2 — fricatives (partial air blockage, hissing/buzzing)
    "f": (2, 0),
    "h": (2, 1),
    "s": (2, 2),
    "v": (2, 3),
    "x": (2, 4),
    "z": (2, 5),
    # class 3 — nasals (air through nose)
    "m": (3, 0),
    "n": (3, 1),
    # class 4 — liquids (smooth flowing sounds)
    "l": (4, 0),
    "r": (4, 1),
    # class 5 — other consonants
    "c": (5, 0),
    "j": (5, 1),
    "q": (5, 2),
    "w": (5, 3),
    "y": (5, 4),
    # class 6 — pipeline-only stream markers (^ and $ are the only ones
    # ever written into the char stream; _ is now class-5 punctuation)
    "^": (6, 0),  # word start marker
    "$": (6, 1),  # word end marker
    # (6, 2) was formerly '_' whitespace sentinel — now unused / reserved
}

_PUNCTUATION_CHARS = [
    ",",
    ";",
    ":",
    "!",
    "?",
    "-",
    "'",
    '"',
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    "/",
    "\\",
    "@",
    "#",
    "$",
    "%",
    "&",
    "*",
    "+",
    "=",
    "<",
    ">",
    "~",
    "|",
    "`",
    "_",   # moved here from class-6 sentinel so it round-trips as a real char
]

_DIGIT_CHARS = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

_punct_start = max(pos for ch, (cls, pos) in PHONETIC_CLASSES.items() if cls == 5) + 1
for offset, ch in enumerate(_PUNCTUATION_CHARS):
    if ch not in PHONETIC_CLASSES:
        PHONETIC_CLASSES[ch] = (5, _punct_start + offset)

_digit_start = _punct_start + len(_PUNCTUATION_CHARS)
for offset, ch in enumerate(_DIGIT_CHARS):
    if ch not in PHONETIC_CLASSES:
        PHONETIC_CLASSES[ch] = (5, _digit_start + offset)

MORPH_ROLES: dict[str, int] = {
    "start": 0,  # first character of a word (follows ^)
    "middle": 1,  # interior character
    "end": 2,  # last character of a word (before $)
    "standalone": 3,  # single-character word
}


def get_coords(char: str) -> tuple[int, int]:
    """Return (phonetic_class, position_in_class) for a character.

    Falls back to (6, 4) for unknown characters.
    """
    if not char:
        return (6, 4)
    key = char.lower()
    return PHONETIC_CLASSES.get(key, (6, 4))


def get_morph_role(position: int, word_length: int) -> int:
    """Return morphological role (0-3) for a character at position in a word."""
    if word_length <= 0:
        raise ValueError("word_length must be positive.")
    if word_length == 1:
        return MORPH_ROLES["standalone"]
    if position <= 0:
        return MORPH_ROLES["start"]
    if position >= word_length - 1:
        return MORPH_ROLES["end"]
    return MORPH_ROLES["middle"]


def char_to_triple(char: str, position: int, word_length: int) -> tuple[int, int, int]:
    """Return full (class, pos, role) triple for a character."""
    phon_class, pos_in_class = get_coords(char)
    role = get_morph_role(position, word_length)
    return (phon_class, pos_in_class, role)


def compute_deltas(
    triples: list[tuple[int, int, int]],
) -> tuple[list[int], list[int], list[int]]:
    """Given a list of (class, pos, role) triples, return three delta lists.

    The first value of each list is stored absolute. The rest are differences.
    """
    if not triples:
        return ([], [], [])
    class_deltas: list[int] = [triples[0][0]]
    pos_deltas: list[int] = [triples[0][1]]
    role_deltas: list[int] = [triples[0][2]]
    for prev, current in zip(triples, triples[1:]):
        class_deltas.append(current[0] - prev[0])
        pos_deltas.append(current[1] - prev[1])
        role_deltas.append(current[2] - prev[2])
    return (class_deltas, pos_deltas, role_deltas)


def delta_magnitude_2d(dc: int, dp: int) -> float:
    """Euclidean magnitude of a 2D (class, position) delta."""
    return float(sqrt(dc * dc + dp * dp))


class PhoneticMap:
    """Utility wrapper around the fixed phonetic coordinate system."""

    def __init__(self) -> None:
        self.phonetic_classes = PHONETIC_CLASSES
        self.morph_roles = MORPH_ROLES

    def get_coords(self, char: str) -> tuple[int, int]:
        return get_coords(char)

    def get_morph_role(self, position: int, word_length: int) -> int:
        return get_morph_role(position, word_length)

    def char_to_triple(
        self, char: str, position: int, word_length: int
    ) -> tuple[int, int, int]:
        return char_to_triple(char, position, word_length)

    def compute_deltas(
        self, triples: list[tuple[int, int, int]]
    ) -> tuple[list[int], list[int], list[int]]:
        return compute_deltas(triples)

    def delta_magnitude_2d(self, dc: int, dp: int) -> float:
        return delta_magnitude_2d(dc, dp)
