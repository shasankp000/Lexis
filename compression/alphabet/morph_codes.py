"""Morphological transformation codes and factoriadic helpers."""

from __future__ import annotations

from math import factorial
from typing import Dict, List

try:
    import lemminflect as _lemminflect  # type: ignore[import-not-found]

    _LEMMINFLECT_AVAILABLE = True
except ImportError:
    _lemminflect = None
    _LEMMINFLECT_AVAILABLE = False

MORPH_CODES: Dict[str, int] = {
    "BASE": 0,  # no transformation
    "PLURAL": 1,  # dogs → dog
    "PAST_TENSE": 2,  # walked → walk
    "PRESENT_PART": 3,  # running → run
    "PAST_PART": 4,  # broken → break (irregular past participle)
    "THIRD_SING": 5,  # runs → run
    "COMPARATIVE": 6,  # faster → fast
    "SUPERLATIVE": 7,  # fastest → fast
    "ADVERBIAL": 8,  # quickly → quick
    "NEGATION": 9,  # unhappy → happy
    "AGENT": 10,  # runner → run
    "NOMINALIZE": 11,  # darkness → dark
    "IRREGULAR": 12,  # went → go (stored with separate irregular lookup)
}

MORPH_CODE_NAMES: Dict[int, str] = {v: k for k, v in MORPH_CODES.items()}

BASE = MORPH_CODES["BASE"]
PLURAL = MORPH_CODES["PLURAL"]
PAST_TENSE = MORPH_CODES["PAST_TENSE"]
PRESENT_PART = MORPH_CODES["PRESENT_PART"]
PAST_PART = MORPH_CODES["PAST_PART"]
THIRD_SING = MORPH_CODES["THIRD_SING"]
COMPARATIVE = MORPH_CODES["COMPARATIVE"]
SUPERLATIVE = MORPH_CODES["SUPERLATIVE"]
ADVERBIAL = MORPH_CODES["ADVERBIAL"]
NEGATION = MORPH_CODES["NEGATION"]
AGENT = MORPH_CODES["AGENT"]
NOMINALIZE = MORPH_CODES["NOMINALIZE"]
IRREGULAR = MORPH_CODES["IRREGULAR"]

# Must be after PLURAL, PAST_TENSE etc. are defined
_CODE_TO_PTB: Dict[int, str | None] = {
    PLURAL: "NNS",
    PAST_TENSE: "VBD",
    PRESENT_PART: "VBG",
    PAST_PART: "VBN",
    THIRD_SING: "VBZ",
    COMPARATIVE: "JJR",
    SUPERLATIVE: "JJS",
    ADVERBIAL: "RB",
    NEGATION: None,
    AGENT: None,
    NOMINALIZE: None,
}

_DECODE_OVERRIDES: Dict[str, str] = {
    # be-verb ambiguities (lemminflect returns "was" for all be/VBD)
    "are": "are",
    "were": "were",
    "been": "been",
    # Accusative pronouns
    "me": "me",
    "him": "him",
    "her": "her",
    "us": "us",
    "them": "them",
    # Possessive pronouns
    "my": "my",
    "his": "his",
    "its": "its",
    "our": "our",
    "their": "their",
    "your": "your",
    # Reflexive pronouns
    "myself": "myself",
    "himself": "himself",
    "herself": "herself",
    "itself": "itself",
    "ourselves": "ourselves",
    "themselves": "themselves",
    "yourself": "yourself",
    "yourselves": "yourselves",
}


def encode_morph(code: int) -> List[int]:
    """Encode morph code as factoriadic digits."""
    return _encode_factoradic(code)


def decode_morph(digits: List[int]) -> int:
    """Decode factoriadic digits back to morph code integer."""
    return _decode_factoradic(digits)


def _inflect(root: str, ptb_tag: str) -> str | None:
    """Return the inflected form of root for the given PTB tag, or None."""
    if not _LEMMINFLECT_AVAILABLE or _lemminflect is None:
        return None
    forms = _lemminflect.getInflection(root, tag=ptb_tag)  # type: ignore[attr-defined]
    return forms[0] if forms else None


def apply_morph(root: str, code: int) -> str:
    """Reconstruct surface form from root + morph code (decode direction)."""
    if code == BASE:
        return root

    if code == IRREGULAR:
        if root in _DECODE_OVERRIDES:
            return _DECODE_OVERRIDES[root]
        result = _inflect(root, "VBD")
        return result if result else root

    if code == NEGATION:
        if root.startswith("un"):
            return root
        return f"un{root}"

    if code == AGENT:
        if root.endswith("e"):
            return root + "r"
        return root + "er"

    if code == NOMINALIZE:
        if root.endswith("y"):
            return root[:-1] + "iness"
        if root.endswith("e"):
            return root[:-1] + "eness"
        return root + "ness"

    ptb_tag = _CODE_TO_PTB.get(code)
    if ptb_tag:
        result = _inflect(root, ptb_tag)
        if result:
            return result

    return root


def _encode_factoradic(value: int) -> List[int]:
    if value < 0:
        raise ValueError("value must be non-negative")
    if value == 0:
        return [0]
    digits: List[int] = []
    base = 1
    n = value
    while n > 0:
        n, rem = divmod(n, base)
        digits.append(rem)
        base += 1
    return list(reversed(digits))


def _decode_factoradic(digits: List[int]) -> int:
    if not digits:
        return 0
    total = 0
    length = len(digits)
    for idx, digit in enumerate(digits):
        weight = factorial(length - 1 - idx)
        total += digit * weight
    return total
