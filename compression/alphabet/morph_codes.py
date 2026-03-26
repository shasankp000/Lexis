"""Morphological transformation codes and factoriadic helpers."""

from __future__ import annotations

from math import factorial
from typing import Dict, List

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

_IRREGULAR_FORMS: Dict[str, str] = {
    "be": "was",
    "begin": "began",
    "break": "broke",
    "come": "came",
    "do": "did",
    "drive": "drove",
    "eat": "ate",
    "get": "got",
    "give": "gave",
    "go": "went",
    "have": "had",
    "make": "made",
    "run": "ran",
    "say": "said",
    "see": "saw",
    "sit": "sat",
    "speak": "spoke",
    "take": "took",
    "write": "wrote",
}


def encode_morph(code: int) -> List[int]:
    """Encode morph code as factoriadic digits."""
    return _encode_factoradic(code)


def decode_morph(digits: List[int]) -> int:
    """Decode factoriadic digits back to morph code integer."""
    return _decode_factoradic(digits)


def apply_morph(root: str, code: int) -> str:
    """Reconstruct surface form from root + morph code.

    This is the DECODE direction — used in Stage 8.
    """
    if code == BASE:
        return root
    if code == IRREGULAR:
        return _IRREGULAR_FORMS.get(root, root)
    if code == NEGATION:
        return f"un{root}"
    if code == PLURAL:
        return _pluralize(root)
    if code == THIRD_SING:
        return _pluralize(root)
    if code == PAST_TENSE:
        return _past_tense(root)
    if code == PAST_PART:
        return _past_tense(root)
    if code == PRESENT_PART:
        return _present_participle(root)
    if code == COMPARATIVE:
        return _comparative(root)
    if code == SUPERLATIVE:
        return _superlative(root)
    if code == ADVERBIAL:
        return _adverbial(root)
    if code == AGENT:
        return _agentive(root)
    if code == NOMINALIZE:
        return _nominalize(root)
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


def _is_vowel(char: str) -> bool:
    return char.lower() in {"a", "e", "i", "o", "u"}


def _is_consonant(char: str) -> bool:
    return char.isalpha() and not _is_vowel(char)


def _ends_with_cvc(word: str) -> bool:
    if len(word) < 3:
        return False
    c1, c2, c3 = word[-3], word[-2], word[-1]
    return (
        _is_consonant(c1)
        and _is_vowel(c2)
        and _is_consonant(c3)
        and c3.lower() not in {"w", "x", "y"}
    )


def _pluralize(root: str) -> str:
    lower = root.lower()
    if lower.endswith(("s", "x", "z", "ch", "sh")):
        return root + "es"
    if lower.endswith("y") and len(root) > 1 and _is_consonant(root[-2]):
        return root[:-1] + "ies"
    return root + "s"


def _past_tense(root: str) -> str:
    lower = root.lower()
    if lower.endswith("e"):
        return root + "d"
    if lower.endswith("y") and len(root) > 1 and _is_consonant(root[-2]):
        return root[:-1] + "ied"
    if _ends_with_cvc(root):
        return root + root[-1] + "ed"
    return root + "ed"


def _present_participle(root: str) -> str:
    lower = root.lower()
    if lower.endswith("ie"):
        return root[:-2] + "ying"
    if lower.endswith("e") and not lower.endswith(("ee", "oe", "ye")):
        return root[:-1] + "ing"
    if _ends_with_cvc(root):
        return root + root[-1] + "ing"
    return root + "ing"


def _comparative(root: str) -> str:
    lower = root.lower()
    if lower.endswith("y") and len(root) > 1 and _is_consonant(root[-2]):
        return root[:-1] + "ier"
    if lower.endswith("e"):
        return root + "r"
    if _ends_with_cvc(root):
        return root + root[-1] + "er"
    return root + "er"


def _superlative(root: str) -> str:
    lower = root.lower()
    if lower.endswith("y") and len(root) > 1 and _is_consonant(root[-2]):
        return root[:-1] + "iest"
    if lower.endswith("e"):
        return root + "st"
    if _ends_with_cvc(root):
        return root + root[-1] + "est"
    return root + "est"


def _adverbial(root: str) -> str:
    lower = root.lower()
    if lower.endswith("y") and len(root) > 1 and _is_consonant(root[-2]):
        return root[:-1] + "ily"
    if lower.endswith("le") and len(root) > 2 and _is_consonant(root[-3]):
        return root[:-2] + "ly"
    return root + "ly"


def _agentive(root: str) -> str:
    return _comparative(root)


def _nominalize(root: str) -> str:
    lower = root.lower()
    if lower.endswith("y") and len(root) > 1 and _is_consonant(root[-2]):
        return root[:-1] + "iness"
    if lower.endswith("e"):
        return root[:-1] + "eness"
    return root + "ness"
