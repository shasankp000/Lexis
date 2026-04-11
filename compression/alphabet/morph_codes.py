"""Morphological transformation codes and factoriadic helpers."""

from __future__ import annotations

import re
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
    "PLURAL": 1,  # dogs -> dog
    "PAST_TENSE": 2,  # walked -> walk
    "PRESENT_PART": 3,  # running -> run
    "PAST_PART": 4,  # broken -> break (irregular past participle)
    "THIRD_SING": 5,  # runs -> run
    "COMPARATIVE": 6,  # faster -> fast
    "SUPERLATIVE": 7,  # fastest -> fast
    "ADVERBIAL": 8,  # quickly -> quick
    "NEGATION": 9,  # unhappy -> happy
    "AGENT": 10,  # runner -> run
    "NOMINALIZE": 11,  # darkness -> dark
    "IRREGULAR": 12,  # went -> go (stored with separate irregular lookup)
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
    COMPARATIVE: None,  # lemminflect JJR is broken for irregulars
    SUPERLATIVE: None,  # lemminflect JJS is broken for irregulars
    ADVERBIAL: None,  # handled manually below, lemminflect doesn't do derivation
    NEGATION: None,
    AGENT: None,
    NOMINALIZE: None,
}

# Safety-net overrides for the IRREGULAR decode path.
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
    # Irregular comparatives / superlatives and other identity-mapped words.
    "most": "most",
    "best": "best",
    "worst": "worst",
    "least": "least",
    "more": "more",
    "better": "better",
    "worse": "worse",
    "less": "less",
    "further": "further",
    "furthest": "furthest",
    "many": "many",
    "much": "much",
}

# Regex for the CVC (consonant-vowel-consonant) doubling pattern.
# Matches a root that ends in: one consonant, one vowel, one consonant
# where the final consonant is NOT w, x, or y (standard English rule).
_VOWELS = "aeiou"
_CVC_RE = re.compile(
    r"[^aeiou][aeiou][^aeiouwxy]$",
    re.IGNORECASE,
)


def _double_final_consonant(root: str) -> str:
    """Double the final consonant of *root* if it matches the CVC pattern.

    Examples:
        run  -> runn  (then caller appends 'er' -> runner)
        swim -> swimm
        bat  -> batt
        read -> read  (ends in vowel-consonant but preceded by vowel: no double)
        eat  -> eat   (two vowels before final t)
    """
    if len(root) >= 3 and _CVC_RE.search(root):
        return root + root[-1]
    return root


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
        # Apply CVC consonant-doubling before appending the agentive suffix.
        # Examples: run -> runn+er = runner, write -> writ+er = writer (e-drop
        # is already absent in the root), swim -> swimm+er = swimmer.
        doubled = _double_final_consonant(root)
        if root.endswith("e"):
            # root already ends in -e: just append -r (e.g. "write" -> "writer")
            return root + "r"
        return doubled + "er"

    if code == NOMINALIZE:
        if root.endswith("y"):
            return root[:-1] + "iness"
        if root.endswith("e"):
            return root[:-1] + "eness"
        return root + "ness"

    if code == ADVERBIAL:
        if root.endswith("ly"):
            return root
        if root.endswith("y") and len(root) > 1:
            return root[:-1] + "ily"
        if root.endswith("le") and len(root) > 2:
            return root[:-2] + "ly"
        return root + "ly"

    if code == COMPARATIVE:
        _irreg_cmp = {
            "good": "better",
            "bad": "worse",
            "far": "further",
            "little": "less",
            "many": "more",
            "much": "more",
            "well": "better",
        }
        if root in _irreg_cmp:
            return _irreg_cmp[root]
        if root.endswith("y") and len(root) > 1:
            return root[:-1] + "ier"
        if root.endswith("e"):
            return root + "r"
        return root + "er"

    if code == SUPERLATIVE:
        _irreg_sup = {
            "good": "best",
            "bad": "worst",
            "far": "furthest",
            "little": "least",
            "many": "most",
            "much": "most",
            "well": "best",
        }
        if root in _irreg_sup:
            return _irreg_sup[root]
        if root.endswith("y") and len(root) > 1:
            return root[:-1] + "iest"
        if root.endswith("e"):
            return root + "st"
        return root + "est"

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
