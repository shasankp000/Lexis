"""Stage 1c: Symbol slot substitution for §-prefixed tokens.

§W / §E / §R tokens cannot pass through the phonetic character encoder
because '§' is not in PHONETIC_CLASSES and would be silently dropped.

This stage replaces every §-prefixed token with a short pure-alpha
placeholder zz{base26(n)} that:
  - Is pure-alpha (no digits — spaCy won't strip trailing digits)
  - Never appears in natural English (zz prefix is not a real morpheme)
  - Survives spaCy lemmatisation (lowercased but otherwise unchanged)
  - Is short: zza, zzb, ..., zzz, zzaa, zzab, ... (3-4 chars for n<702)
  - Maps back to the original symbol via slot_map stored in the payload

Slot index encoding (base-26 alpha suffix)
------------------------------------------
  0  -> 'a'    (zza)
  1  -> 'b'    (zzb)
  25 -> 'z'    (zzz)
  26 -> 'aa'   (zzaa)
  27 -> 'ab'   (zzab)
  ...

Encode / decode contract
------------------------
    text_with_slots, slot_map = encode_slots_in_text(text)
    # ... encode_sentences(text_with_slots) ...
    # ... decode back to root list ...
    roots = decode_slots_in_roots(roots, slot_map)
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_SYMBOL_RE = re.compile(r"\u00a7[EWR]\d+")  # §E1, §W3, §R0 …
# Case-insensitive: spaCy lowercases tokens, so 'ZZA' becomes 'zza'
_SLOT_RE   = re.compile(r"\bzz([a-z]+)\b", re.IGNORECASE)
_SLOT_PREFIX = "zz"


# ---------------------------------------------------------------------------
# Base-26 index <-> alpha-string helpers
# ---------------------------------------------------------------------------

def _idx_to_alpha(n: int) -> str:
    """Encode non-negative integer n to a base-26 lowercase alpha string.
    0->'a', 25->'z', 26->'aa', 27->'ab', ...
    """
    result = []
    n += 1  # shift so 0->'a' not empty string
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result.append(chr(ord('a') + rem))
    return ''.join(reversed(result))


def _alpha_to_idx(s: str) -> int:
    """Decode base-26 alpha string back to integer. Inverse of _idx_to_alpha."""
    s = s.lower()
    result = 0
    for ch in s:
        result = result * 26 + (ord(ch) - ord('a') + 1)
    return result - 1


def _make_slot(n: int) -> str:
    """Return the placeholder string for slot index n, e.g. 0->'zza'."""
    return _SLOT_PREFIX + _idx_to_alpha(n)


# ---------------------------------------------------------------------------
# Text-level encode / decode
# ---------------------------------------------------------------------------

def encode_slots_in_text(
    text: str,
) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Replace every §-prefixed token in `text` with zz{alpha(n)}.
    Returns (modified_text, slot_map) where slot_map is a list of
    (slot_index, original_symbol) pairs in left-to-right order.
    """
    slot_map: List[Tuple[int, str]] = []
    slot_idx = 0

    def _replacer(m: re.Match) -> str:
        nonlocal slot_idx
        sym = m.group(0)
        slot_map.append((slot_idx, sym))
        placeholder = _make_slot(slot_idx)
        slot_idx += 1
        return placeholder

    result = _SYMBOL_RE.sub(_replacer, text)
    return result, slot_map


def decode_slots_in_text(
    text: str,
    slot_map: List[Tuple[int, str]],
) -> str:
    """Restore zz{alpha} placeholders in text back to original § symbols."""
    lookup: Dict[int, str] = {idx: sym for idx, sym in slot_map}

    def _replacer(m: re.Match) -> str:
        idx = _alpha_to_idx(m.group(1))
        return lookup.get(idx, m.group(0))

    return _SLOT_RE.sub(_replacer, text)


# ---------------------------------------------------------------------------
# Root-list encode / decode (used by decompressor)
# ---------------------------------------------------------------------------

def encode_slots_in_roots(
    roots: List[str],
) -> Tuple[List[str], List[Tuple[int, str]]]:
    """Replace §-prefixed entries in a root word list with zz{alpha(n)}."""
    slot_map: List[Tuple[int, str]] = []
    new_roots: List[str] = []
    slot_idx = 0
    for root in roots:
        if _SYMBOL_RE.match(root):
            slot_map.append((slot_idx, root))
            new_roots.append(_make_slot(slot_idx))
            slot_idx += 1
        else:
            new_roots.append(root)
    return new_roots, slot_map


def decode_slots_in_roots(
    roots: List[str],
    slot_map: List[Tuple[int, str]],
) -> List[str]:
    """
    Restore zz{alpha} entries in a decoded root list back to § symbols.
    Matches case-insensitively since spaCy lowercases all tokens.
    """
    lookup: Dict[int, str] = {idx: sym for idx, sym in slot_map}
    result: List[str] = []
    for root in roots:
        m = _SLOT_RE.fullmatch(root)
        if m:
            idx = _alpha_to_idx(m.group(1))
            result.append(lookup.get(idx, root))
        else:
            result.append(root)
    return result


# ---------------------------------------------------------------------------
# Payload serialisation
# ---------------------------------------------------------------------------

def pack_slot_map(slot_map: List[Tuple[int, str]]) -> List[List]:
    """Serialise slot_map to a msgpack-friendly list of [index, symbol] pairs."""
    return [[idx, sym] for idx, sym in slot_map]


def unpack_slot_map(packed: List[List]) -> List[Tuple[int, str]]:
    """Deserialise slot_map from msgpack payload."""
    return [(int(item[0]), str(item[1])) for item in packed]
