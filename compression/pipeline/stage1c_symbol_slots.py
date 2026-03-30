"""Stage 1c: Symbol slot substitution for §-prefixed tokens.

§W / §E / §R tokens cannot pass through the phonetic character encoder
because '§' is not in PHONETIC_CLASSES and would be silently dropped.

This stage replaces every §-prefixed token in the sentence-level word
list with a deterministic placeholder ZSLOT{n} that:
  - Is pure-alpha (spaCy tokenises it as one token)
  - Never appears in natural English text
  - Has a known, short root length
  - Maps back to the original symbol via slot_map stored in the payload

The slot_map is a list of (token_index, symbol) pairs in document order,
stored as a compact list in the msgpack payload.

Encode / decode contract
------------------------
    text_with_slots, slot_map = encode_slots(text)
    # ... encode_sentences(text_with_slots) ...
    # ... decode back to word list ...
    words = decode_slots(words, slot_map)

The text-level encode/decode is also exposed for the compress pipeline:
    text_with_slots, slot_map = encode_slots_in_text(text)
    text_restored  = decode_slots_in_text(text_with_slots, slot_map)
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_SYMBOL_RE = re.compile(r"\u00a7[EWR]\d+")  # matches §E1, §W3, §R0 etc.
_SLOT_RE   = re.compile(r"\bZSLOT(\d+)\b")

# Placeholder root used in the char stream. Must be:
#   - pure alpha (no digits in the root itself, digits in the suffix)
#   - never appear in natural English
_SLOT_PREFIX = "ZSLOT"


def encode_slots_in_text(
    text: str,
) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Replace every §-prefixed token in `text` with ZSLOT{n}.
    Returns (modified_text, slot_map) where slot_map is a list of
    (slot_index, original_symbol) pairs in left-to-right order.
    """
    slot_map: List[Tuple[int, str]] = []
    result = text
    # Find all symbol occurrences left to right, replace with slot placeholder
    # Must use re.sub with a counter to assign sequential slot indices.
    slot_idx = 0

    def _replacer(m: re.Match) -> str:
        nonlocal slot_idx
        sym = m.group(0)
        slot_map.append((slot_idx, sym))
        placeholder = f"{_SLOT_PREFIX}{slot_idx}"
        slot_idx += 1
        return placeholder

    result = _SYMBOL_RE.sub(_replacer, result)
    return result, slot_map


def decode_slots_in_text(
    text: str,
    slot_map: List[Tuple[int, str]],
) -> str:
    """Restore ZSLOT{n} placeholders in text back to original § symbols."""
    lookup: Dict[int, str] = {idx: sym for idx, sym in slot_map}

    def _replacer(m: re.Match) -> str:
        idx = int(m.group(1))
        return lookup.get(idx, m.group(0))

    return _SLOT_RE.sub(_replacer, text)


def encode_slots_in_roots(
    roots: List[str],
) -> Tuple[List[str], List[Tuple[int, str]]]:
    """
    Replace §-prefixed entries in a root word list with ZSLOT{n}.
    Returns (modified_roots, slot_map).
    Used by the compressor after morphological analysis.
    """
    slot_map: List[Tuple[int, str]] = []
    new_roots: List[str] = []
    slot_idx = 0
    for root in roots:
        if _SYMBOL_RE.match(root):
            slot_map.append((slot_idx, root))
            new_roots.append(f"{_SLOT_PREFIX}{slot_idx}")
            slot_idx += 1
        else:
            new_roots.append(root)
    return new_roots, slot_map


def decode_slots_in_roots(
    roots: List[str],
    slot_map: List[Tuple[int, str]],
) -> List[str]:
    """
    Restore ZSLOT{n} entries in a decoded root list back to § symbols.
    """
    lookup: Dict[int, str] = {idx: sym for idx, sym in slot_map}
    result: List[str] = []
    for root in roots:
        m = _SLOT_RE.fullmatch(root)
        if m:
            idx = int(m.group(1))
            result.append(lookup.get(idx, root))
        else:
            result.append(root)
    return result


def pack_slot_map(slot_map: List[Tuple[int, str]]) -> List[List]:
    """Serialise slot_map to a msgpack-friendly list of [index, symbol] pairs."""
    return [[idx, sym] for idx, sym in slot_map]


def unpack_slot_map(packed: List[List]) -> List[Tuple[int, str]]:
    """Deserialise slot_map from msgpack payload."""
    return [(int(item[0]), str(item[1])) for item in packed]
