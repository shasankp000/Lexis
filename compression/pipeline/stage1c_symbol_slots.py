"""Stage 1c: Symbol slot extraction for §-prefixed tokens.

§W / §E / §R tokens cannot pass through the phonetic character encoder
because '§' is not in PHONETIC_CLASSES and would be silently dropped.

Approach: extract all §-tokens from the text BEFORE encoding, recording
their word positions, then re-inject them into the decoded root list in
the decompressor. Zero placeholder chars enter the char stream.

Extract / inject contract
-------------------------
    clean_text, slot_map = extract_symbols(text)
    # encode clean_text through char encoder ...
    # decode back to root list ...
    roots = inject_symbols(roots, slot_map)
    # then _join_words(roots) -> text with § tokens in correct positions
    # then decode_symbols restores originals

slot_map format: list of (word_index, symbol) pairs where word_index is
the position in the token list AT WHICH the symbol should be re-inserted
(i.e. before the token currently at that position in the clean list).

Also exposes the older placeholder-based helpers (encode_slots_in_text,
decode_slots_in_text, etc.) for backwards compatibility.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_SYMBOL_RE  = re.compile(r"\u00a7[EWR]\d+")   # §E1, §W3, §R0 …
_SLOT_RE    = re.compile(r"\bzz([a-z]+)\b", re.IGNORECASE)
_SLOT_PREFIX = "zz"


# ---------------------------------------------------------------------------
# Base-26 helpers (kept for placeholder fallback)
# ---------------------------------------------------------------------------

def _idx_to_alpha(n: int) -> str:
    result = []
    n += 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        result.append(chr(ord('a') + rem))
    return ''.join(reversed(result))


def _alpha_to_idx(s: str) -> int:
    s = s.lower()
    result = 0
    for ch in s:
        result = result * 26 + (ord(ch) - ord('a') + 1)
    return result - 1


def _make_slot(n: int) -> str:
    return _SLOT_PREFIX + _idx_to_alpha(n)


# ---------------------------------------------------------------------------
# PRIMARY API: zero-overhead positional extraction
# ---------------------------------------------------------------------------

def extract_symbols(
    text: str,
) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Strip all §-prefixed tokens from `text`, recording their positions.

    Splits text on whitespace, identifies §-tokens, removes them, and
    records (insertion_index, symbol) for each one where insertion_index
    is the index in the REMAINING (clean) token list before which the
    symbol should be re-inserted on decode.

    Returns
    -------
    clean_text : str
        Whitespace-joined text with all § tokens removed.
    slot_map : List[Tuple[int, str]]
        [(insertion_index, symbol), ...] in document order.
    """
    tokens = text.split(' ')
    slot_map: List[Tuple[int, str]] = []
    clean_tokens: List[str] = []

    for tok in tokens:
        if _SYMBOL_RE.fullmatch(tok):
            # insertion_index = where in clean_tokens this symbol goes
            # (i.e. before the next clean token, = current length)
            slot_map.append((len(clean_tokens), tok))
        else:
            clean_tokens.append(tok)

    return ' '.join(clean_tokens), slot_map


def inject_symbols(
    roots: List[str],
    slot_map: List[Tuple[int, str]],
) -> List[str]:
    """
    Re-insert § symbols into a decoded root list at their saved positions.

    slot_map entries are processed in reverse insertion-index order so
    that earlier insertions don't shift the indices of later ones.

    Parameters
    ----------
    roots : List[str]
        Decoded root word list (output of _split_roots).
    slot_map : List[Tuple[int, str]]
        [(insertion_index, symbol), ...] as produced by extract_symbols.

    Returns
    -------
    List[str] with § symbols re-inserted at the correct positions.
    """
    if not slot_map:
        return roots

    result = list(roots)
    # Insert from highest index to lowest so earlier indices stay valid
    for ins_idx, sym in sorted(slot_map, key=lambda x: -x[0]):
        ins_idx = min(ins_idx, len(result))  # clamp to end if list is short
        result.insert(ins_idx, sym)
    return result


# ---------------------------------------------------------------------------
# Payload serialisation (shared by both approaches)
# ---------------------------------------------------------------------------

def pack_slot_map(slot_map: List[Tuple[int, str]]) -> List[List]:
    """Serialise slot_map to a msgpack-friendly list of [index, symbol] pairs."""
    return [[idx, sym] for idx, sym in slot_map]


def unpack_slot_map(packed: List[List]) -> List[Tuple[int, str]]:
    """Deserialise slot_map from msgpack payload."""
    return [(int(item[0]), str(item[1])) for item in packed]


# ---------------------------------------------------------------------------
# LEGACY: placeholder-based encode/decode (kept for compatibility)
# ---------------------------------------------------------------------------

def encode_slots_in_text(
    text: str,
) -> Tuple[str, List[Tuple[int, str]]]:
    """Replace every §-prefixed token in `text` with zz{alpha(n)} placeholder."""
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
    """Restore zz{alpha} entries in a decoded root list back to § symbols."""
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
