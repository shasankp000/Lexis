"""Stage 1c: Symbol slot extraction for §-prefixed tokens.

§W / §E / §R tokens cannot pass through the phonetic character encoder
because '§' is not in PHONETIC_CLASSES and would be silently dropped.

Approach
--------
Compressor: strip all §-tokens from text before encoding, recording
(char_offset, symbol) for each, where char_offset is the character
position in the CLEAN text immediately before which the symbol appeared.

Decompressor: after _join_words produces the final string, re-splice
§-symbols at their saved positions using a char-offset mapping from the
original clean text to the joined result. This is purely a string
operation, fully decoupled from spaCy tokenisation.

Contract
--------
    clean_text, slot_map = extract_symbols(text)
    # ... encode clean_text, decode back to joined_text ...
    result = splice_symbols(joined_text, slot_map)
    result = decode_symbols(result, symbol_table)  # restores originals

slot_map format: list of (char_offset, symbol) pairs where char_offset
is the byte position in clean_text before which the symbol was extracted.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_SYMBOL_RE   = re.compile(r"\u00a7[EWR]\d+")   # §E1, §W3, §R0 …
# Legacy placeholder support
_SLOT_RE     = re.compile(r"\bzz([a-z]+)\b", re.IGNORECASE)
_SLOT_PREFIX = "zz"


# ---------------------------------------------------------------------------
# Base-26 helpers (legacy placeholder fallback)
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
# PRIMARY API: char-offset extraction + splice
# ---------------------------------------------------------------------------

def extract_symbols(
    text: str,
) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Strip all §-prefixed tokens from `text`, recording char offsets.

    Scans left-to-right with a regex. For each §-token found:
      - Records (char_offset_in_clean_output, symbol)
        where char_offset is the position in the clean output string
        immediately before which the symbol appeared (after removing
        the symbol and any adjacent extra whitespace).
      - Removes the symbol from the output.

    Surrounding whitespace handling:
      - If symbol is preceded and followed by spaces: collapse to one space.
      - If symbol is at start/end: remove adjacent space only.

    Returns
    -------
    clean_text : str
        Text with all § tokens removed and whitespace normalised.
    slot_map : List[Tuple[int, str]]
        [(char_offset, symbol), ...] in document order, where
        char_offset is the position in clean_text before which the
        symbol should be re-inserted on decode.
    """
    slot_map: List[Tuple[int, str]] = []
    # Build clean text and offset map simultaneously
    result_chars: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        m = _SYMBOL_RE.match(text, i)
        if m:
            sym = m.group(0)
            # char_offset in clean output = current length of result_chars
            # but we want the offset AFTER any preceding space is kept
            char_offset = len(result_chars)
            # consume trailing space if present (we collapsed it)
            j = m.end()
            if j < n and text[j] == ' ' and char_offset > 0 and result_chars[-1] == ' ':
                j += 1  # drop the extra space
            slot_map.append((char_offset, sym))
            i = j
        else:
            result_chars.append(text[i])
            i += 1

    clean_text = ''.join(result_chars)
    return clean_text, slot_map


def splice_symbols(
    joined: str,
    slot_map: List[Tuple[int, str]],
    clean_text: str,
) -> str:
    """
    Re-insert § symbols into `joined` at positions corresponding to
    their original offsets in `clean_text`.

    Maps each char_offset in clean_text to the equivalent position in
    joined via linear scaling (robust to minor length drift from
    _join_words punctuation collapsing).

    Inserts in reverse offset order so earlier insertions don't shift
    later ones.

    Parameters
    ----------
    joined : str
        Final text produced by _join_words + capitalisation.
    slot_map : List[Tuple[int, str]]
        [(char_offset_in_clean, symbol), ...] from extract_symbols.
    clean_text : str
        The clean text passed to the encoder (used as reference for scaling).

    Returns
    -------
    str with § symbols spliced in.
    """
    if not slot_map or not clean_text:
        return joined

    clean_len  = len(clean_text)
    joined_len = len(joined)
    scale      = joined_len / clean_len if clean_len else 1.0

    # Map char offsets to joined positions
    mapped: List[Tuple[int, str]] = []
    for clean_off, sym in slot_map:
        j_pos = min(int(round(clean_off * scale)), joined_len)
        # Snap to nearest space boundary for clean insertion
        # Search left then right for a space within a small window
        window = max(1, int(len(sym) * 2))
        best   = j_pos
        for delta in range(window + 1):
            for sign in (-1, 1):
                candidate = j_pos + sign * delta
                if 0 <= candidate <= joined_len:
                    if candidate == 0 or candidate == joined_len or joined[candidate - 1] == ' ' or (candidate < joined_len and joined[candidate] == ' '):
                        best = candidate
                        break
            else:
                continue
            break
        mapped.append((best, sym))

    # Insert in reverse position order
    result = list(joined)
    for j_pos, sym in sorted(mapped, key=lambda x: -x[0]):
        # Insert " sym " with smart space handling
        pre  = result[j_pos - 1] if j_pos > 0 else ''
        post = result[j_pos]     if j_pos < len(result) else ''
        if pre == ' ' and post == ' ':
            insert = list(sym)
        elif pre == ' ' or pre == '':
            insert = list(sym + ' ')
        elif post == ' ' or post == '':
            insert = list(' ' + sym)
        else:
            insert = list(' ' + sym + ' ')
        result[j_pos:j_pos] = insert

    return ''.join(result)


# ---------------------------------------------------------------------------
# Payload serialisation
# ---------------------------------------------------------------------------

def pack_slot_map(slot_map: List[Tuple[int, str]]) -> List[List]:
    """Serialise slot_map to a msgpack-friendly list."""
    return [[off, sym] for off, sym in slot_map]


def unpack_slot_map(packed: List[List]) -> List[Tuple[int, str]]:
    """Deserialise slot_map from msgpack payload."""
    return [(int(item[0]), str(item[1])) for item in packed]


# ---------------------------------------------------------------------------
# LEGACY: placeholder-based encode/decode
# ---------------------------------------------------------------------------

def encode_slots_in_text(
    text: str,
) -> Tuple[str, List[Tuple[int, str]]]:
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


def decode_slots_in_text(text: str, slot_map: List[Tuple[int, str]]) -> str:
    lookup: Dict[int, str] = {idx: sym for idx, sym in slot_map}

    def _replacer(m: re.Match) -> str:
        idx = _alpha_to_idx(m.group(1))
        return lookup.get(idx, m.group(0))

    return _SLOT_RE.sub(_replacer, text)


def encode_slots_in_roots(roots: List[str]) -> Tuple[List[str], List[Tuple[int, str]]]:
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


def decode_slots_in_roots(roots: List[str], slot_map: List[Tuple[int, str]]) -> List[str]:
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
