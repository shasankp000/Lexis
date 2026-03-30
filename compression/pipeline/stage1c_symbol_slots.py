"""Stage 1c: Symbol slot extraction for §-prefixed tokens.

§W / §E / §R tokens cannot pass through the phonetic character encoder
because '§' is not in PHONETIC_CLASSES and would be silently dropped.

Approach
--------
Compressor: strip all §-tokens from text before encoding, recording
(char_offset, symbol) for each, where char_offset is the character
position in the CLEAN text immediately before which the symbol appeared.
Only the length of clean_text is stored in the payload (not the text
itself) to avoid bloating the file.

Decompressor: after _join_words produces the final string, re-splice
§-symbols at their saved positions using a char-offset mapping from the
original clean text length to the joined result length. This is purely
a string operation, fully decoupled from spaCy tokenisation.

Contract
--------
    clean_text, slot_map = extract_symbols(text)
    clean_len = len(clean_text)
    # ... encode clean_text, decode back to joined_text ...
    result = splice_symbols(joined_text, slot_map, clean_len)
    result = decode_symbols(result, symbol_table)  # restores originals

slot_map format: list of (char_offset, symbol) pairs where char_offset
is the byte position in clean_text before which the symbol was extracted.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_SYMBOL_RE   = re.compile(r"\u00a7[EWR]\d+")   # §E1, §W3, §R0 …
_SLOT_RE     = re.compile(r"\bzz([a-z]+)\b", re.IGNORECASE)
_SLOT_PREFIX = "zz"


# ---------------------------------------------------------------------------
# Base-26 helpers (legacy)
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
# PRIMARY API
# ---------------------------------------------------------------------------

def extract_symbols(
    text: str,
) -> Tuple[str, List[Tuple[int, str]]]:
    """
    Strip all §-prefixed tokens from `text`, recording char offsets.

    Returns
    -------
    clean_text : str
        Text with all § tokens removed.
    slot_map : List[Tuple[int, str]]
        [(char_offset, symbol), ...] in document order, where
        char_offset is the position in clean_text before which the
        symbol should be re-inserted on decode.
    """
    slot_map: List[Tuple[int, str]] = []
    result_chars: List[str] = []
    i = 0
    n = len(text)

    while i < n:
        m = _SYMBOL_RE.match(text, i)
        if m:
            sym = m.group(0)
            char_offset = len(result_chars)
            j = m.end()
            # collapse double-space left by removal
            if j < n and text[j] == ' ' and char_offset > 0 and result_chars[-1] == ' ':
                j += 1
            slot_map.append((char_offset, sym))
            i = j
        else:
            result_chars.append(text[i])
            i += 1

    return ''.join(result_chars), slot_map


def splice_symbols(
    joined: str,
    slot_map: List[Tuple[int, str]],
    clean_len: int,
) -> str:
    """
    Re-insert § symbols into `joined` at positions corresponding to
    their original offsets in the clean text.

    Maps each char_offset in [0, clean_len] to the equivalent position
    in joined via linear scaling.  Snaps to nearest space boundary for
    clean insertion.

    Parameters
    ----------
    joined    : final text from _join_words + capitalisation.
    slot_map  : [(char_offset_in_clean, symbol), ...] from extract_symbols.
    clean_len : len(clean_text) at compress time.
    """
    if not slot_map or not clean_len:
        return joined

    joined_len = len(joined)
    scale      = joined_len / clean_len

    mapped: List[Tuple[int, str]] = []
    for clean_off, sym in slot_map:
        j_pos  = min(int(round(clean_off * scale)), joined_len)
        window = max(1, len(sym) * 2)
        best   = j_pos
        for delta in range(window + 1):
            for sign in (-1, 1):
                c = j_pos + sign * delta
                if 0 <= c <= joined_len:
                    prev = joined[c - 1] if c > 0 else ''
                    nxt  = joined[c]     if c < joined_len else ''
                    if prev == ' ' or prev == '' or nxt == ' ' or nxt == '':
                        best = c
                        break
            else:
                continue
            break
        mapped.append((best, sym))

    result = list(joined)
    for j_pos, sym in sorted(mapped, key=lambda x: -x[0]):
        pre  = result[j_pos - 1] if j_pos > 0          else ''
        post = result[j_pos]     if j_pos < len(result) else ''
        if pre == ' ' and post == ' ':
            insert = list(sym)
        elif pre in (' ', ''):
            insert = list(sym + ' ')
        elif post in (' ', ''):
            insert = list(' ' + sym)
        else:
            insert = list(' ' + sym + ' ')
        result[j_pos:j_pos] = insert

    return ''.join(result)


# ---------------------------------------------------------------------------
# Payload serialisation
# ---------------------------------------------------------------------------

def pack_slot_map(slot_map: List[Tuple[int, str]]) -> List[List]:
    return [[off, sym] for off, sym in slot_map]


def unpack_slot_map(packed: List[List]) -> List[Tuple[int, str]]:
    return [(int(item[0]), str(item[1])) for item in packed]


# ---------------------------------------------------------------------------
# LEGACY placeholder-based helpers
# ---------------------------------------------------------------------------

def encode_slots_in_text(text: str) -> Tuple[str, List[Tuple[int, str]]]:
    slot_map: List[Tuple[int, str]] = []
    slot_idx = 0

    def _replacer(m: re.Match) -> str:
        nonlocal slot_idx
        sym = m.group(0)
        slot_map.append((slot_idx, sym))
        placeholder = _make_slot(slot_idx)
        slot_idx += 1
        return placeholder

    return _SYMBOL_RE.sub(_replacer, text), slot_map


def decode_slots_in_text(text: str, slot_map: List[Tuple[int, str]]) -> str:
    lookup: Dict[int, str] = {idx: sym for idx, sym in slot_map}

    def _replacer(m: re.Match) -> str:
        return lookup.get(_alpha_to_idx(m.group(1)), m.group(0))

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
            result.append(lookup.get(_alpha_to_idx(m.group(1)), root))
        else:
            result.append(root)
    return result
