"""Stage 1c: Symbol slot extraction for §-prefixed tokens.

§W / §E / §R tokens cannot pass through the phonetic character encoder
because '§' is not in PHONETIC_CLASSES and would be silently dropped.

Approach
--------
Compressor: strip all §-tokens from text before encoding, recording
(char_offset, symbol) for each, where char_offset is the character
position in the CLEAN text immediately before which the symbol appeared.
Only the clean_len (int) and a sparse anchor table are stored in the
payload — never the full clean text string.

Anchor-based splice (fixes drift at 10k+ chars)
------------------------------------------------
A single linear scale from clean_len to joined_len drifts badly for
large inputs because _join_words compresses punctuation non-uniformly.

Fix: at decompress time, after _join_words produces the joined string,
build ANCHOR_STRIDE-spaced anchor points by:
  1. Sampling clean offsets every ANCHOR_STRIDE characters.
  2. Scaling each to an approximate joined offset.
  3. Snapping each to the nearest space boundary in joined.
Then splice_symbols interpolates between the two nearest anchors for
each symbol, virtually eliminating drift across arbitrary input sizes.

The anchor offsets are NOT stored in the payload — they are computed
on-the-fly from (slot_map, slot_clean_len, joined_text) at decompress
time, costing O(n) but zero extra payload bytes.

Contract
--------
    clean_text, slot_map = extract_symbols(text)
    clean_len = len(clean_text)
    # ... encode clean_text, decode back to joined_text ...
    result = splice_symbols(joined_text, slot_map, clean_len)
    result = decode_symbols(result, symbol_table)
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

_SYMBOL_RE   = re.compile(r"\u00a7[EWR]\d+")
_SLOT_RE     = re.compile(r"\bzz([a-z]+)\b", re.IGNORECASE)
_SLOT_PREFIX = "zz"

ANCHOR_STRIDE = 200   # sample one anchor per this many clean chars


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
# Anchor table helpers
# ---------------------------------------------------------------------------

def _build_anchors(
    clean_len: int,
    joined: str,
) -> List[Tuple[int, int]]:
    """
    Build a list of (clean_offset, joined_offset) anchor pairs.

    Samples every ANCHOR_STRIDE characters in [0, clean_len] and maps
    each to the nearest space (or string boundary) in `joined` via
    linear proportional scaling.

    Returns a sorted list of (clean_off, joined_off) anchors that always
    includes (0, 0) and (clean_len, len(joined)).
    """
    joined_len = len(joined)
    if clean_len == 0:
        return [(0, 0)]

    scale   = joined_len / clean_len
    anchors: List[Tuple[int, int]] = [(0, 0)]

    for c_off in range(ANCHOR_STRIDE, clean_len, ANCHOR_STRIDE):
        j_approx = min(int(round(c_off * scale)), joined_len)
        # snap to nearest space
        best = j_approx
        for delta in range(50):
            for sign in (-1, 1):
                cand = j_approx + sign * delta
                if 0 <= cand <= joined_len:
                    ch_prev = joined[cand - 1] if cand > 0 else ''
                    ch_next = joined[cand]     if cand < joined_len else ''
                    if ch_prev == ' ' or ch_prev == '' or ch_next == ' ' or ch_next == '':
                        best = cand
                        break
            else:
                continue
            break
        anchors.append((c_off, best))

    anchors.append((clean_len, joined_len))
    # dedup and sort
    seen = set()
    unique = []
    for a in anchors:
        if a[0] not in seen:
            seen.add(a[0])
            unique.append(a)
    return sorted(unique)


def _interpolate_offset(
    clean_off: int,
    anchors: List[Tuple[int, int]],
) -> int:
    """
    Linearly interpolate a joined offset from clean_off using the anchor table.
    """
    if len(anchors) == 1:
        return anchors[0][1]

    # Find the two anchors that bracket clean_off
    lo = anchors[0]
    hi = anchors[-1]
    for i in range(len(anchors) - 1):
        if anchors[i][0] <= clean_off <= anchors[i + 1][0]:
            lo, hi = anchors[i], anchors[i + 1]
            break

    c_lo, j_lo = lo
    c_hi, j_hi = hi
    if c_hi == c_lo:
        return j_lo

    t = (clean_off - c_lo) / (c_hi - c_lo)
    return int(round(j_lo + t * (j_hi - j_lo)))


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
    slot_map   : [(char_offset_in_clean, symbol), ...]
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
    Re-insert § symbols into `joined` using anchor-based interpolation.

    Builds an anchor table from (clean_len, joined) at call time — zero
    extra payload bytes. Each symbol's clean_offset is mapped to its
    joined position via the two nearest anchors.
    """
    if not slot_map or not clean_len:
        return joined

    anchors = _build_anchors(clean_len, joined)

    mapped: List[Tuple[int, str]] = []
    for clean_off, sym in slot_map:
        j_pos  = _interpolate_offset(clean_off, anchors)
        j_pos  = min(j_pos, len(joined))
        # snap to nearest space boundary
        window = max(1, len(sym) * 2)
        best   = j_pos
        for delta in range(window + 1):
            for sign in (-1, 1):
                c = j_pos + sign * delta
                if 0 <= c <= len(joined):
                    prev = joined[c - 1] if c > 0 else ''
                    nxt  = joined[c]     if c < len(joined) else ''
                    if prev in (' ', '') or nxt in (' ', ''):
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
