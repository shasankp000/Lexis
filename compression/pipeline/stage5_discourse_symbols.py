"""Stage 5: Discourse symbol encoding — lossless coreference and relation substitution.

Takes Stage 4 output (coreference_chains + discourse_relations) and produces:
  - A compressed text stream where repeated entity mentions are replaced with §E{n} symbols
  - A SymbolTable mapping each symbol back to its canonical surface form
  - Discourse connectives aligned to detected relations are replaced with §R{n} symbols

Round-trip guarantee: decode_symbols(encode_symbols(text, stage4)[0], stage4) == text
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

SymbolTable = Dict[str, str]
MentionTuple = Tuple  # 3-tuple (sent_idx, token_idx, surface) or 5-tuple with char spans

PRONOUNS: frozenset = frozenset({
    "he", "she", "it", "they", "him", "her", "his", "their", "them", "we", "i",
    "He", "She", "It", "They", "Him", "Her", "His", "Their", "Them", "We", "I",
})

_WORD_CHAR = re.compile(r"\w")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_all_occurrences(text: str, surface: str) -> List[Tuple[int, int]]:
    """
    Find every occurrence of `surface` in `text` that is NOT a strict prefix
    of a longer token (word char immediately after) or a possessive stem
    (apostrophe + word char immediately after).
    """
    results = []
    for m in re.finditer(re.escape(surface), text):
        end = m.end()
        if end < len(text) and _WORD_CHAR.match(text[end]):
            continue
        if end < len(text) and text[end] in ("'", "\u2019", "\u2018"):
            if end + 1 < len(text) and _WORD_CHAR.match(text[end + 1]):
                continue
        results.append((m.start(), m.end()))
    return results


def _sentence_char_boundaries(text: str) -> List[Tuple[int, int]]:
    """
    Split text into sentences (same regex as Stage 4's _chunk_at_sentences)
    and return (char_start, char_end) for each sentence.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text)
    boundaries: List[Tuple[int, int]] = []
    cursor = 0
    for sent in sentences:
        start = text.index(sent, cursor)
        end = start + len(sent)
        boundaries.append((start, end))
        cursor = end
    return boundaries


def _get_char_offsets(
    text: str,
    mentions: List[MentionTuple],
) -> List[Tuple[int, int, str]]:
    """
    Resolve mention tuples to (char_start, char_end, surface), sorted by char_start.

    If Stage 4 provides 5-tuples with char spans, use them directly.

    Otherwise: for each mention (sent_idx, token_idx, surface), find the occurrence
    of `surface` that falls within the sentence indicated by sent_idx. This prevents
    grabbing the wrong occurrence of a common pronoun (e.g. "Him") that appears in
    multiple sentences.
    """
    has_offsets = mentions and len(mentions[0]) >= 5
    if has_offsets:
        resolved = [(m[3], m[4], m[2]) for m in mentions]
        return sorted(resolved, key=lambda x: x[0])

    # Build sentence boundaries once
    sent_bounds = _sentence_char_boundaries(text)

    # Pre-build occurrence list per surface
    surface_pool: Dict[str, List[Tuple[int, int]]] = {}
    for m in mentions:
        surface = m[2]
        if surface not in surface_pool:
            surface_pool[surface] = _find_all_occurrences(text, surface)

    resolved: List[Tuple[int, int, str]] = []
    used_spans: set = set()

    for m in mentions:
        sent_idx, token_idx, surface = m[0], m[1], m[2]
        occurrences = surface_pool[surface]

        # Determine the char range for this sentence
        if sent_idx < len(sent_bounds):
            sent_start, sent_end = sent_bounds[sent_idx]
        else:
            sent_start, sent_end = 0, len(text)

        # Find the first unused occurrence that falls within the sentence window.
        # If none found in the sentence, fall back to the nearest unused occurrence
        # after sent_start (handles slight sentence-split mismatches).
        best: Optional[Tuple[int, int]] = None
        for occ_start, occ_end in occurrences:
            if (occ_start, occ_end) in used_spans:
                continue
            if sent_start <= occ_start < sent_end:
                best = (occ_start, occ_end)
                break

        if best is None:
            # Fallback: nearest unused occurrence at or after sent_start
            for occ_start, occ_end in occurrences:
                if (occ_start, occ_end) in used_spans and occ_start >= sent_start:
                    continue
                if occ_start >= sent_start:
                    best = (occ_start, occ_end)
                    break

        if best is not None:
            used_spans.add(best)
            resolved.append((best[0], best[1], surface))

    return sorted(resolved, key=lambda x: x[0])


def _pick_anchor(resolved: List[Tuple[int, int, str]]) -> int:
    """Return index of first non-pronoun mention; fall back to 0."""
    for i, (_, _, surface) in enumerate(resolved):
        if surface.strip() not in PRONOUNS:
            return i
    return 0


def _resolve_relation_offsets(
    text: str,
    relations: List[Tuple],
) -> List[Tuple[int, int, str, str]]:
    """Resolve discourse relation tuples to absolute char offsets."""
    has_offsets = relations and len(relations[0]) >= 6
    if has_offsets:
        return sorted(
            [(r[4], r[5], r[2], r[3]) for r in relations],
            key=lambda x: x[0],
        )

    sent_bounds = _sentence_char_boundaries(text)
    connective_pool: Dict[str, List[Tuple[int, int]]] = {}
    for r in relations:
        c = r[2]
        if c not in connective_pool:
            connective_pool[c] = _find_all_occurrences(text, c)

    used_spans: set = set()
    resolved: List[Tuple[int, int, str, str]] = []
    for r in relations:
        sent_idx, connective, rel_type = r[0], r[2], r[3]
        pool = connective_pool[connective]
        sent_start, sent_end = (
            sent_bounds[sent_idx] if sent_idx < len(sent_bounds) else (0, len(text))
        )
        best: Optional[Tuple[int, int]] = None
        for occ_start, occ_end in pool:
            if (occ_start, occ_end) not in used_spans and sent_start <= occ_start < sent_end:
                best = (occ_start, occ_end)
                break
        if best is None:
            for occ_start, occ_end in pool:
                if (occ_start, occ_end) not in used_spans and occ_start >= sent_start:
                    best = (occ_start, occ_end)
                    break
        if best is not None:
            used_spans.add(best)
            resolved.append((best[0], best[1], connective, rel_type))

    return sorted(resolved, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_symbol_table(stage4_result: Dict) -> SymbolTable:
    """Build a SymbolTable from Stage 4 output without modifying text."""
    table: SymbolTable = {}
    for eid, mentions in stage4_result.get("coreference_chains", {}).items():
        if mentions:
            table[f"§E{eid}"] = mentions[0][2]
    for ridx, rel in enumerate(stage4_result.get("discourse_relations", [])):
        table[f"§R{ridx}"] = rel[2]
    return table


def encode_symbols(
    text: str,
    stage4_result: Dict,
    encode_relations: bool = False,
) -> Tuple[str, SymbolTable]:
    """
    Replace repeated entity mentions (and optionally discourse connectives)
    with compact symbols. Lossless: decode_symbols(result, table) == text.
    """
    symbol_table: SymbolTable = {}
    ops: List[Tuple[int, int, str]] = []

    for eid, mentions in stage4_result.get("coreference_chains", {}).items():
        if not mentions:
            continue
        resolved = _get_char_offsets(text, mentions)
        if not resolved:
            continue

        symbol = f"§E{eid}"
        anchor_idx = _pick_anchor(resolved)
        symbol_table[symbol] = resolved[anchor_idx][2]

        for i, (char_start, char_end, _) in enumerate(resolved):
            if i != anchor_idx:
                ops.append((char_start, char_end, symbol))

    if encode_relations:
        resolved_rels = _resolve_relation_offsets(
            text, stage4_result.get("discourse_relations", [])
        )
        for ridx, (char_start, char_end, connective, _) in enumerate(resolved_rels):
            symbol = f"§R{ridx}"
            symbol_table[symbol] = connective
            ops.append((char_start, char_end, symbol))

    ops.sort(key=lambda x: -x[0])
    compressed = text
    for char_start, char_end, sym in ops:
        compressed = compressed[:char_start] + sym + compressed[char_end:]

    return compressed, symbol_table


def decode_symbols(compressed: str, symbol_table: SymbolTable) -> str:
    """
    Reconstruct original text. Symbols replaced longest-first to prevent
    partial matches (e.g. §E10 matched before §E1).
    """
    result = compressed
    for symbol, surface in sorted(symbol_table.items(), key=lambda x: -len(x[0])):
        result = result.replace(symbol, surface)
    return result


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_round_trip(
    original: str,
    stage4_result: Dict,
    encode_relations: bool = False,
) -> Dict:
    """
    Full encode → decode cycle with detailed diagnostics.

    Returns:
      round_trip_ok, original_tokens, compressed_tokens,
      reduction_pct, first_mismatch_idx, symbols_used, symbol_table
    """
    compressed, symbol_table = encode_symbols(
        original, stage4_result, encode_relations=encode_relations
    )
    decoded = decode_symbols(compressed, symbol_table)
    round_trip_ok = decoded == original

    first_mismatch: Optional[int] = None
    if not round_trip_ok:
        for i, (a, b) in enumerate(zip(decoded, original)):
            if a != b:
                first_mismatch = i
                break
        if first_mismatch is None:
            first_mismatch = min(len(decoded), len(original))

    orig_tokens = len(original.split())
    comp_tokens = len(compressed.split())
    reduction = ((orig_tokens - comp_tokens) / orig_tokens * 100) if orig_tokens else 0.0

    return {
        "round_trip_ok": round_trip_ok,
        "original_tokens": orig_tokens,
        "compressed_tokens": comp_tokens,
        "reduction_pct": round(reduction, 2),
        "first_mismatch_idx": first_mismatch,
        "symbols_used": len(symbol_table),
        "symbol_table": symbol_table,
    }
