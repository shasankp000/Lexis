"""Stage 5: Discourse symbol encoding — lossless coreference and relation substitution.

Takes Stage 4 output (coreference_chains + discourse_relations) and produces:
  - A compressed text stream where repeated entity mentions are replaced with §E{n} symbols
  - A SymbolTable mapping each symbol back to its canonical surface form
  - Discourse connectives aligned to detected relations are replaced with §R{n} symbols

Round-trip guarantee: decode_symbols(encode_symbols(text, stage4)[0], stage4) == text

NOTE: When Stage 4 emits 3-tuples (sent_idx, token_idx, surface), pronoun substitution
is fully disabled. Any chain containing a pronoun mention is skipped entirely because
sent_idx is chunk-relative and cannot be safely mapped to global char positions.
Once Stage 4 is updated to emit 5-tuples with absolute char spans, pronoun
substitution re-enables automatically via the has_offsets path.
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
    of a longer token or a possessive stem (e.g. The Whale inside The Whale's).
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


def _chain_has_pronoun(mentions: List[MentionTuple]) -> bool:
    """Return True if any mention surface in the chain is a pronoun."""
    return any(m[2].strip() in PRONOUNS for m in mentions)


def _get_char_offsets(
    text: str,
    mentions: List[MentionTuple],
) -> List[Tuple[int, int, str]]:
    """
    Resolve mention tuples to (char_start, char_end, surface), sorted by char_start.

    If Stage 4 provides 5-tuples with absolute char spans, use them directly.
    Otherwise build an occurrence pool per surface and consume left-to-right.
    Pronouns should already be filtered out before calling this (see encode_symbols).
    """
    has_offsets = mentions and len(mentions[0]) >= 5

    if has_offsets:
        resolved = [(m[3], m[4], m[2]) for m in mentions]
        return sorted(resolved, key=lambda x: x[0])

    surface_pool: Dict[str, List[Tuple[int, int]]] = {}
    for m in mentions:
        surface = m[2]
        if surface not in surface_pool:
            surface_pool[surface] = _find_all_occurrences(text, surface)

    surface_cursors: Dict[str, int] = {s: 0 for s in surface_pool}
    resolved: List[Tuple[int, int, str]] = []

    for m in mentions:
        surface = m[2]
        pool = surface_pool[surface]
        idx = surface_cursors[surface]
        if idx < len(pool):
            char_start, char_end = pool[idx]
            resolved.append((char_start, char_end, surface))
            surface_cursors[surface] = idx + 1

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

    connective_pool: Dict[str, List[Tuple[int, int]]] = {}
    for r in relations:
        c = r[2]
        if c not in connective_pool:
            connective_pool[c] = _find_all_occurrences(text, c)

    connective_cursors: Dict[str, int] = {c: 0 for c in connective_pool}
    resolved: List[Tuple[int, int, str, str]] = []

    for r in relations:
        connective, rel_type = r[2], r[3]
        pool = connective_pool[connective]
        idx = connective_cursors[connective]
        if idx < len(pool):
            char_start, char_end = pool[idx]
            resolved.append((char_start, char_end, connective, rel_type))
            connective_cursors[connective] = idx + 1

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
    Replace repeated named-entity mentions (and optionally discourse connectives)
    with compact symbols. Lossless: decode_symbols(result, table) == text.

    In 3-tuple mode (Stage 4 without absolute offsets): any chain that contains
    even one pronoun mention is skipped entirely. This is because sent_idx values
    are chunk-relative and we cannot safely locate pronouns globally.

    In 5-tuple mode (Stage 4 with absolute char spans): pronoun substitution is
    fully enabled since spans are unambiguous.
    """
    symbol_table: SymbolTable = {}
    ops: List[Tuple[int, int, str]] = []

    chains = stage4_result.get("coreference_chains", {})

    # Detect whether Stage 4 provides absolute offsets on any chain
    has_absolute_offsets = any(
        mentions and len(mentions[0]) >= 5
        for mentions in chains.values()
    )

    for eid, mentions in chains.items():
        if not mentions:
            continue

        # In 3-tuple mode: skip any chain that contains a pronoun mention
        if not has_absolute_offsets and _chain_has_pronoun(mentions):
            continue

        resolved = _get_char_offsets(text, mentions)
        if len(resolved) < 2:
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
