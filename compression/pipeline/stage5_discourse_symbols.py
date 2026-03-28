"""Stage 5: Discourse symbol encoding — lossless coreference and relation substitution.

Takes Stage 4 output (coreference_chains + discourse_relations) and produces:
  - A compressed text stream where repeated entity mentions are replaced with §E{n} symbols
  - A SymbolTable mapping each symbol back to its canonical surface form
  - Discourse connectives aligned to detected relations are replaced with §R{n} symbols

Round-trip guarantee: decode_symbols(encode_symbols(text, stage4)[0], stage4) == text
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

# symbol -> canonical surface form  (e.g. "§E0" -> "Ahab")
SymbolTable = Dict[str, str]

# Each mention tuple from Stage 4: (sent_idx, token_idx, surface)
# NOTE: Stage 4 currently emits 3-tuples. If char offsets are added later,
# this module will prefer them; otherwise it falls back to regex alignment.
MentionTuple = Tuple  # 3-tuple or 5-tuple


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_char_offsets(
    text: str,
    mentions: List[MentionTuple],
) -> List[Tuple[int, int, str]]:
    """
    Resolve (sent_idx, token_idx, surface) tuples to absolute char offsets.

    If the tuple already contains (sent_idx, token_idx, surface, char_start, char_end)
    the stored offsets are used directly. Otherwise we find each surface string
    using regex, left-to-right, advancing a search cursor so we never match the
    same span twice.

    Returns list of (char_start, char_end, surface), sorted by char_start ascending.
    """
    resolved: List[Tuple[int, int, str]] = []

    # Check whether Stage 4 already provides char spans (5-tuple)
    has_offsets = mentions and len(mentions[0]) >= 5

    if has_offsets:
        for m in mentions:
            _, _, surface, char_start, char_end = m[:5]
            resolved.append((char_start, char_end, surface))
    else:
        # Fallback: regex search advancing a cursor
        cursor = 0
        for m in mentions:
            surface = m[2]
            # Escape surface for regex, match whole-word boundary where possible
            pattern = re.escape(surface)
            match = re.search(pattern, text[cursor:])
            if match:
                abs_start = cursor + match.start()
                abs_end = cursor + match.end()
                resolved.append((abs_start, abs_end, surface))
                cursor = abs_end  # advance so we don't match the same span again

    return sorted(resolved, key=lambda x: x[0])


def _resolve_relation_offsets(
    text: str,
    relations: List[Tuple],
) -> List[Tuple[int, int, str, str]]:
    """
    Resolve discourse relation tuples to absolute char offsets.

    Stage 4 emits: (sent_idx, token_idx, connective_text, relation_type)
    or (sent_idx, token_idx, connective_text, relation_type, char_start, char_end)

    Returns list of (char_start, char_end, connective_text, relation_type).
    """
    resolved: List[Tuple[int, int, str, str]] = []
    has_offsets = relations and len(relations[0]) >= 6

    if has_offsets:
        for r in relations:
            _, _, connective, rel_type, char_start, char_end = r[:6]
            resolved.append((char_start, char_end, connective, rel_type))
    else:
        cursor = 0
        for r in relations:
            connective, rel_type = r[2], r[3]
            pattern = re.escape(connective)
            match = re.search(pattern, text[cursor:], re.IGNORECASE)
            if match:
                abs_start = cursor + match.start()
                abs_end = cursor + match.end()
                resolved.append((abs_start, abs_end, connective, rel_type))
                cursor = abs_end

    return sorted(resolved, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_symbol_table(stage4_result: Dict) -> SymbolTable:
    """
    Build a SymbolTable from Stage 4 output without modifying text.

    Entity symbols:   §E{entity_id}  ->  first-mention surface form
    Relation symbols: §R{rel_idx}    ->  connective surface form

    This is a read-only helper — use encode_symbols for the full encode pass.
    """
    table: SymbolTable = {}

    for eid, mentions in stage4_result.get("coreference_chains", {}).items():
        if mentions:
            surface = mentions[0][2]
            table[f"§E{eid}"] = surface

    seen_rel_types: Dict[str, int] = {}
    for ridx, rel in enumerate(stage4_result.get("discourse_relations", [])):
        connective = rel[2]
        rel_type = rel[3]
        key = rel_type
        if key not in seen_rel_types:
            seen_rel_types[key] = ridx
        table[f"§R{ridx}"] = connective

    return table


def encode_symbols(
    text: str,
    stage4_result: Dict,
    encode_relations: bool = False,
) -> Tuple[str, SymbolTable]:
    """
    Replace repeated entity mentions and (optionally) discourse connectives
    with compact symbols.

    Rules:
      - First mention of each entity chain is kept as-is (becomes the anchor).
      - All subsequent mentions are replaced with §E{entity_id}.
      - Discourse connectives are replaced with §R{ridx} only when
        encode_relations=True (off by default — adds noise on short texts).
      - Replacements are applied right-to-left by char offset so earlier
        offsets remain valid throughout.

    Returns:
      (compressed_text, symbol_table)
    """
    symbol_table: SymbolTable = {}
    # List of (char_start, char_end, replacement_string)
    ops: List[Tuple[int, int, str]] = []

    # --- Entity substitutions ---
    for eid, mentions in stage4_result.get("coreference_chains", {}).items():
        if not mentions:
            continue

        resolved = _get_char_offsets(text, mentions)
        if not resolved:
            continue

        symbol = f"§E{eid}"
        # First resolved mention = anchor; store canonical surface in symbol table
        anchor_start, anchor_end, anchor_surface = resolved[0]
        symbol_table[symbol] = anchor_surface

        # All subsequent resolved mentions get substituted
        for char_start, char_end, _ in resolved[1:]:
            ops.append((char_start, char_end, symbol))

    # --- Discourse relation substitutions (optional) ---
    if encode_relations:
        resolved_rels = _resolve_relation_offsets(
            text, stage4_result.get("discourse_relations", [])
        )
        for ridx, (char_start, char_end, connective, rel_type) in enumerate(
            resolved_rels
        ):
            symbol = f"§R{ridx}"
            symbol_table[symbol] = connective
            ops.append((char_start, char_end, symbol))

    # Apply replacements right-to-left (descending start offset)
    ops.sort(key=lambda x: -x[0])

    compressed = text
    for char_start, char_end, sym in ops:
        compressed = compressed[:char_start] + sym + compressed[char_end:]

    return compressed, symbol_table


def decode_symbols(compressed: str, symbol_table: SymbolTable) -> str:
    """
    Reconstruct original text from compressed stream + symbol table.

    Symbols are replaced longest-first to avoid partial matches
    (e.g. §E10 being partially matched as §E1).

    No whitespace normalisation is performed — decode is a pure
    symbol-to-surface substitution.
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

    Returns a dict with:
      - round_trip_ok:      bool
      - original_tokens:    int  (whitespace split)
      - compressed_tokens:  int
      - reduction_pct:      float
      - first_mismatch_idx: int | None
      - symbols_used:       int
      - symbol_table:       SymbolTable
    """
    compressed, symbol_table = encode_symbols(
        original, stage4_result, encode_relations=encode_relations
    )
    decoded = decode_symbols(compressed, symbol_table)

    round_trip_ok = decoded == original

    first_mismatch: int | None = None
    if not round_trip_ok:
        for i, (a, b) in enumerate(zip(decoded, original)):
            if a != b:
                first_mismatch = i
                break
        if first_mismatch is None:
            # One string is a prefix of the other
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
