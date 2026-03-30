"""Lexis-R decompressor.

Reads a .lexisr msgpack file, loads the serialised ContextMixingModel
directly from the payload (no re-training), then arithmetic-decodes
the char stream back to text.

Usage
-----
    from lexis_r.decompress import decompress
    text = decompress("output.lexisr")
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import msgpack

from compression.alphabet.morph_codes import apply_morph
from compression.alphabet.phonetic_map import PHONETIC_CLASSES
from compression.pipeline.stage5_discourse_symbols import decode_symbols
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.stage9_autocorrect import autocorrect
from lexis_r.arithmetic import ArithmeticDecoder
from lexis_r.payload import (
    unpack_deltas_counts,
    unpack_huffman_bits,
    unpack_root_lengths,
    unpack_token_array,
    unpack_u8_list,
)


# ---------------------------------------------------------------------------
# Context model loader
# ---------------------------------------------------------------------------

def _load_model(model_bytes: bytes) -> ContextMixingModel:
    """Load a ContextMixingModel from raw bytes (written by compress.py)."""
    with tempfile.NamedTemporaryFile(suffix=".lcm", delete=False) as tf:
        tf.write(model_bytes)
        tmp_path = tf.name
    try:
        model = ContextMixingModel()
        model.load(tmp_path)
        return model
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# encoded_sentences rebuilder  (used only for build_context_stream)
# ---------------------------------------------------------------------------

def _build_encoded_sentences(payload: Dict[str, Any]) -> List[Dict]:
    """
    Rebuild the per-symbol context stream lists from packed metadata.
    These are only used by ArithmeticDecoder.decode() via
    build_context_stream() — not for model training.
    """
    from lexis_r.payload import unpack_pos_tags

    pt_data, pt_bits   = payload["packed_pos_tags"]
    pos_tags_nested    = unpack_pos_tags(bytes(pt_data), pt_bits)

    mc_data, mc_bits   = payload["packed_morph_codes"]
    morph_codes_nested = unpack_token_array(bytes(mc_data), mc_bits, 4)

    root_lengths_nested = unpack_root_lengths(
        bytes(payload["packed_root_lengths_vlq"])
    )

    pos_bits   = unpack_huffman_bits(bytes(payload["packed_pos_huffman_bits"]))
    pos_n_tags = unpack_u8_list(bytes(payload["packed_pos_n_tags"]))

    encoded: List[Dict] = []
    for idx, lengths in enumerate(root_lengths_nested):
        sent_pos   = pos_tags_nested[idx]    if idx < len(pos_tags_nested)    else []
        sent_morph = morph_codes_nested[idx] if idx < len(morph_codes_nested) else []

        char_pos_tags:    List[str] = []
        char_morph_codes: List[int] = []

        for tok_idx, length in enumerate(lengths):
            pos_tag    = sent_pos[tok_idx]   if tok_idx < len(sent_pos)   else "X"
            morph_code = sent_morph[tok_idx] if tok_idx < len(sent_morph) else 0
            char_pos_tags.append("X");      char_morph_codes.append(0)            # ^ marker
            char_pos_tags.extend([pos_tag]      * length)
            char_morph_codes.extend([morph_code] * length)
            char_pos_tags.append("X");      char_morph_codes.append(0)            # $ marker
            if tok_idx < len(lengths) - 1:
                char_pos_tags.append("X"); char_morph_codes.append(0)            # _ space

        encoded.append({
            "char_pos_tags":    char_pos_tags,
            "char_morph_codes": char_morph_codes,
            "pos_huffman_bits": float(pos_bits[idx])   if idx < len(pos_bits)   else 0.0,
            "pos_n_tags":       int(pos_n_tags[idx]) if idx < len(pos_n_tags) else 0,
            "pos_tags":         sent_pos,
            "morph_codes":      sent_morph,
        })

    return encoded


# ---------------------------------------------------------------------------
# Character stream reconstruction helpers
# ---------------------------------------------------------------------------

def _cumulative_from_deltas(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    values = [deltas[0]]
    for d in deltas[1:]:
        values.append(values[-1] + d)
    return values


def _reconstruct_chars(
    char_classes:         List[int],
    pos_deltas:           List[int],
    sentence_char_counts: List[int],
) -> str:
    inverse_map = {coords: ch for ch, coords in PHONETIC_CLASSES.items()}
    chars: List[str] = []
    idx = 0
    for count in sentence_char_counts:
        classes   = char_classes[idx: idx + count]
        deltas    = pos_deltas[idx:  idx + count]
        positions = _cumulative_from_deltas(deltas)
        for cls, pos in zip(classes, positions):
            ch = inverse_map.get((cls, pos))
            if ch is not None:
                chars.append(ch)
        idx += count
    if idx < len(char_classes):
        classes   = char_classes[idx:]
        deltas    = pos_deltas[idx:]
        positions = _cumulative_from_deltas(deltas)
        for cls, pos in zip(classes, positions):
            ch = inverse_map.get((cls, pos))
            if ch is not None:
                chars.append(ch)
    return "".join(chars)


def _split_roots(char_stream: str) -> List[str]:
    roots: List[str] = []
    current: List[str] = []
    for ch in char_stream:
        if ch == "^":
            current = []
        elif ch == "$":
            if current:
                roots.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        roots.append("".join(current))
    return roots


_ATTACH_LEFT  = set(".,;:!?)'-—%-/")
_ATTACH_RIGHT = set("($#/")
_OPEN_QUOTE_AFTER = set("!?( — ")


def _join_words(words: List[str]) -> str:
    if not words:
        return ""
    parts: List[str] = [words[0]]
    for word in words[1:]:
        if not word:
            continue
        if word == "*":
            parts.append(" *" if not parts[-1].endswith("*") else "*")
            continue
        if parts[-1].endswith(("-", "/")):
            parts[-1] += word; continue
        if parts[-1] and parts[-1][-1] in _ATTACH_RIGHT:
            parts[-1] += word; continue
        fc = word[0]
        if fc == '"':
            prev = parts[-1][-1] if parts[-1] else ""
            parts.append(' "' if prev in _OPEN_QUOTE_AFTER or not parts[-1] else '"')
            continue
        if fc == "'" or fc in _ATTACH_LEFT:
            parts[-1] += word
        else:
            parts.append(" " + word)
    result = "".join(parts)
    result = result.replace("( ", "(").replace(" )", ")")
    result = result.replace("[ ", "[").replace(" ]", "]")
    result = result.replace(" — ", "—")
    return result.strip()


# ---------------------------------------------------------------------------
# Public decompress function
# ---------------------------------------------------------------------------

def decompress(input_path: str) -> str:
    """
    Decompress a Lexis-R .lexisr file back to text.

    Steps:
      1. Unpack msgpack payload.
      2. Load ContextMixingModel directly from payload bytes.
      3. Rebuild encoded_sentences for build_context_stream.
      4. Arithmetic-decode char bitstream.
      5. Decode pos-delta bitstream.
      6. Reconstruct char stream → roots → words → text.
      7. Re-expand discourse symbols and autocorrect.
    """
    raw: bytes = Path(input_path).read_bytes()
    payload: Dict[str, Any] = msgpack.unpackb(raw, raw=False, strict_map_key=False)

    symbol_table: dict = payload.get("symbol_table", {})

    # ─ Step 2: load exact trained model ───────────────────────────────────
    print("[Decompress] Loading context model from payload...")
    context_model = _load_model(bytes(payload["context_model_data"]))

    # ─ Step 3: rebuild encoded_sentences (for context stream only) ────────
    print("[Decompress] Rebuilding context stream from metadata...")
    encoded_sentences = _build_encoded_sentences(payload)

    # ─ Step 4: decode char bitstream ──────────────────────────────────────
    print("[Decompress] Arithmetic decoding char stream...")
    num_symbols  = int(payload["num_symbols"])
    dec          = ArithmeticDecoder()
    char_classes = dec.decode(
        bytes(payload["compressed_bitstream"]),
        context_model,
        encoded_sentences,
        num_symbols,
    )

    # ─ Step 5: decode pos-delta bitstream ─────────────────────────────────
    pos_delta_counts = unpack_deltas_counts(
        bytes(payload["packed_pos_deltas_counts"])
    )
    pos_deltas_count = int(payload["pos_deltas_count"])
    pos_deltas = ArithmeticDecoder().decode_unigram_counts(
        bytes(payload["pos_deltas_bitstream"]),
        pos_delta_counts,
        pos_deltas_count,
    )

    # ─ Step 6: reconstruct text ───────────────────────────────────────────
    sentence_char_counts = unpack_u8_list(
        bytes(payload["packed_sentence_char_counts"])
    )
    char_stream = _reconstruct_chars(char_classes, pos_deltas, sentence_char_counts)
    roots       = _split_roots(char_stream)

    mc_data, mc_bits   = payload["packed_morph_codes"]
    morph_codes_nested = unpack_token_array(bytes(mc_data), mc_bits, 4)
    morph_codes_flat   = [c for sent in morph_codes_nested for c in sent]

    words  = [
        apply_morph(root, morph_codes_flat[i] if i < len(morph_codes_flat) else 0)
        for i, root in enumerate(roots)
    ]
    result = _join_words(words)
    result = result[0].upper() + result[1:] if result else result

    # ─ Step 7: discourse decode + autocorrect ─────────────────────────────
    if symbol_table:
        result = decode_symbols(result, symbol_table)

    return autocorrect(result)
