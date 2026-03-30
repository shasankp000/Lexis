"""Lexis-R decompressor.

Reads a .lexisr file (zstd-wrapped msgpack), loads the serialised
ContextMixingModel directly from the payload, then arithmetic-decodes
the char stream back to text.

Usage
-----
    from lexis_r.decompress import decompress
    text = decompress("output.lexisr")
"""

from __future__ import annotations

import os
import tempfile
import zlib
from pathlib import Path
from typing import Any, Dict, List

import msgpack

from compression.alphabet.morph_codes import apply_morph
from compression.alphabet.phonetic_map import PHONETIC_CLASSES
from compression.pipeline.stage1c_symbol_slots import splice_symbols, unpack_slot_map
from compression.pipeline.stage5_discourse_symbols import decode_symbols
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.stage9_autocorrect import autocorrect
from lexis_r import huffman
from lexis_r.arithmetic import ArithmeticDecoder
from lexis_r.lz77_pos import unpack_pos_tags_lz77
from lexis_r.payload import (
    POS_VOCAB,
    unpack_huffman_bits,
    unpack_token_array,
    unpack_u8_list,
    unpack_vlq_list,
)
from lexis_r.zstd_wrap import decompress_payload, is_zstd


# ---------------------------------------------------------------------------
# Context model loader
# ---------------------------------------------------------------------------

def _load_model(model_bytes: bytes) -> ContextMixingModel:
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
# root_lengths reconstruction from Huffman stream
# ---------------------------------------------------------------------------

def _unpack_root_lengths_huffman(
    huff_table: bytes,
    huff_stream: bytes,
    total_count: int,
    sent_counts: List[int],
) -> List[List[int]]:
    flat = huffman.decode(huff_table, huff_stream, total_count)
    nested: List[List[int]] = []
    offset = 0
    for count in sent_counts:
        nested.append(flat[offset: offset + count])
        offset += count
    return nested


# ---------------------------------------------------------------------------
# encoded_sentences rebuilder
# ---------------------------------------------------------------------------

def _build_encoded_sentences(
    payload: Dict[str, Any],
    root_lengths_nested: List[List[int]],
) -> List[Dict]:
    pos_n_tags      = unpack_u8_list(bytes(payload["packed_pos_n_tags"]))
    lz77_bytes      = bytes(payload["packed_pos_tags_lz77"])
    pos_tags_nested = unpack_pos_tags_lz77(lz77_bytes, pos_n_tags, POS_VOCAB)

    mc_data, mc_bits   = payload["packed_morph_codes"]
    morph_codes_nested = unpack_token_array(bytes(mc_data), mc_bits, 4)

    pos_bits        = unpack_huffman_bits(bytes(payload["packed_pos_huffman_bits"]))
    pos_n_tags_list = unpack_u8_list(bytes(payload["packed_pos_n_tags"]))

    encoded: List[Dict] = []
    for idx, lengths in enumerate(root_lengths_nested):
        sent_pos   = pos_tags_nested[idx]    if idx < len(pos_tags_nested)    else []
        sent_morph = morph_codes_nested[idx] if idx < len(morph_codes_nested) else []

        char_pos_tags:    List[str] = []
        char_morph_codes: List[int] = []

        for tok_idx, length in enumerate(lengths):
            pos_tag    = sent_pos[tok_idx]   if tok_idx < len(sent_pos)   else "X"
            morph_code = sent_morph[tok_idx] if tok_idx < len(sent_morph) else 0
            char_pos_tags.append("X");      char_morph_codes.append(0)
            char_pos_tags.extend([pos_tag]       * length)
            char_morph_codes.extend([morph_code] * length)
            char_pos_tags.append("X");      char_morph_codes.append(0)
            if tok_idx < len(lengths) - 1:
                char_pos_tags.append("X"); char_morph_codes.append(0)

        encoded.append({
            "char_pos_tags":    char_pos_tags,
            "char_morph_codes": char_morph_codes,
            "pos_huffman_bits": float(pos_bits[idx])      if idx < len(pos_bits)      else 0.0,
            "pos_n_tags":       int(pos_n_tags_list[idx]) if idx < len(pos_n_tags_list) else 0,
            "pos_tags":         sent_pos,
            "morph_codes":      sent_morph,
        })

    return encoded


# ---------------------------------------------------------------------------
# Sentinel reconstruction
# ---------------------------------------------------------------------------

def _sentinel_layout(lengths: List[int]) -> List[bool]:
    layout: List[bool] = []
    for t_idx, length in enumerate(lengths):
        layout.append(True)
        layout.extend([False] * length)
        layout.append(True)
        if t_idx < len(lengths) - 1:
            layout.append(False)
    return layout


def _reconstruct_sentinel_deltas_per_sentence(
    content_nested:      List[List[int]],
    root_lengths_nested: List[List[int]],
) -> List[List[int]]:
    full_nested: List[List[int]] = []
    for s_idx, content in enumerate(content_nested):
        lengths      = root_lengths_nested[s_idx] if s_idx < len(root_lengths_nested) else []
        layout       = _sentinel_layout(lengths)
        content_iter = iter(content)
        full:          List[int] = []
        prev_abs_pos: int | None = None
        sent_count               = 0

        for is_sent in layout:
            if is_sent:
                target_abs = 0 if (sent_count % 2 == 0) else 1
                delta = target_abs if prev_abs_pos is None else target_abs - prev_abs_pos
                full.append(delta)
                prev_abs_pos = target_abs
                sent_count  += 1
            else:
                delta = next(content_iter, 0)
                prev_abs_pos = delta if prev_abs_pos is None else prev_abs_pos + delta
                full.append(delta)

        full_nested.append(full)
    return full_nested


# ---------------------------------------------------------------------------
# Character stream reconstruction
# ---------------------------------------------------------------------------

def _cumulative_from_deltas(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    values = [deltas[0]]
    for d in deltas[1:]:
        values.append(values[-1] + d)
    return values


def _reconstruct_chars_per_sentence(
    char_classes:         List[int],
    pos_deltas_nested:    List[List[int]],
    sentence_char_counts: List[int],
) -> str:
    inverse_map = {coords: ch for ch, coords in PHONETIC_CLASSES.items()}
    chars: List[str] = []
    cls_offset = 0
    for s_idx, count in enumerate(sentence_char_counts):
        classes   = char_classes[cls_offset: cls_offset + count]
        deltas    = pos_deltas_nested[s_idx] if s_idx < len(pos_deltas_nested) else []
        positions = _cumulative_from_deltas(deltas)
        for cls, pos in zip(classes, positions):
            ch = inverse_map.get((cls, pos))
            if ch is not None:
                chars.append(ch)
        cls_offset += count
    return "".join(chars)


def _split_roots(char_stream: str) -> List[str]:
    roots:   List[str] = []
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


_ATTACH_LEFT      = set(".,;:!?)'-—%-/")
_ATTACH_RIGHT     = set("($#/")
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
    raw: bytes = Path(input_path).read_bytes()

    if is_zstd(raw):
        print("[Decompress] Detected zstd wrapper, decompressing...")
        raw = decompress_payload(raw)

    payload: Dict[str, Any] = msgpack.unpackb(raw, raw=False, strict_map_key=False)

    symbol_table: dict  = payload.get("symbol_table", {})
    slot_map            = unpack_slot_map(payload.get("slot_map", []))
    clean_text: str     = payload.get("slot_clean_text", "")

    print("[Decompress] Loading context model from payload...")
    context_model = _load_model(zlib.decompress(bytes(payload["context_model_data"])))

    print("[Decompress] Decoding root_lengths from Huffman stream...")
    rl_sent_counts      = unpack_vlq_list(bytes(payload["root_lengths_sent_counts"]))
    root_lengths_nested = _unpack_root_lengths_huffman(
        bytes(payload["root_lengths_huffman_table"]),
        bytes(payload["root_lengths_huffman_stream"]),
        int(payload["root_lengths_total_count"]),
        rl_sent_counts,
    )

    print("[Decompress] Rebuilding context stream from metadata...")
    encoded_sentences = _build_encoded_sentences(payload, root_lengths_nested)

    print("[Decompress] Arithmetic decoding char stream...")
    num_symbols  = int(payload["num_symbols"])
    dec          = ArithmeticDecoder()
    char_classes = dec.decode(
        bytes(payload["compressed_bitstream"]),
        context_model,
        encoded_sentences,
        num_symbols,
    )

    print("[Decompress] Huffman decoding pos_deltas...")
    total_count  = int(payload["pos_deltas_total_count"])
    flat_content = huffman.decode(
        bytes(payload["pos_deltas_huffman_table"]),
        bytes(payload["pos_deltas_huffman_stream"]),
        total_count,
    )

    content_counts = unpack_vlq_list(bytes(payload["pos_deltas_content_counts"]))
    content_nested: List[List[int]] = []
    offset = 0
    for count in content_counts:
        content_nested.append(flat_content[offset: offset + count])
        offset += count

    pos_deltas_nested = _reconstruct_sentinel_deltas_per_sentence(
        content_nested, root_lengths_nested
    )

    sentence_char_counts = unpack_vlq_list(
        bytes(payload["packed_sentence_char_counts"])
    )

    char_stream = _reconstruct_chars_per_sentence(
        char_classes, pos_deltas_nested, sentence_char_counts
    )
    roots = _split_roots(char_stream)

    mc_data, mc_bits   = payload["packed_morph_codes"]
    morph_codes_nested = unpack_token_array(bytes(mc_data), mc_bits, 4)
    morph_codes_flat   = [c for sent in morph_codes_nested for c in sent]

    words = [
        apply_morph(root, morph_codes_flat[i] if i < len(morph_codes_flat) else 0)
        for i, root in enumerate(roots)
    ]
    result = _join_words(words)
    result = result[0].upper() + result[1:] if result else result

    # Re-splice § symbols into the joined string at their original char offsets
    if slot_map and clean_text:
        result = splice_symbols(result, slot_map, clean_text)

    if symbol_table:
        result = decode_symbols(result, symbol_table)

    return autocorrect(result)
