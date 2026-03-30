"""Lexis-R decompressor."""

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


def _unpack_root_lengths_huffman(
    huff_table: bytes, huff_stream: bytes, total_count: int, sent_counts: List[int]
) -> List[List[int]]:
    flat = huffman.decode(huff_table, huff_stream, total_count)
    nested, offset = [], 0
    for count in sent_counts:
        nested.append(flat[offset: offset + count])
        offset += count
    return nested


def _build_encoded_sentences(payload: Dict[str, Any], root_lengths_nested: List[List[int]]) -> List[Dict]:
    pos_n_tags      = unpack_u8_list(bytes(payload["packed_pos_n_tags"]))
    pos_tags_nested = unpack_pos_tags_lz77(bytes(payload["packed_pos_tags_lz77"]), pos_n_tags, POS_VOCAB)
    mc_data, mc_bits   = payload["packed_morph_codes"]
    morph_codes_nested = unpack_token_array(bytes(mc_data), mc_bits, 4)
    pos_bits        = unpack_huffman_bits(bytes(payload["packed_pos_huffman_bits"]))
    pos_n_tags_list = unpack_u8_list(bytes(payload["packed_pos_n_tags"]))
    encoded: List[Dict] = []
    for idx, lengths in enumerate(root_lengths_nested):
        sent_pos   = pos_tags_nested[idx]    if idx < len(pos_tags_nested)    else []
        sent_morph = morph_codes_nested[idx] if idx < len(morph_codes_nested) else []
        char_pos_tags, char_morph_codes = [], []
        for tok_idx, length in enumerate(lengths):
            pos_tag    = sent_pos[tok_idx]   if tok_idx < len(sent_pos)   else "X"
            morph_code = sent_morph[tok_idx] if tok_idx < len(sent_morph) else 0
            char_pos_tags.append("X");      char_morph_codes.append(0)
            char_pos_tags.extend([pos_tag] * length)
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
    content_nested: List[List[int]], root_lengths_nested: List[List[int]]
) -> List[List[int]]:
    full_nested: List[List[int]] = []
    for s_idx, content in enumerate(content_nested):
        lengths      = root_lengths_nested[s_idx] if s_idx < len(root_lengths_nested) else []
        layout       = _sentinel_layout(lengths)
        content_iter = iter(content)
        full: List[int] = []
        prev_abs_pos: int | None = None
        sent_count = 0
        for is_sent in layout:
            if is_sent:
                target_abs = 0 if sent_count % 2 == 0 else 1
                full.append(target_abs if prev_abs_pos is None else target_abs - prev_abs_pos)
                prev_abs_pos = target_abs
                sent_count  += 1
            else:
                delta = next(content_iter, 0)
                prev_abs_pos = delta if prev_abs_pos is None else prev_abs_pos + delta
                full.append(delta)
        full_nested.append(full)
    return full_nested


def _cumulative_from_deltas(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    values = [deltas[0]]
    for d in deltas[1:]:
        values.append(values[-1] + d)
    return values


def _reconstruct_chars_per_sentence(
    char_classes: List[int], pos_deltas_nested: List[List[int]], sentence_char_counts: List[int]
) -> str:
    """Decode phonetic char stream. Returns only printable phonetic chars.

    NOTE: ^ and $ sentinel markers are NOT present in the output — they
    exist only in the encoder’s char_classes stream as structural markers
    but have no PHONETIC_CLASSES entry. Token boundaries are recovered
    via root_lengths_nested in _split_roots_by_lengths instead.
    """
    inverse_map = {coords: ch for ch, coords in PHONETIC_CLASSES.items()}
    chars: List[str] = []
    cls_offset = 0
    for s_idx, count in enumerate(sentence_char_counts):
        classes   = char_classes[cls_offset: cls_offset + count]
        positions = _cumulative_from_deltas(pos_deltas_nested[s_idx] if s_idx < len(pos_deltas_nested) else [])
        for cls, pos in zip(classes, positions):
            ch = inverse_map.get((cls, pos))
            if ch is not None:
                chars.append(ch)
        cls_offset += count
    return "".join(chars)


def _split_roots_by_lengths(
    char_stream: str,
    root_lengths_nested: List[List[int]],
) -> List[str]:
    """Slice the phonetic char stream into roots using root_lengths as
    ground-truth token boundaries.

    This replaces the old _split_roots which relied on ^ / $ sentinel
    chars in the stream. Those sentinels are silently dropped when their
    (cls, pos) pair is absent from PHONETIC_CLASSES, causing root count
    mismatches at 10k+ chars and cascading morph-code misalignment.

    Each token contributes exactly root_lengths[sent][tok] phonetic chars
    to the stream (sentinels ^ and $ are structural-only and not decoded).
    """
    roots: List[str] = []
    offset = 0
    for lengths in root_lengths_nested:
        for length in lengths:
            roots.append(char_stream[offset: offset + length])
            offset += length
    return roots


# kept for backward compat / debugging
def _split_roots(char_stream: str) -> List[str]:
    roots, current = [], []
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
            parts.append(" *" if not parts[-1].endswith("*") else "*"); continue
        if parts[-1].endswith(("-", "/")):
            parts[-1] += word; continue
        if parts[-1] and parts[-1][-1] in _ATTACH_RIGHT:
            parts[-1] += word; continue
        fc = word[0]
        if fc == '"':
            prev = parts[-1][-1] if parts[-1] else ""
            parts.append(' "' if prev in _OPEN_QUOTE_AFTER or not parts[-1] else '"'); continue
        if fc == "'" or fc in _ATTACH_LEFT:
            parts[-1] += word
        else:
            parts.append(" " + word)
    result = "".join(parts)
    result = result.replace("( ", "(").replace(" )", ")")
    result = result.replace("[ ", "[").replace(" ]", "]")
    result = result.replace(" — ", "—")
    return result.strip()


def decompress(input_path: str) -> str:
    raw: bytes = Path(input_path).read_bytes()
    if is_zstd(raw):
        print("[Decompress] Detected zstd wrapper, decompressing...")
        raw = decompress_payload(raw)

    payload: Dict[str, Any] = msgpack.unpackb(raw, raw=False, strict_map_key=False)

    symbol_table: dict = payload.get("symbol_table", {})
    slot_map  = unpack_slot_map(payload.get("slot_map", []))
    clean_len = int(payload.get("slot_clean_len", 0))

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
    char_classes = ArithmeticDecoder().decode(
        bytes(payload["compressed_bitstream"]),
        context_model,
        encoded_sentences,
        int(payload["num_symbols"]),
    )

    print("[Decompress] Huffman decoding pos_deltas...")
    flat_content = huffman.decode(
        bytes(payload["pos_deltas_huffman_table"]),
        bytes(payload["pos_deltas_huffman_stream"]),
        int(payload["pos_deltas_total_count"]),
    )
    content_counts = unpack_vlq_list(bytes(payload["pos_deltas_content_counts"]))
    content_nested, offset = [], 0
    for count in content_counts:
        content_nested.append(flat_content[offset: offset + count])
        offset += count

    pos_deltas_nested    = _reconstruct_sentinel_deltas_per_sentence(content_nested, root_lengths_nested)
    sentence_char_counts = unpack_vlq_list(bytes(payload["packed_sentence_char_counts"]))
    char_stream          = _reconstruct_chars_per_sentence(char_classes, pos_deltas_nested, sentence_char_counts)

    # Use root_lengths as ground-truth token boundaries — do NOT use
    # _split_roots which relies on ^ / $ markers that may be dropped.
    roots = _split_roots_by_lengths(char_stream, root_lengths_nested)

    mc_data, mc_bits   = payload["packed_morph_codes"]
    morph_codes_nested = unpack_token_array(bytes(mc_data), mc_bits, 4)
    morph_codes_flat   = [c for sent in morph_codes_nested for c in sent]

    words  = [apply_morph(root, morph_codes_flat[i] if i < len(morph_codes_flat) else 0) for i, root in enumerate(roots)]
    result = _join_words(words)
    result = result[0].upper() + result[1:] if result else result

    if slot_map and clean_len:
        result = splice_symbols(result, slot_map, clean_len)

    if symbol_table:
        result = decode_symbols(result, symbol_table)

    return autocorrect(result)
