"""Lexis-R payload serialisation helpers.

All token arrays and context matrices are stored in compact binary form.
Critically, root_lengths uses VLQ (no upper bound) instead of 4-bit
packing, which would silently clamp roots longer than 15 characters.

Layout summary
--------------
  pos_tags              5-bit/token bit-pack  (8-bit sentence-length prefix)
  morph_codes           4-bit/token bit-pack  (same scheme; morph codes 0-12)
  root_lengths          base-128 VLQ  — (n_sentences)(n_tokens)(len …)
  char_context          flat 7×7   base-128 VLQ bytes, row-major, no keys
  morph_context         flat 13×7  base-128 VLQ bytes
  struct_context        flat 18×7  base-128 VLQ bytes + zlib
  pos_huffman_bits      base-256 VLQ per value, scaled ×100  (1B count prefix)
  pos_n_tags            base-256 VLQ per value               (1B count prefix)
  sentence_char_counts  base-256 VLQ per value               (1B count prefix)
  pos_freq_table        base-256 VLQ per tag, fixed _POS_VOCAB order
  pos_deltas_counts     1B n_pairs + zigzag base-256 VLQ (delta, count) pairs
  model_weights         3 × float32 big-endian
  char/morph_vocab      uint8 flat (1B count + 1B/entry)
  pos_vocab             packed 5-bit ids

Base-256 VLQ encoding
---------------------
  Non-negative integer n is stored as:
    (number of payload bytes - 1) in bits [7:4] of the first byte,
  ... no — we use a simpler 1-byte-length-prefix scheme:

    [len_byte][b0][b1]…   where len_byte = number of following bytes (1-8)
    value = big-endian unsigned integer from b0…b_n

  This costs 2 bytes for values 0-255, 3 bytes for 256-65535, etc.
  For the uint16 huffman_bits values (typically 0-2000 after ×100 scaling)
  this saves nothing vs uint16 — the win comes from the freq_table where
  most uint32 counts are < 256 (→ 2 bytes vs 4) and from removing the
  hard uint8 clamp on sentence_char_counts / pos_n_tags.

All helpers are pure functions: bytes in, bytes/value out.
"""

from __future__ import annotations

import struct
import zlib
from typing import Dict, List, Tuple

# Fixed POS vocab — order is the implicit schema (never reorder)
_POS_VOCAB: List[str] = [
    "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
_POS_TO_IDX: Dict[str, int] = {tag: i for i, tag in enumerate(_POS_VOCAB)}

_CHAR_CLASSES:  List[int] = list(range(7))    # char_context row keys
_MORPH_CODES_R: List[int] = list(range(13))   # morph_context row keys


# ===========================================================================
# base-128 VLQ (protobuf-style) — used for root_lengths and context matrices
# ===========================================================================

def _vlq_encode(value: int) -> bytes:
    assert value >= 0, f"VLQ requires non-negative value, got {value}"
    groups: List[int] = []
    while True:
        groups.append(value & 0x7F)
        value >>= 7
        if value == 0:
            break
    groups.reverse()
    return bytes([
        (g | 0x80) if i < len(groups) - 1 else g
        for i, g in enumerate(groups)
    ])


def _vlq_decode(data: bytes, offset: int) -> Tuple[int, int]:
    """Decode one base-128 VLQ integer. Returns (value, new_offset)."""
    value = 0
    while True:
        b = data[offset]; offset += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, offset


# ===========================================================================
# base-256 VLQ — length-prefixed big-endian unsigned integers
#
# Wire format:  [n_bytes: uint8][b0 b1 … b_{n-1}: big-endian value]
# n_bytes is in range [1, 8].  Value 0 → [0x01][0x00] (2 bytes).
# ===========================================================================

def _b256_encode(value: int) -> bytes:
    """Encode a non-negative integer as length-prefixed base-256."""
    assert value >= 0, f"b256 requires non-negative value, got {value}"
    if value == 0:
        return b"\x01\x00"
    n_bytes = (value.bit_length() + 7) // 8
    return bytes([n_bytes]) + value.to_bytes(n_bytes, "big")


def _b256_decode(data: bytes, offset: int) -> Tuple[int, int]:
    """Decode one base-256 integer. Returns (value, new_offset)."""
    n_bytes = data[offset]; offset += 1
    value   = int.from_bytes(data[offset: offset + n_bytes], "big")
    return value, offset + n_bytes


# Zigzag encoding for signed integers (maps negatives to positives)
def _zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 63)  # works for any signed int width


def _zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


# ===========================================================================
# Bit-packed token arrays  (pos_tags → 5-bit, morph_codes → 4-bit)
# ===========================================================================

def pack_token_array(
    sentences: List[List[int]], bits_per_val: int
) -> Tuple[bytes, int]:
    """Pack nested int list into a compact bitstream. Returns (data, n_bits)."""
    bits = ""
    for sentence in sentences:
        bits += format(len(sentence), "08b")
        for val in sentence:
            bits += format(val, f"0{bits_per_val}b")
    n_bits  = len(bits)
    n_bytes = (n_bits + 7) // 8
    padded  = bits + "0" * (n_bytes * 8 - n_bits)
    data    = bytes(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8))
    return data, n_bits


def unpack_token_array(
    data: bytes, n_bits: int, bits_per_val: int
) -> List[List[int]]:
    """Inverse of pack_token_array."""
    bits      = "".join(format(b, "08b") for b in data)[:n_bits]
    sentences: List[List[int]] = []
    i = 0
    while i + 8 <= len(bits):
        length = int(bits[i:i + 8], 2)
        i += 8
        vals = [
            int(bits[i + j * bits_per_val: i + (j + 1) * bits_per_val], 2)
            for j in range(length)
        ]
        i += length * bits_per_val
        sentences.append(vals)
    return sentences


# ===========================================================================
# POS tags  (5-bit per token, 18-label vocab)
# ===========================================================================

def pack_pos_tags(
    sentences: List[List[str]],
) -> Tuple[bytes, int]:
    int_sents = [[_POS_TO_IDX.get(t, 17) for t in s] for s in sentences]
    return pack_token_array(int_sents, 5)


def unpack_pos_tags(data: bytes, n_bits: int) -> List[List[str]]:
    int_sents = unpack_token_array(data, n_bits, 5)
    return [[_POS_VOCAB[i] for i in s] for s in int_sents]


# ===========================================================================
# root_lengths  — base-128 VLQ, NO upper bound
# ===========================================================================

def pack_root_lengths(sentences: List[List[int]]) -> bytes:
    """
    Wire format:
        n_sentences (VLQ)
        for each sentence:
            n_tokens (VLQ)
            for each token:
                root_length (VLQ)
    """
    out = bytearray(_vlq_encode(len(sentences)))
    for sent in sentences:
        out += _vlq_encode(len(sent))
        for length in sent:
            out += _vlq_encode(length)
    return bytes(out)


def unpack_root_lengths(data: bytes) -> List[List[int]]:
    offset = 0
    n_sents, offset = _vlq_decode(data, offset)
    sentences: List[List[int]] = []
    for _ in range(n_sents):
        n_tokens, offset = _vlq_decode(data, offset)
        lengths = []
        for _ in range(n_tokens):
            length, offset = _vlq_decode(data, offset)
            lengths.append(length)
        sentences.append(lengths)
    return sentences


# ===========================================================================
# Context matrices  (char_context, morph_context, struct_context)
# ===========================================================================

def pack_context_matrix(
    ctx: Dict, row_keys: List, n_cols: int
) -> bytes:
    """Flatten dict-of-dict to base-128 VLQ byte stream, row-major."""
    out = bytearray()
    for key in row_keys:
        row = ctx.get(key, {})
        for col in range(n_cols):
            out += _vlq_encode(row.get(col, 0))
    return bytes(out)


def unpack_context_matrix(
    data: bytes, row_keys: List, n_cols: int
) -> Dict:
    ctx: Dict = {}
    offset = 0
    for key in row_keys:
        row: Dict = {}
        for col in range(n_cols):
            val, offset = _vlq_decode(data, offset)
            if val > 0:
                row[col] = val
        if row:
            ctx[key] = row
    return ctx


# ===========================================================================
# pos_huffman_bits  — base-256 VLQ, scaled ×100
#
# Wire format: [1B n_values] then n_values × b256_encode(round(v * 100))
# Saves vs uint16 when scaled value < 256 (i.e. v < 2.56 bits — common).
# ===========================================================================

def pack_huffman_bits(values: List[float]) -> bytes:
    out = bytearray([len(values)])
    for v in values:
        out += _b256_encode(min(round(v * 100), 0xFFFFFFFF))
    return bytes(out)


def unpack_huffman_bits(data: bytes) -> List[float]:
    n      = data[0]
    offset = 1
    result: List[float] = []
    for _ in range(n):
        val, offset = _b256_decode(data, offset)
        result.append(val / 100.0)
    return result


# ===========================================================================
# Variable-length uint lists  (pos_n_tags, sentence_char_counts)
#
# Wire format: [1B n_values] then n_values × b256_encode(v)
# Replaces hard uint8 clamp — no silent truncation on long sentences.
# ===========================================================================

def pack_u8_list(values: List[int]) -> bytes:
    """1B count + base-256 VLQ per value. No upper-bound clamp."""
    out = bytearray([len(values)])
    for v in values:
        out += _b256_encode(v)
    return bytes(out)


def unpack_u8_list(data: bytes) -> List[int]:
    n      = data[0]
    offset = 1
    result: List[int] = []
    for _ in range(n):
        val, offset = _b256_decode(data, offset)
        result.append(val)
    return result


# ===========================================================================
# pos_freq_table  — base-256 VLQ, fixed _POS_VOCAB order
#
# Replaces 18 × uint32 (72 bytes).  Most per-tag counts after a short
# training window are < 256, so they cost 2 bytes each (→ 36 bytes).
# ===========================================================================

def pack_pos_freq_table(table: Dict[str, int]) -> bytes:
    out = bytearray()
    for tag in _POS_VOCAB:
        out += _b256_encode(table.get(tag, 0))
    return bytes(out)


def unpack_pos_freq_table(data: bytes) -> Dict[str, int]:
    offset = 0
    result: Dict[str, int] = {}
    for tag in _POS_VOCAB:
        val, offset = _b256_decode(data, offset)
        if val > 0:
            result[tag] = val
    return result


# ===========================================================================
# model_weights  (3 × float32)
# ===========================================================================

def pack_weights(weights: List[float]) -> bytes:
    return struct.pack(">3f", *weights[:3])


def unpack_weights(data: bytes) -> List[float]:
    return list(struct.unpack(">3f", data))


# ===========================================================================
# uint8 vocab arrays  (char_vocab, morph_vocab)
# ===========================================================================

def pack_int_vocab(vocab: List[int]) -> bytes:
    return bytes([len(vocab)] + [v & 0xFF for v in vocab])


def unpack_int_vocab(data: bytes) -> List[int]:
    n = data[0]
    return list(data[1: n + 1])


# ===========================================================================
# pos_vocab  (packed 5-bit ids)
# ===========================================================================

def pack_pos_vocab(vocab: List[str]) -> Tuple[bytes, int]:
    ids = [_POS_TO_IDX.get(t, 17) for t in vocab]
    return pack_token_array([ids], 5)


def unpack_pos_vocab(data: bytes, n_bits: int) -> List[str]:
    sents = unpack_token_array(data, n_bits, 5)
    ids   = sents[0] if sents else []
    return [_POS_VOCAB[i] for i in ids]


# ===========================================================================
# pos_deltas_counts  — zigzag + base-256 VLQ pairs
#
# Wire format: [1B n_pairs] then n_pairs × (b256(zigzag(delta)), b256(count))
# Replaces fixed int8+uint16 (3B/pair).  Small deltas cost 4B total
# (2B each) vs 3B before — but removes the ±127 delta clamp and the
# 65535 count clamp.  Worth it for correctness; counts are typically 1
# so b256(1) = [0x01][0x01] = 2 bytes, saving 1B vs uint16.
# ===========================================================================

def pack_deltas_counts(counts: Dict[int, int]) -> bytes:
    """1B n_pairs + zigzag+b256 (delta, count) pairs."""
    items = sorted(counts.items())
    out   = bytearray([len(items)])
    for delta, count in items:
        out += _b256_encode(_zigzag_encode(delta))
        out += _b256_encode(count)
    return bytes(out)


def unpack_deltas_counts(data: bytes) -> Dict[int, int]:
    n      = data[0]
    offset = 1
    result: Dict[int, int] = {}
    for _ in range(n):
        zz,    offset = _b256_decode(data, offset)
        count, offset = _b256_decode(data, offset)
        result[_zigzag_decode(zz)] = count
    return result


# ===========================================================================
# struct_context  (base-128 VLQ row-major + zlib)
# ===========================================================================

def pack_struct_context(ctx: Dict) -> bytes:
    raw = pack_context_matrix(
        {k: dict(v) for k, v in ctx.items()}, _POS_VOCAB, 7
    )
    return zlib.compress(raw, level=9)


def unpack_struct_context(data: bytes) -> Dict:
    from collections import Counter
    raw = zlib.decompress(data)
    raw_ctx = unpack_context_matrix(raw, _POS_VOCAB, 7)
    return {k: Counter(v) for k, v in raw_ctx.items()}


# ===========================================================================
# Convenience re-exports
# ===========================================================================

POS_VOCAB    = _POS_VOCAB
POS_TO_IDX   = _POS_TO_IDX
CHAR_CLASSES  = _CHAR_CLASSES
MORPH_CODES_R = _MORPH_CODES_R
