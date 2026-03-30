"""Lexis-R payload serialisation helpers.

All token arrays and context matrices are stored in compact binary form.
Critically, root_lengths uses VLQ (no upper bound) instead of 4-bit
packing, which would silently clamp roots longer than 15 characters.

Layout summary
--------------
  pos_tags              LZ77 pointer encoded (token-level, cross-sentence)
  morph_codes           4-bit packed per token
  root_lengths          delta + zigzag + base-128 VLQ
  char/morph_context    flat VLQ bytes, row-major, no keys
  struct_context        flat VLQ bytes + zlib
  pos_huffman_bits      uint8 list  (1B count + 1B/sentence, values already integers)
  pos_n_tags            uint8 list (1B count + 1B/sentence)
  sentence_char_counts  uint8 list (1B count + 1B/sentence)
  pos_freq_table        base-256 VLQ per tag, fixed _POS_VOCAB order
  pos_deltas_counts     1B n_pairs + zigzag base-128 VLQ (delta, count) pairs
  model_weights         3 x float32 big-endian
  char/morph_vocab      uint8 flat (1B count + 1B/entry)
  pos_vocab             packed 5-bit ids

Encoding selection rationale
-----------------------------
  base-128 VLQ : best for values with unbounded range; costs 1B for 0-127
  base-256 VLQ : best for values 256+ that are rarely small (e.g. freq counts)
  uint8        : best when values are provably < 256 (sentence lengths, n_tags,
                 pos_huffman_bits which are integer costs < 256)
  delta+zigzag : best when consecutive values are correlated (root_lengths)
  RLE          : best for any repeated-value runs
  LZ77         : best when short repeated sequences exist (pos_tags patterns)

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
# base-128 VLQ (protobuf-style)
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
    value = 0
    while True:
        b = data[offset]; offset += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, offset


# ===========================================================================
# base-256 VLQ (length-prefixed big-endian)
# Used only for pos_freq_table where counts grow large on real text.
# ===========================================================================

def _b256_encode(value: int) -> bytes:
    assert value >= 0
    if value == 0:
        return b"\x01\x00"
    n = (value.bit_length() + 7) // 8
    return bytes([n]) + value.to_bytes(n, "big")


def _b256_decode(data: bytes, offset: int) -> Tuple[int, int]:
    n     = data[offset]; offset += 1
    value = int.from_bytes(data[offset: offset + n], "big")
    return value, offset + n


# Zigzag for signed integers
def _zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def _zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


# ===========================================================================
# Bit-packed token arrays  (pos_tags -> 5-bit, morph_codes -> 4-bit)
# ===========================================================================

def pack_token_array(
    sentences: List[List[int]], bits_per_val: int
) -> Tuple[bytes, int]:
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
# POS tags  (5-bit per token, 18-label vocab) — kept for fallback
# ===========================================================================

def pack_pos_tags(sentences: List[List[str]]) -> Tuple[bytes, int]:
    int_sents = [[_POS_TO_IDX.get(t, 17) for t in s] for s in sentences]
    return pack_token_array(int_sents, 5)


def unpack_pos_tags(data: bytes, n_bits: int) -> List[List[str]]:
    int_sents = unpack_token_array(data, n_bits, 5)
    return [[_POS_VOCAB[i] for i in s] for s in int_sents]


# ===========================================================================
# morph_codes  — generalised RLE (kept for reference)
# Production uses 4-bit pack_token_array.
# ===========================================================================

def pack_morph_codes_rle(sentences: List[List[int]]) -> bytes:
    out = bytearray(_vlq_encode(len(sentences)))
    for sent in sentences:
        out += bytes([len(sent)])
        i = 0
        n = len(sent)
        while i < n:
            val = sent[i]
            run = 1
            while i + run < n and sent[i + run] == val and run < 255:
                run += 1
            if run >= 2:
                out += bytes([0x00, run, val])
            else:
                out += bytes([0x01, val])
            i += run
    return bytes(out)


def unpack_morph_codes_rle(data: bytes) -> List[List[int]]:
    offset = 0
    n_sents, offset = _vlq_decode(data, offset)
    sentences: List[List[int]] = []
    for _ in range(n_sents):
        n_tokens = data[offset]; offset += 1
        tokens: List[int] = []
        while len(tokens) < n_tokens:
            flag = data[offset]; offset += 1
            if flag == 0x00:
                run = data[offset]; offset += 1
                val = data[offset]; offset += 1
                tokens.extend([val] * run)
            else:  # 0x01
                val = data[offset]; offset += 1
                tokens.append(val)
        sentences.append(tokens)
    return sentences


# ===========================================================================
# root_lengths  — delta + zigzag + base-128 VLQ
# ===========================================================================

def pack_root_lengths(sentences: List[List[int]]) -> bytes:
    out = bytearray(_vlq_encode(len(sentences)))
    for sent in sentences:
        out += _vlq_encode(len(sent))
        if not sent:
            continue
        out += _vlq_encode(sent[0])
        prev = sent[0]
        for length in sent[1:]:
            out += _vlq_encode(_zigzag_encode(length - prev))
            prev = length
    return bytes(out)


def unpack_root_lengths(data: bytes) -> List[List[int]]:
    offset = 0
    n_sents, offset = _vlq_decode(data, offset)
    sentences: List[List[int]] = []
    for _ in range(n_sents):
        n_tokens, offset = _vlq_decode(data, offset)
        if n_tokens == 0:
            sentences.append([])
            continue
        first, offset = _vlq_decode(data, offset)
        lengths = [first]
        prev = first
        for _ in range(n_tokens - 1):
            zz, offset = _vlq_decode(data, offset)
            length = prev + _zigzag_decode(zz)
            lengths.append(length)
            prev = length
        sentences.append(lengths)
    return sentences


# ===========================================================================
# Context matrices
# ===========================================================================

def pack_context_matrix(ctx: Dict, row_keys: List, n_cols: int) -> bytes:
    out = bytearray()
    for key in row_keys:
        row = ctx.get(key, {})
        for col in range(n_cols):
            out += _vlq_encode(row.get(col, 0))
    return bytes(out)


def unpack_context_matrix(data: bytes, row_keys: List, n_cols: int) -> Dict:
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
# pos_huffman_bits  — plain uint8 list
#
# Values are sentence-level Huffman costs that arrive as floats but are
# always integers in practice (e.g. 164.0, 131.0) and fit in a uint8.
# x100 scaling was inflating 2-digit integers to 5-digit VLQ values;
# storing them raw as uint8 (same layout as pos_n_tags) is optimal.
#
# Wire format:  1B n  |  n x uint8
# ===========================================================================

def pack_huffman_bits(values: List[float]) -> bytes:
    clamped = [min(int(round(v)), 255) for v in values]
    return bytes([len(clamped)] + clamped)


def unpack_huffman_bits(data: bytes) -> List[float]:
    n = data[0]
    return [float(b) for b in data[1: n + 1]]


# ===========================================================================
# uint8 lists  (pos_n_tags, sentence_char_counts)
# ===========================================================================

def pack_u8_list(values: List[int]) -> bytes:
    clamped = [min(v, 255) for v in values]
    return bytes([len(clamped)] + clamped)


def unpack_u8_list(data: bytes) -> List[int]:
    n = data[0]
    return list(data[1: n + 1])


# ===========================================================================
# pos_freq_table  — base-256 VLQ, fixed _POS_VOCAB order
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
# model_weights  (3 x float32)
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
    return list(data[1: data[0] + 1])


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
# pos_deltas_counts  — zigzag + base-128 VLQ pairs
# ===========================================================================

def pack_deltas_counts(counts: Dict[int, int]) -> bytes:
    items = sorted(counts.items())
    out   = bytearray([len(items)])
    for delta, count in items:
        out += _vlq_encode(_zigzag_encode(delta))
        out += _vlq_encode(count)
    return bytes(out)


def unpack_deltas_counts(data: bytes) -> Dict[int, int]:
    n      = data[0]
    offset = 1
    result: Dict[int, int] = {}
    for _ in range(n):
        zz,    offset = _vlq_decode(data, offset)
        count, offset = _vlq_decode(data, offset)
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
