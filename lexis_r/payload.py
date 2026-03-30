"""Lexis-R payload serialisation helpers.

All token arrays and context matrices are stored in compact binary form.
Critically, root_lengths uses VLQ (no upper bound) instead of 4-bit
packing, which would silently clamp roots longer than 15 characters.

Layout summary
--------------
  pos_tags          5-bit/token bit-pack  (8-bit sentence-length prefix)
  morph_codes       4-bit/token bit-pack  (same scheme; morph codes 0-12)
  root_lengths      VLQ  — (n_sentences VLQ)(n_tokens VLQ)(len VLQ …)
  char_context      flat 7×7   VLQ bytes, row-major, no keys
  morph_context     flat 13×7  VLQ bytes
  struct_context    flat 18×7  VLQ bytes + zlib
  pos_huffman_bits  uint16 ×N, scaled ×100 (0.01-bit precision)
  pos_n_tags        uint8 list (1B count + 1B/sentence)
  sentence_char_counts  uint8 list
  pos_freq_table    18 × uint32 fixed-order flat array
  model_weights     3 × float32 big-endian
  char/morph_vocab  uint8 flat (1B count + 1B/entry)
  pos_vocab         packed 5-bit ids
  pos_deltas_counts sorted (int8, uint16) pairs

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
# VLQ (base-128, protobuf-style)
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
    """Decode one VLQ integer starting at offset. Returns (value, new_offset)."""
    value = 0
    while True:
        b = data[offset]; offset += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, offset


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
# root_lengths  — VLQ, NO upper bound
# ===========================================================================

def pack_root_lengths(sentences: List[List[int]]) -> bytes:
    """
    Serialise nested root-length list as VLQ bytes.

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
    """Inverse of pack_root_lengths."""
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
    """Flatten dict-of-dict to VLQ byte stream, row-major. No keys stored."""
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
# pos_huffman_bits  (uint16 ×N, scaled ×100)
# ===========================================================================

def pack_huffman_bits(values: List[float]) -> bytes:
    """1B count + 2B per value (big-endian uint16, 0.01-bit precision)."""
    n = len(values)
    q = [min(round(v * 100), 65535) for v in values]
    return bytes([n]) + b"".join(struct.pack(">H", x) for x in q)


def unpack_huffman_bits(data: bytes) -> List[float]:
    n = data[0]
    return [
        struct.unpack(">H", data[1 + i * 2: 3 + i * 2])[0] / 100.0
        for i in range(n)
    ]


# ===========================================================================
# uint8 lists  (pos_n_tags, sentence_char_counts)
# ===========================================================================

def pack_u8_list(values: List[int]) -> bytes:
    """1B count + 1B/value. Values clamped to uint8."""
    clamped = [min(v, 255) for v in values]
    return bytes([len(clamped)] + clamped)


def unpack_u8_list(data: bytes) -> List[int]:
    n = data[0]
    return list(data[1: n + 1])


# ===========================================================================
# pos_freq_table  (18 × uint32, fixed _POS_VOCAB order)
# ===========================================================================

def pack_pos_freq_table(table: Dict[str, int]) -> bytes:
    return b"".join(struct.pack(">I", table.get(tag, 0)) for tag in _POS_VOCAB)


def unpack_pos_freq_table(data: bytes) -> Dict[str, int]:
    return {
        tag: struct.unpack(">I", data[i * 4: i * 4 + 4])[0]
        for i, tag in enumerate(_POS_VOCAB)
        if struct.unpack(">I", data[i * 4: i * 4 + 4])[0] > 0
    }


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
# pos_deltas_counts  (sorted int8 + uint16 pairs)
# ===========================================================================

def pack_deltas_counts(counts: Dict[int, int]) -> bytes:
    """1B count + sorted (int8 delta, uint16 count) pairs."""
    items = sorted(counts.items())
    out   = bytearray([len(items)])
    for delta, count in items:
        out += struct.pack(">bH", max(-128, min(127, delta)), min(count, 65535))
    return bytes(out)


def unpack_deltas_counts(data: bytes) -> Dict[int, int]:
    n      = data[0]
    result: Dict[int, int] = {}
    for i in range(n):
        delta, count = struct.unpack(">bH", data[1 + i * 3: 4 + i * 3])
        result[delta] = count
    return result


# ===========================================================================
# struct_context  (VLQ row-major + zlib)
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
# Convenience re-exports for the rest of lexis_r
# ===========================================================================

POS_VOCAB    = _POS_VOCAB
POS_TO_IDX   = _POS_TO_IDX
CHAR_CLASSES  = _CHAR_CLASSES
MORPH_CODES_R = _MORPH_CODES_R
