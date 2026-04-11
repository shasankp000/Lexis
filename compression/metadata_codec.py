"""Metadata compression codec for Lexis-E.

Option-B binary envelope:
    4 bytes  magic   : b'LEXI'
    1 byte   version : 0x01
    repeated:
        1 byte   field_id
        3 bytes  length  (little-endian uint24)
        N bytes  blob

Field IDs (FIELD_*):
    0  COMPRESSED_BITSTREAM   raw bytes
    1  POS_DELTAS_BITSTREAM   raw bytes
    2  SYMBOL_TABLE           mode symbol_table
    3  POS_DELTAS_COUNTS      mode flat_dict
    4  POS_DELTAS_COUNT       mode scalar
    5  SENTENCE_CHAR_COUNTS   mode flat_uint
    6  POS_HUFFMAN_BITS       mode float  (nested, 1 sentence per inner list)
    7  POS_N_TAGS             mode flat_uint
    8  POS_TAGS               mode pos
    9  MORPH_CODES            mode int
   10  ROOT_LENGTHS           mode int  (unsigned, but reuse signed encoder)
   11  MODEL_WEIGHTS          mode model_weights
   12  CHAR_CONTEXT           mode sparse_dict
   13  MORPH_CONTEXT          mode sparse_dict
   14  STRUCT_CONTEXT         mode sparse_dict_pos
   15  CHAR_VOCAB             mode flat_uint
   16  MORPH_VOCAB            mode flat_uint
   17  POS_VOCAB              mode pos_freq  (keys only, values all 0)
   18  NUM_SYMBOLS            mode scalar
   19  NUM_CHAR_CLASSES       mode scalar
   20  POS_FREQ_TABLE         mode pos_freq
"""

from __future__ import annotations

import struct
import zlib
from collections import Counter
from typing import Any, Dict, List, Tuple

# ---------------------------------------------------------------------------
# Field ID constants
# ---------------------------------------------------------------------------
FIELD_COMPRESSED_BITSTREAM  = 0
FIELD_POS_DELTAS_BITSTREAM  = 1
FIELD_SYMBOL_TABLE          = 2
FIELD_POS_DELTAS_COUNTS     = 3
FIELD_POS_DELTAS_COUNT      = 4
FIELD_SENTENCE_CHAR_COUNTS  = 5
FIELD_POS_HUFFMAN_BITS      = 6
FIELD_POS_N_TAGS            = 7
FIELD_POS_TAGS              = 8
FIELD_MORPH_CODES           = 9
FIELD_ROOT_LENGTHS          = 10
FIELD_MODEL_WEIGHTS         = 11
FIELD_CHAR_CONTEXT          = 12
FIELD_MORPH_CONTEXT         = 13
FIELD_STRUCT_CONTEXT        = 14
FIELD_CHAR_VOCAB            = 15
FIELD_MORPH_VOCAB           = 16
FIELD_POS_VOCAB             = 17
FIELD_NUM_SYMBOLS           = 18
FIELD_NUM_CHAR_CLASSES      = 19
FIELD_POS_FREQ_TABLE        = 20

MAGIC   = b"LEXI"
VERSION = 0x01

# ---------------------------------------------------------------------------
# UPOS constants (must match main.py / stage3)
# ---------------------------------------------------------------------------
UPOS_TAGS = [
    "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
    "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
TAG_TO_ID = {tag: i for i, tag in enumerate(UPOS_TAGS)}
ID_TO_TAG = {i: tag for i, tag in enumerate(UPOS_TAGS)}

# ---------------------------------------------------------------------------
# Low-level varint / zigzag
# ---------------------------------------------------------------------------

def _zz_enc(n: int) -> int:
    return (n << 1) ^ (n >> 63)

def _zz_dec(n: int) -> int:
    return (n >> 1) ^ -(n & 1)

def _enc_varint(n: int) -> bytearray:
    if n < 0:
        raise ValueError(f"varint requires non-negative value: {n}")
    if n == 0:
        return bytearray([0])
    res = bytearray()
    while n:
        byte = n & 0x7F
        n >>= 7
        if n:
            byte |= 0x80
        res.append(byte)
    return res

def _dec_varint(it) -> int:
    num, shift = 0, 0
    while True:
        byte = next(it)
        num |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            return num
        shift += 7

# ---------------------------------------------------------------------------
# Delta / RLE helpers
# ---------------------------------------------------------------------------

def _delta_enc(data: List[int]) -> List[int]:
    if not data:
        return []
    return [data[0]] + [data[i] - data[i-1] for i in range(1, len(data))]

def _delta_dec(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    res = [deltas[0]]
    for i in range(1, len(deltas)):
        res.append(res[-1] + deltas[i])
    return res

def _rle_enc(data: List[int]):
    if not data:
        return []
    res, prev, count = [], data[0], 1
    for i in range(1, len(data)):
        if data[i] == prev:
            count += 1
        else:
            res.append((count, prev))
            prev, count = data[i], 1
    res.append((count, prev))
    return res

def _f2u(f: float) -> int:
    return struct.unpack("!I", struct.pack("!f", f))[0]

def _u2f(u: int) -> float:
    return struct.unpack("!f", struct.pack("!I", u))[0]

# ---------------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------------

def _best_uint_blob(data: List[int]) -> Tuple[int, bytes]:
    vle = bytearray().join(_enc_varint(x) for x in data)
    d_vle = bytearray().join(_enc_varint(_zz_enc(x)) for x in _delta_enc(data))
    r_vle = bytearray()
    for c, v in _rle_enc(data):
        r_vle.extend(_enc_varint(c))
        r_vle.extend(_enc_varint(v))
    _, strat, blob = min(
        (len(vle),   0, vle),
        (len(d_vle), 1, d_vle),
        (len(r_vle), 2, r_vle),
    )
    return strat, bytes(blob)

def _best_int_blob(data: List[int]) -> Tuple[int, bytes]:
    vle   = bytearray().join(_enc_varint(_zz_enc(x)) for x in data)
    d_vle = bytearray().join(_enc_varint(_zz_enc(x)) for x in _delta_enc(data))
    r_vle = bytearray()
    for c, v in _rle_enc(data):
        r_vle.extend(_enc_varint(c))
        r_vle.extend(_enc_varint(_zz_enc(v)))
    _, strat, blob = min(
        (len(vle),   0, vle),
        (len(d_vle), 1, d_vle),
        (len(r_vle), 2, r_vle),
    )
    return strat, bytes(blob)

def _dec_uint_strat(it, length: int, strat: int) -> List[int]:
    if strat == 0:
        return [_dec_varint(it) for _ in range(length)]
    elif strat == 1:
        return _delta_dec([_zz_dec(_dec_varint(it)) for _ in range(length)])
    else:
        raw: List[int] = []
        while len(raw) < length:
            count = _dec_varint(it)
            val   = _dec_varint(it)
            raw.extend([val] * count)
        return raw

def _dec_int_strat(it, length: int, strat: int) -> List[int]:
    if strat == 0:
        return [_zz_dec(_dec_varint(it)) for _ in range(length)]
    elif strat == 1:
        return _delta_dec([_zz_dec(_dec_varint(it)) for _ in range(length)])
    else:
        raw: List[int] = []
        while len(raw) < length:
            count = _dec_varint(it)
            val   = _zz_dec(_dec_varint(it))
            raw.extend([val] * count)
        return raw

# ---------------------------------------------------------------------------
# Per-mode encode/decode
# ---------------------------------------------------------------------------

# ---- mode: raw bytes (fields 0, 1) ----------------------------------------

def _enc_raw(data: bytes) -> bytes:
    return data

def _dec_raw(blob: bytes) -> bytes:
    return blob

# ---- mode: scalar ---------------------------------------------------------

def _enc_scalar(value: int) -> bytes:
    blob = bytearray()
    blob.extend(_enc_varint(_zz_enc(value)))
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_scalar(blob: bytes) -> int:
    it   = iter(blob[:-4])
    val  = _zz_dec(_dec_varint(it))
    stored = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("scalar: CRC mismatch")
    return val

# ---- mode: flat_uint ------------------------------------------------------

def _enc_flat_uint(data: List[int]) -> bytes:
    blob = bytearray()
    blob.extend(_enc_varint(len(data)))
    strat, payload = _best_uint_blob(data)
    blob.append(strat)
    blob.extend(payload)
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_flat_uint(blob: bytes) -> List[int]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("flat_uint: CRC mismatch")
    it     = iter(blob[:-4])
    length = _dec_varint(it)
    strat  = next(it)
    return _dec_uint_strat(it, length, strat)

# ---- mode: flat_int -------------------------------------------------------

def _enc_flat_int(data: List[int]) -> bytes:
    blob = bytearray()
    blob.extend(_enc_varint(len(data)))
    strat, payload = _best_int_blob(data)
    blob.append(strat)
    blob.extend(payload)
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_flat_int(blob: bytes) -> List[int]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("flat_int: CRC mismatch")
    it     = iter(blob[:-4])
    length = _dec_varint(it)
    strat  = next(it)
    return _dec_int_strat(it, length, strat)

# ---- mode: flat_dict (int->int) ------------------------------------------

def _enc_flat_dict(d: Dict[int, int]) -> bytes:
    blob = bytearray()
    keys = sorted(d.keys())
    blob.extend(_enc_varint(len(keys)))
    if keys:
        last = 0
        for k in keys:
            blob.extend(_enc_varint(_zz_enc(k - last)))
            last = k
        for k in keys:
            blob.extend(_enc_varint(_zz_enc(d[k])))
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_flat_dict(blob: bytes) -> Dict[int, int]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("flat_dict: CRC mismatch")
    it     = iter(blob[:-4])
    length = _dec_varint(it)
    keys: List[int] = []
    last = 0
    for _ in range(length):
        delta = _zz_dec(_dec_varint(it))
        last += delta
        keys.append(last)
    values = [_zz_dec(_dec_varint(it)) for _ in range(length)]
    return dict(zip(keys, values))

# ---- mode: pos (nested list[list[str]]) -----------------------------------

def _enc_pos(sentences: List[List[str]]) -> bytes:
    blob = bytearray()
    blob.extend(_enc_varint(len(sentences)))
    for sent in sentences:
        blob.extend(_enc_varint(len(sent)))
        for tag in sent:
            blob.extend(_enc_varint(TAG_TO_ID.get(tag, TAG_TO_ID["X"])))
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_pos(blob: bytes) -> List[List[str]]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("pos: CRC mismatch")
    it  = iter(blob[:-4])
    n   = _dec_varint(it)
    out: List[List[str]] = []
    for _ in range(n):
        k    = _dec_varint(it)
        sent = [ID_TO_TAG.get(_dec_varint(it), "X") for _ in range(k)]
        out.append(sent)
    return out

# ---- mode: int (nested list[list[int]], signed) ---------------------------

def _enc_int_nested(sentences: List[List[int]]) -> bytes:
    blob = bytearray()
    blob.extend(_enc_varint(len(sentences)))
    for sent in sentences:
        blob.extend(_enc_varint(len(sent)))
        strat, payload = _best_int_blob(sent)
        blob.append(strat)
        blob.extend(payload)
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_int_nested(blob: bytes) -> List[List[int]]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("int_nested: CRC mismatch")
    it  = iter(blob[:-4])
    n   = _dec_varint(it)
    out: List[List[int]] = []
    for _ in range(n):
        k     = _dec_varint(it)
        strat = next(it)
        out.append(_dec_int_strat(it, k, strat))
    return out

# ---- mode: float (nested list[list[float]]) via XOR-delta varint ----------

def _enc_float_nested(sentences: List[List[float]]) -> bytes:
    blob = bytearray()
    blob.extend(_enc_varint(len(sentences)))
    for sent in sentences:
        blob.extend(_enc_varint(len(sent)))
        prev = 0
        for f in sent:
            u     = _f2u(f)
            delta = u ^ prev
            blob.extend(_enc_varint(delta))
            prev  = u
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_float_nested(blob: bytes) -> List[List[float]]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("float_nested: CRC mismatch")
    it  = iter(blob[:-4])
    n   = _dec_varint(it)
    out: List[List[float]] = []
    for _ in range(n):
        k    = _dec_varint(it)
        prev = 0
        sent: List[float] = []
        for _ in range(k):
            delta = _dec_varint(it)
            u     = delta ^ prev
            sent.append(_u2f(u))
            prev  = u
        out.append(sent)
    return out

# ---- mode: sparse_dict (dict[int, dict[int, int]]) -----------------------
# Wire format per parent:
#   zigzag-varint  parent_key_delta
#   varint         n_child
#   n_child x zigzag-varint  child_key_deltas  (sorted, delta from 0 each parent)
#   n_child x varint         child_values      (unsigned — counts are always >= 0)

def _enc_sparse_dict(d: Dict[int, Dict[int, int]]) -> bytes:
    blob = bytearray()
    parent_keys = sorted(d.keys())
    blob.extend(_enc_varint(len(parent_keys)))
    last_parent = 0
    for pk in parent_keys:
        blob.extend(_enc_varint(_zz_enc(pk - last_parent)))
        last_parent = pk
        children   = d[pk]
        child_keys = sorted(children.keys())
        blob.extend(_enc_varint(len(child_keys)))
        last_child = 0
        for ck in child_keys:
            blob.extend(_enc_varint(_zz_enc(ck - last_child)))
            last_child = ck
        # values are unsigned counts — no zigzag needed
        for ck in child_keys:
            blob.extend(_enc_varint(children[ck]))
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_sparse_dict(blob: bytes) -> Dict[int, Dict[int, int]]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("sparse_dict: CRC mismatch")
    it     = iter(blob[:-4])
    n_par  = _dec_varint(it)
    out: Dict[int, Dict[int, int]] = {}
    last_parent = 0
    for _ in range(n_par):
        pk          = last_parent + _zz_dec(_dec_varint(it))
        last_parent = pk
        n_child     = _dec_varint(it)
        child_keys: List[int] = []
        last_child  = 0
        for _ in range(n_child):
            ck         = last_child + _zz_dec(_dec_varint(it))
            last_child = ck
            child_keys.append(ck)
        # values are unsigned counts
        values = [_dec_varint(it) for _ in range(n_child)]
        out[pk] = dict(zip(child_keys, values))
    return out

# ---- mode: sparse_dict_pos (dict[str, dict[int, int]]) -------------------

def _enc_sparse_dict_pos(d: Dict[str, Dict[int, int]]) -> bytes:
    int_keyed = {TAG_TO_ID.get(k, TAG_TO_ID["X"]): v for k, v in d.items()}
    return _enc_sparse_dict(int_keyed)

def _dec_sparse_dict_pos(blob: bytes) -> Dict[str, Dict[int, int]]:
    int_keyed = _dec_sparse_dict(blob)
    return {ID_TO_TAG.get(k, "X"): v for k, v in int_keyed.items()}

# ---- mode: pos_freq (dict[str, int]) ------------------------------------

def _enc_pos_freq(d: Dict[str, int]) -> bytes:
    blob = bytearray()
    blob.extend(_enc_varint(len(d)))
    for tag in UPOS_TAGS:
        if tag in d:
            blob.extend(_enc_varint(TAG_TO_ID[tag]))
            blob.extend(_enc_varint(d[tag]))
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)

def _dec_pos_freq(blob: bytes) -> Dict[str, int]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("pos_freq: CRC mismatch")
    it  = iter(blob[:-4])
    n   = _dec_varint(it)
    out: Dict[str, int] = {}
    for _ in range(n):
        tag_id = _dec_varint(it)
        count  = _dec_varint(it)
        out[ID_TO_TAG.get(tag_id, "X")] = count
    return out

# ---- mode: symbol_table (dict[str, str]) ---------------------------------
# Keys are §E{n} (entity) or §R{n} (relation).
# Values are plain surface strings (e.g. "Ishmael", "however").
#
# Wire format (before CRC):
#   varint  n_entries
#   for each entry:
#     byte    key_type   (0 = §E, 1 = §R)
#     varint  key_index  (the {n} in §E{n})
#     varint  surface_len  (byte length of UTF-8 surface)
#     bytes   surface_utf8

def _enc_symbol_table(d: Dict[str, str]) -> bytes:
    blob = bytearray()
    entries: List[Tuple[int, int, str]] = []
    for key, surface in d.items():
        if key.startswith("\u00a7E"):
            try:
                entries.append((0, int(key[2:]), surface))
            except ValueError:
                pass
        elif key.startswith("\u00a7R"):
            try:
                entries.append((1, int(key[2:]), surface))
            except ValueError:
                pass
    entries.sort()
    blob.extend(_enc_varint(len(entries)))
    for key_type, key_index, surface in entries:
        surface_bytes = surface.encode("utf-8")
        blob.append(key_type)
        blob.extend(_enc_varint(key_index))
        blob.extend(_enc_varint(len(surface_bytes)))
        blob.extend(surface_bytes)
    crc = zlib.crc32(bytes(blob)) & 0xFFFFFFFF
    blob.extend(struct.pack("<I", crc))
    return bytes(blob)


def _dec_symbol_table(blob: bytes) -> Dict[str, str]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("symbol_table: CRC mismatch")
    it  = iter(blob[:-4])
    n   = _dec_varint(it)
    out: Dict[str, str] = {}
    for _ in range(n):
        key_type  = next(it)
        key_index = _dec_varint(it)
        surf_len  = _dec_varint(it)
        surf_bytes = bytes(next(it) for _ in range(surf_len))
        surface   = surf_bytes.decode("utf-8")
        prefix    = "\u00a7E" if key_type == 0 else "\u00a7R"
        out[f"{prefix}{key_index}"] = surface
    return out

# ---- mode: model_weights (list[float], exactly 3 elements) --------------
# Stored as three IEEE-754 single-precision (float32) values + CRC.
# This gives ~7 significant digits, far better than the old uint16 scheme
# which had precision of only 0.0001 and derived w2 = 1 - w0 - w1 (lossy).

def _enc_model_weights(weights: List[float]) -> bytes:
    w = (list(weights) + [0.0, 0.0, 0.0])[:3]
    blob = struct.pack("<fff", w[0], w[1], w[2])
    crc  = zlib.crc32(blob) & 0xFFFFFFFF
    return blob + struct.pack("<I", crc)

def _dec_model_weights(blob: bytes) -> List[float]:
    stored   = struct.unpack("<I", blob[-4:])[0]
    computed = zlib.crc32(blob[:-4]) & 0xFFFFFFFF
    if stored != computed:
        raise ValueError("model_weights: CRC mismatch")
    w0, w1, w2 = struct.unpack("<fff", blob[:12])
    return [float(w0), float(w1), float(w2)]

# ---------------------------------------------------------------------------
# Envelope helpers
# ---------------------------------------------------------------------------

def _pack_field(field_id: int, blob: bytes) -> bytes:
    """1-byte id + 3-byte LE length + blob."""
    n = len(blob)
    if n > 0xFFFFFF:
        raise ValueError(f"Field {field_id} blob too large: {n} bytes")
    header = bytes([field_id]) + struct.pack("<I", n)[:3]
    return header + blob

def _parse_fields(data: bytes) -> Dict[int, bytes]:
    """Parse the field stream after the 5-byte file header."""
    fields: Dict[int, bytes] = {}
    i = 0
    while i < len(data):
        if i + 4 > len(data):
            break
        field_id = data[i]
        n = struct.unpack("<I", data[i+1:i+4] + b"\x00")[0]
        i += 4
        fields[field_id] = data[i:i+n]
        i += n
    return fields

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_metadata(meta: Dict[str, Any]) -> bytes:
    """Encode the full metadata dict to the LEXI binary envelope."""
    parts = bytearray(MAGIC + bytes([VERSION]))

    def _field(fid: int, blob: bytes) -> None:
        parts.extend(_pack_field(fid, blob))

    # Raw bitstreams
    _field(FIELD_COMPRESSED_BITSTREAM, bytes(meta["compressed_bitstream"]))
    _field(FIELD_POS_DELTAS_BITSTREAM, bytes(meta["pos_deltas_bitstream"]))

    # Symbol table
    _field(FIELD_SYMBOL_TABLE, _enc_symbol_table(meta.get("symbol_table", {})))

    # pos_deltas_counts: dict[int, int]
    pdc = {int(k): int(v) for k, v in meta.get("pos_deltas_counts", {}).items()}
    _field(FIELD_POS_DELTAS_COUNTS, _enc_flat_dict(pdc))

    # Scalars
    _field(FIELD_POS_DELTAS_COUNT,   _enc_scalar(int(meta.get("pos_deltas_count", 0))))
    _field(FIELD_NUM_SYMBOLS,        _enc_scalar(int(meta.get("num_symbols", 0))))
    _field(FIELD_NUM_CHAR_CLASSES,   _enc_scalar(int(meta.get("num_char_classes", 7))))

    # Flat uint arrays
    _field(FIELD_SENTENCE_CHAR_COUNTS, _enc_flat_uint([int(x) for x in meta.get("sentence_char_counts", [])]))
    _field(FIELD_POS_N_TAGS,           _enc_flat_uint([int(x) for x in meta.get("pos_n_tags", [])]))
    _field(FIELD_CHAR_VOCAB,           _enc_flat_uint([int(x) for x in meta.get("char_vocab", [])]))
    _field(FIELD_MORPH_VOCAB,          _enc_flat_uint([int(x) for x in meta.get("morph_vocab", [])]))

    # Float nested (pos_huffman_bits)
    phb: List[List[float]] = [[float(x)] for x in meta.get("pos_huffman_bits", [])]
    _field(FIELD_POS_HUFFMAN_BITS, _enc_float_nested(phb))

    # POS tags
    _field(FIELD_POS_TAGS, _enc_pos(meta.get("pos_tags", [])))

    # Nested int arrays
    _field(FIELD_MORPH_CODES,  _enc_int_nested([[int(x) for x in s] for s in meta.get("morph_codes",  [])]))
    _field(FIELD_ROOT_LENGTHS, _enc_int_nested([[int(x) for x in s] for s in meta.get("root_lengths", [])]))

    # Model weights — all 3 stored as float32
    _field(FIELD_MODEL_WEIGHTS, _enc_model_weights(list(meta.get("model_weights", [1/3, 1/3, 1/3]))))

    # Sparse dicts (context tables)
    char_ctx  = {int(k): {int(ck): int(cv) for ck, cv in v.items()} for k, v in meta.get("char_context",  {}).items()}
    morph_ctx = {int(k): {int(ck): int(cv) for ck, cv in v.items()} for k, v in meta.get("morph_context", {}).items()}
    struct_ctx: Dict[str, Dict[int, int]] = {str(k): {int(ck): int(cv) for ck, cv in v.items()} for k, v in meta.get("struct_context", {}).items()}

    _field(FIELD_CHAR_CONTEXT,   _enc_sparse_dict(char_ctx))
    _field(FIELD_MORPH_CONTEXT,  _enc_sparse_dict(morph_ctx))
    _field(FIELD_STRUCT_CONTEXT, _enc_sparse_dict_pos(struct_ctx))

    # pos_vocab
    pv_freq = {tag: 1 for tag in meta.get("pos_vocab", []) if tag in TAG_TO_ID}
    _field(FIELD_POS_VOCAB, _enc_pos_freq(pv_freq))

    # pos_freq_table
    pft = {str(k): int(v) for k, v in meta.get("pos_freq_table", {}).items()}
    _field(FIELD_POS_FREQ_TABLE, _enc_pos_freq(pft))

    return bytes(parts)


def decode_metadata(data: bytes) -> Dict[str, Any]:
    """Decode the LEXI binary envelope back to a metadata dict."""
    if data[:4] != MAGIC:
        raise ValueError("Not a LEXI file (bad magic)")
    version = data[4]
    if version != VERSION:
        raise ValueError(f"Unsupported LEXI version: {version}")

    fields = _parse_fields(data[5:])

    def _get(fid: int) -> bytes:
        return fields.get(fid, b"")

    meta: Dict[str, Any] = {}

    meta["compressed_bitstream"] = _dec_raw(_get(FIELD_COMPRESSED_BITSTREAM))
    meta["pos_deltas_bitstream"]  = _dec_raw(_get(FIELD_POS_DELTAS_BITSTREAM))

    st_blob = _get(FIELD_SYMBOL_TABLE)
    meta["symbol_table"] = _dec_symbol_table(st_blob) if st_blob else {}

    pdc_blob = _get(FIELD_POS_DELTAS_COUNTS)
    meta["pos_deltas_counts"] = _dec_flat_dict(pdc_blob) if pdc_blob else {}

    meta["pos_deltas_count"] = _dec_scalar(_get(FIELD_POS_DELTAS_COUNT)) if _get(FIELD_POS_DELTAS_COUNT) else 0
    meta["num_symbols"]      = _dec_scalar(_get(FIELD_NUM_SYMBOLS))      if _get(FIELD_NUM_SYMBOLS)      else 0
    meta["num_char_classes"] = _dec_scalar(_get(FIELD_NUM_CHAR_CLASSES)) if _get(FIELD_NUM_CHAR_CLASSES) else 7

    meta["sentence_char_counts"] = _dec_flat_uint(_get(FIELD_SENTENCE_CHAR_COUNTS)) if _get(FIELD_SENTENCE_CHAR_COUNTS) else []
    meta["pos_n_tags"]           = _dec_flat_uint(_get(FIELD_POS_N_TAGS))           if _get(FIELD_POS_N_TAGS)           else []
    meta["char_vocab"]           = _dec_flat_uint(_get(FIELD_CHAR_VOCAB))           if _get(FIELD_CHAR_VOCAB)           else []
    meta["morph_vocab"]          = _dec_flat_uint(_get(FIELD_MORPH_VOCAB))          if _get(FIELD_MORPH_VOCAB)          else []

    phb_blob = _get(FIELD_POS_HUFFMAN_BITS)
    if phb_blob:
        nested = _dec_float_nested(phb_blob)
        meta["pos_huffman_bits"] = [row[0] if row else 0.0 for row in nested]
    else:
        meta["pos_huffman_bits"] = []

    meta["pos_tags"]     = _dec_pos(_get(FIELD_POS_TAGS))             if _get(FIELD_POS_TAGS)     else []
    meta["morph_codes"]  = _dec_int_nested(_get(FIELD_MORPH_CODES))  if _get(FIELD_MORPH_CODES)  else []
    meta["root_lengths"] = _dec_int_nested(_get(FIELD_ROOT_LENGTHS)) if _get(FIELD_ROOT_LENGTHS) else []

    mw_blob = _get(FIELD_MODEL_WEIGHTS)
    meta["model_weights"] = _dec_model_weights(mw_blob) if mw_blob else [1/3, 1/3, 1/3]

    meta["char_context"]   = _dec_sparse_dict(_get(FIELD_CHAR_CONTEXT))       if _get(FIELD_CHAR_CONTEXT)   else {}
    meta["morph_context"]  = _dec_sparse_dict(_get(FIELD_MORPH_CONTEXT))      if _get(FIELD_MORPH_CONTEXT)  else {}
    meta["struct_context"] = _dec_sparse_dict_pos(_get(FIELD_STRUCT_CONTEXT)) if _get(FIELD_STRUCT_CONTEXT) else {}

    pv_blob = _get(FIELD_POS_VOCAB)
    meta["pos_vocab"] = list(_dec_pos_freq(pv_blob).keys()) if pv_blob else []

    pft_blob = _get(FIELD_POS_FREQ_TABLE)
    meta["pos_freq_table"] = _dec_pos_freq(pft_blob) if pft_blob else {}

    return meta


def is_lexi_file(data: bytes) -> bool:
    """Return True if data starts with the LEXI magic bytes."""
    return data[:4] == MAGIC
