"""Canonical Huffman encoder/decoder for the Lexis-R pos-delta stream.

The encoder:
  1. Counts symbol frequencies.
  2. Builds a Huffman tree and extracts code lengths.
  3. Constructs canonical codes from lengths (sorted by length then symbol).
  4. Serialises only the (symbol, length) pairs — not the full codes.
  5. Packs the bitstream.

The decoder:
  1. Deserialises (symbol, length) pairs.
  2. Reconstructs canonical codes from lengths.
  3. Decodes the bitstream.

Zigzag pre-processing maps signed integers to non-negative before
coding so VLQ and Huffman work on a non-negative alphabet.

Wire format for the code table (packed into bytes):
  [1B n_symbols]
  for each (zigzag_symbol, code_length) pair:
    vlq(zigzag_symbol)  — base-128 VLQ
    [1B code_length]    — max depth 32 is safe

Bitstream: raw bits packed MSB-first into bytes, no length prefix
(caller stores symbol count separately).
"""

from __future__ import annotations

import heapq
from collections import Counter
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Zigzag helpers
# ---------------------------------------------------------------------------

def zigzag_encode(n: int) -> int:
    return (n << 1) ^ (n >> 63)


def zigzag_decode(n: int) -> int:
    return (n >> 1) ^ -(n & 1)


# ---------------------------------------------------------------------------
# VLQ helpers (base-128, duplicated here to keep module self-contained)
# ---------------------------------------------------------------------------

def _vlq_enc(v: int) -> bytes:
    groups: List[int] = []
    while True:
        groups.append(v & 0x7F)
        v >>= 7
        if v == 0:
            break
    groups.reverse()
    return bytes([(g | 0x80) if i < len(groups) - 1 else g
                  for i, g in enumerate(groups)])


def _vlq_dec(data: bytes, off: int) -> Tuple[int, int]:
    v = 0
    while True:
        b = data[off]; off += 1
        v = (v << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return v, off


# ---------------------------------------------------------------------------
# Huffman tree
# ---------------------------------------------------------------------------

def _build_code_lengths(freq: Dict[int, int]) -> Dict[int, int]:
    """Return {symbol: code_length} from frequency table."""
    if len(freq) == 1:
        sym = next(iter(freq))
        return {sym: 1}

    heap: List[Tuple[int, int, object]] = []
    for i, (sym, cnt) in enumerate(freq.items()):
        heapq.heappush(heap, (cnt, i, sym))

    counter = len(freq)
    while len(heap) > 1:
        c1, _, n1 = heapq.heappop(heap)
        c2, _, n2 = heapq.heappop(heap)
        heapq.heappush(heap, (c1 + c2, counter, (n1, n2)))
        counter += 1

    lengths: Dict[int, int] = {}

    def _assign(node, depth: int) -> None:
        if isinstance(node, int):
            lengths[node] = max(depth, 1)
            return
        _assign(node[0], depth + 1)
        _assign(node[1], depth + 1)

    _assign(heap[0][2], 0)
    return lengths


def _canonical_codes(lengths: Dict[int, int]) -> Dict[int, str]:
    """Build canonical Huffman codes from {symbol: length}."""
    # Sort by (length, symbol)
    sorted_syms = sorted(lengths.keys(), key=lambda s: (lengths[s], s))
    codes: Dict[int, str] = {}
    code = 0
    prev_len = 0
    for sym in sorted_syms:
        length = lengths[sym]
        code <<= (length - prev_len)
        codes[sym] = format(code, f"0{length}b")
        code += 1
        prev_len = length
    return codes


# ---------------------------------------------------------------------------
# Serialise / deserialise code table
# ---------------------------------------------------------------------------

def pack_huffman_table(lengths: Dict[int, int]) -> bytes:
    """Serialise {zigzag_symbol: length} pairs."""
    items = sorted(lengths.items(), key=lambda x: (x[1], x[0]))
    out = bytearray([len(items)])
    for sym, length in items:
        out += _vlq_enc(sym)
        out.append(length & 0xFF)
    return bytes(out)


def unpack_huffman_table(data: bytes) -> Dict[int, int]:
    """Deserialise {zigzag_symbol: length} pairs."""
    n = data[0]
    offset = 1
    lengths: Dict[int, int] = {}
    for _ in range(n):
        sym, offset = _vlq_dec(data, offset)
        length = data[offset]; offset += 1
        lengths[sym] = length
    return lengths


# ---------------------------------------------------------------------------
# Bit I/O
# ---------------------------------------------------------------------------

class _BitWriter:
    def __init__(self) -> None:
        self._buf = bytearray()
        self._cur = 0
        self._fill = 0

    def write_bits(self, bits: str) -> None:
        for b in bits:
            self._cur = (self._cur << 1) | int(b)
            self._fill += 1
            if self._fill == 8:
                self._buf.append(self._cur)
                self._cur = 0
                self._fill = 0

    def flush(self) -> bytes:
        if self._fill:
            self._buf.append(self._cur << (8 - self._fill))
        return bytes(self._buf)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte = 0
        self._bit = 0

    def read_bit(self) -> int | None:
        if self._byte >= len(self._data):
            return None
        bit = (self._data[self._byte] >> (7 - self._bit)) & 1
        self._bit += 1
        if self._bit == 8:
            self._bit = 0
            self._byte += 1
        return bit


# ---------------------------------------------------------------------------
# Public encode / decode
# ---------------------------------------------------------------------------

def encode(
    symbols: List[int],
) -> Tuple[bytes, bytes]:
    """
    Zigzag + canonical Huffman encode a list of signed integers.

    Returns (table_bytes, bitstream_bytes).
    """
    zz = [zigzag_encode(s) for s in symbols]
    freq = Counter(zz)
    lengths = _build_code_lengths(dict(freq))
    codes   = _canonical_codes(lengths)

    writer = _BitWriter()
    for z in zz:
        writer.write_bits(codes[z])
    bitstream = writer.flush()
    table     = pack_huffman_table(lengths)
    return table, bitstream


def decode(
    table_bytes: bytes,
    bitstream_bytes: bytes,
    n_symbols: int,
) -> List[int]:
    """
    Decode n_symbols signed integers from a Huffman bitstream.
    Inverse of encode().
    """
    lengths = unpack_huffman_table(table_bytes)
    codes   = _canonical_codes(lengths)
    # Build reverse map: bitstring → original signed symbol
    rev: Dict[str, int] = {
        bits: zigzag_decode(sym) for sym, bits in codes.items()
    }
    # Build a prefix trie for efficient decoding
    reader  = _BitReader(bitstream_bytes)
    decoded: List[int] = []
    buf = ""
    while len(decoded) < n_symbols:
        bit = reader.read_bit()
        if bit is None:
            break
        buf += str(bit)
        if buf in rev:
            decoded.append(rev[buf])
            buf = ""
    return decoded
