"""
VLQ helpers for root_lengths storage.

Layout (bytes):
  [n_sentences : VLQ]
  for each sentence:
    [n_tokens : VLQ]
    for each token:
      [root_length : VLQ]

VLQ == base-128 big-endian (identical to protobuf varint).
No upper bound on root length — fixes the 4-bit clamping bug.
"""
from __future__ import annotations
from typing import List


def _vlq_encode(value: int) -> bytes:
    assert value >= 0, f"VLQ value must be non-negative, got {value}"
    groups: list[int] = []
    while True:
        groups.append(value & 0x7F)
        value >>= 7
        if value == 0:
            break
    groups.reverse()
    return bytes([(g | 0x80) if i < len(groups) - 1 else g
                  for i, g in enumerate(groups)])


def _vlq_decode_stream(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        b = data[offset]; offset += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, offset


def pack_root_lengths(sentences: List[List[int]]) -> bytes:
    """Encode nested root-length list to VLQ bytes."""
    out = bytearray()
    out += _vlq_encode(len(sentences))
    for sentence in sentences:
        out += _vlq_encode(len(sentence))
        for length in sentence:
            out += _vlq_encode(max(length, 0))
    return bytes(out)


def unpack_root_lengths(data: bytes) -> List[List[int]]:
    """Decode VLQ bytes back to nested root-length list."""
    offset = 0
    n_sentences, offset = _vlq_decode_stream(data, offset)
    sentences: List[List[int]] = []
    for _ in range(n_sentences):
        n_tokens, offset = _vlq_decode_stream(data, offset)
        lengths: List[int] = []
        for _ in range(n_tokens):
            length, offset = _vlq_decode_stream(data, offset)
            lengths.append(length)
        sentences.append(lengths)
    return sentences
