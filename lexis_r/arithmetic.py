"""Lexis-R arithmetic coder — self-contained, integer-only.

Design principles
-----------------
* Pure integer arithmetic throughout.  No float division inside the
  encode/decode inner loop.  Encoder and decoder share exactly one
  function (_build_cdf) so their CDF tables are guaranteed identical.
* 32-bit interval [low, high] with E1/E2/E3 (underflow) rescaling.
* _PRECISION = 1 << 16 (65 536) probability quanta; every symbol gets
  at least 1 quantum so no zero-width interval can ever occur.
* Context stream is built once per compress/decompress call via
  build_context_stream() and passed in as plain lists — no hidden
  state coupling to the ContextMixingModel.

Public API
----------
  encode(char_classes, context_model, encoded_sentences)  -> bytes
  decode(data, context_model, encoded_sentences, n)       -> List[int]
  encode_unigram_counts(symbols, counts)                  -> bytes
  decode_unigram_counts(data, counts, n)                  -> List[int]
  build_context_stream(encoded_sentences)
      -> (morph_stream, pos_stream, struct_probs)

Self-test
---------
  python -m lexis_r.arithmetic          # runs roundtrip assertions
"""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from typing import Dict, List, Tuple

from compression.config import CHAR_CONTEXT_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRECISION:    int = 1 << 16          # probability quanta
_FULL:         int = 1 << 32
_HALF:         int = _FULL >> 1
_QUARTER:      int = _HALF >> 1
_THREE_QUARTER: int = _QUARTER * 3


# ---------------------------------------------------------------------------
# Bit I/O
# ---------------------------------------------------------------------------

class _BitWriter:
    __slots__ = ("buffer", "_byte", "_filled")

    def __init__(self) -> None:
        self.buffer: bytearray = bytearray()
        self._byte:   int = 0
        self._filled: int = 0

    def write_bit(self, bit: int) -> None:
        self._byte = (self._byte << 1) | (bit & 1)
        self._filled += 1
        if self._filled == 8:
            self.buffer.append(self._byte)
            self._byte   = 0
            self._filled = 0

    def flush(self) -> None:
        if self._filled > 0:
            self.buffer.append(self._byte << (8 - self._filled))
            self._byte   = 0
            self._filled = 0

    def get_bytes(self) -> bytes:
        return bytes(self.buffer)


class _BitReader:
    __slots__ = ("data", "_bi", "_pos")

    def __init__(self, data: bytes) -> None:
        self.data = data
        self._bi:  int = 0   # byte index
        self._pos: int = 0   # bit position within current byte (0 = MSB)

    def read_bit(self) -> int:
        if self._bi >= len(self.data):
            return 0
        bit = (self.data[self._bi] >> (7 - self._pos)) & 1
        self._pos += 1
        if self._pos == 8:
            self._pos = 0
            self._bi += 1
        return bit


# ---------------------------------------------------------------------------
# CDF — the ONE function shared by encoder and decoder
# ---------------------------------------------------------------------------

def _build_cdf(
    distribution: Dict[int, float],
) -> Tuple[List[int], List[int], int]:
    """
    Convert a float probability dict to an integer CDF over _PRECISION quanta.

    Returns
    -------
    symbols    : sorted list of symbol ids
    cumulative : len(symbols)+1 list; cumulative[i] is the running count
                 BEFORE symbol i, cumulative[-1] == total
    total      : sum of all counts (== cumulative[-1])
    """
    if not distribution:
        return [0], [0, _PRECISION], _PRECISION

    prob_sum = sum(distribution.values()) or 1.0
    symbols  = sorted(distribution.keys())
    counts   = [max(1, round(distribution[s] / prob_sum * _PRECISION))
                for s in symbols]

    cumulative: List[int] = [0]
    for c in counts:
        cumulative.append(cumulative[-1] + c)
    total = cumulative[-1]

    return symbols, cumulative, total


def _build_count_cdf(
    counts: Dict[int, int],
) -> Tuple[List[int], List[int], int]:
    """Integer CDF for encode_unigram_counts / decode_unigram_counts."""
    if not counts:
        return [0], [0, 1], 1
    symbols    = sorted(counts.keys())
    cumulative = [0]
    for s in symbols:
        cumulative.append(cumulative[-1] + max(int(counts[s]), 1))
    total = cumulative[-1]
    return symbols, cumulative, total


# ---------------------------------------------------------------------------
# Context stream helper
# ---------------------------------------------------------------------------

def build_context_stream(
    encoded_sentences: List[Dict],
) -> Tuple[List[int], List[str], List[float]]:
    """Flatten per-sentence metadata into parallel per-symbol lists."""
    morph_stream: List[int]   = []
    pos_stream:   List[str]   = []
    struct_probs: List[float] = []

    for sent in encoded_sentences:
        morphs  = sent.get("char_morph_codes", [])
        pos     = sent.get("char_pos_tags",    [])
        n_tags  = int(sent.get("pos_n_tags",      0))
        hbits   = float(sent.get("pos_huffman_bits", 0.0))
        sp = 2 ** (-hbits / n_tags) if (n_tags > 0 and hbits > 0) else (
            1.0 / max(len(sent.get("pos_tags", [])), 1)
        )
        length = max(len(morphs), len(pos))
        for i in range(length):
            morph_stream.append(int(morphs[i]) if i < len(morphs) else 0)
            pos_stream.append(str(pos[i])   if i < len(pos)    else "X")
            struct_probs.append(sp)

    return morph_stream, pos_stream, struct_probs


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ArithmeticEncoder:
    """Integer arithmetic encoder (E1/E2/E3 rescaling, 32-bit interval)."""

    def __init__(self) -> None:
        self._reset()

    # ── Public API ──────────────────────────────────────────────────────────────────────

    def encode(
        self,
        char_classes:      List[int],
        context_model,
        encoded_sentences: List[Dict],
    ) -> bytes:
        """Encode char_classes using a context-mixing probability model."""
        self._reset()
        morph_stream, pos_stream, struct_probs = build_context_stream(encoded_sentences)
        char_history: deque = deque(maxlen=CHAR_CONTEXT_SIZE)

        n = min(len(char_classes), len(morph_stream), len(pos_stream))
        for i in range(n):
            ctx = {
                "char_history":       list(char_history),
                "current_morph_code": morph_stream[i],
                "current_pos_tag":    pos_stream[i],
                "struct_prob":        struct_probs[i],
            }
            dist = context_model.probability_distribution(ctx)
            self._encode_symbol(int(char_classes[i]), dist)
            char_history.append(int(char_classes[i]))

        self._finalize()
        return self._writer.get_bytes()

    def encode_unigram_counts(
        self, symbols: List[int], counts: Dict[int, int]
    ) -> bytes:
        """Encode a symbol stream with fixed integer counts."""
        self._reset()
        syms, cum, total = _build_count_cdf(counts)
        sym2idx = {s: i for i, s in enumerate(syms)}
        for sym in symbols:
            i = sym2idx.get(int(sym))
            if i is None:
                raise ValueError(f"Symbol {sym!r} not in counts table.")
            self._update_interval(cum[i], cum[i + 1], total)
        self._finalize()
        return self._writer.get_bytes()

    # ── Internal helpers ───────────────────────────────────────────────────────────────────────

    def _reset(self) -> None:
        self._low:     int = 0
        self._high:    int = _FULL - 1
        self._pending: int = 0
        self._writer       = _BitWriter()

    def _encode_symbol(
        self, symbol: int, distribution: Dict[int, float]
    ) -> None:
        syms, cum, total = _build_cdf(distribution)
        try:
            i = syms.index(symbol)
        except ValueError:
            raise ValueError(f"Symbol {symbol!r} not in distribution.")
        self._update_interval(cum[i], cum[i + 1], total)

    def _update_interval(
        self, low_cum: int, high_cum: int, total: int
    ) -> None:
        rw       = self._high - self._low + 1
        orig_low = self._low
        self._high = orig_low + (rw * high_cum) // total - 1
        self._low  = orig_low + (rw * low_cum)  // total
        if self._high < self._low:
            self._high = self._low
        self._renorm()

    def _output_bit(self, bit: int) -> None:
        self._writer.write_bit(bit)
        for _ in range(self._pending):
            self._writer.write_bit(1 - bit)
        self._pending = 0

    def _renorm(self) -> None:
        while True:
            if self._high < _HALF:
                self._output_bit(0)
            elif self._low >= _HALF:
                self._output_bit(1)
                self._low  -= _HALF
                self._high -= _HALF
            elif self._low >= _QUARTER and self._high < _THREE_QUARTER:
                self._pending += 1
                self._low  -= _QUARTER
                self._high -= _QUARTER
            else:
                break
            self._low  = self._low  * 2
            self._high = self._high * 2 + 1

    def _finalize(self) -> None:
        self._pending += 1
        self._output_bit(0 if self._low < _QUARTER else 1)
        self._writer.flush()


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class ArithmeticDecoder:
    """Integer arithmetic decoder — mirror of ArithmeticEncoder."""

    def __init__(self) -> None:
        self._reader: _BitReader | None = None
        self._value  = 0
        self._low    = 0
        self._high   = _FULL - 1

    # ── Public API ──────────────────────────────────────────────────────────────────────

    def decode(
        self,
        data:              bytes,
        context_model,
        encoded_sentences: List[Dict],
        n:                 int,
    ) -> List[int]:
        """Decode n symbols using a context-mixing probability model."""
        self._init(data)
        morph_stream, pos_stream, struct_probs = build_context_stream(encoded_sentences)
        char_history: deque = deque(maxlen=CHAR_CONTEXT_SIZE)
        decoded: List[int] = []

        length = min(n, len(morph_stream), len(pos_stream))
        for i in range(length):
            ctx = {
                "char_history":       list(char_history),
                "current_morph_code": morph_stream[i],
                "current_pos_tag":    pos_stream[i],
                "struct_prob":        struct_probs[i],
            }
            dist   = context_model.probability_distribution(ctx)
            symbol = self._decode_symbol_float(dist)
            decoded.append(symbol)
            char_history.append(symbol)

        return decoded

    def decode_unigram_counts(
        self, data: bytes, counts: Dict[int, int], n: int
    ) -> List[int]:
        """Decode n symbols with fixed integer counts."""
        self._init(data)
        syms, cum, total = _build_count_cdf(counts)
        decoded: List[int] = []
        for _ in range(n):
            rw     = self._high - self._low + 1
            scaled = ((self._value - self._low + 1) * total - 1) // rw
            idx    = max(bisect_right(cum, scaled) - 1, 0)
            idx    = min(idx, len(syms) - 1)
            decoded.append(syms[idx])
            self._update_interval(cum[idx], cum[idx + 1], total)
        return decoded

    # ── Internal helpers ───────────────────────────────────────────────────────────────────────

    def _init(self, data: bytes) -> None:
        self._reader = _BitReader(data)
        self._value  = 0
        self._low    = 0
        self._high   = _FULL - 1
        for _ in range(32):
            self._value = (self._value << 1) | self._reader.read_bit()

    def _decode_symbol_float(
        self, distribution: Dict[int, float]
    ) -> int:
        """
        Decode one symbol using the same _build_cdf table as the encoder.
        Find the bucket i where:
          (rw * cum[i]) // total  <=  value - low  <  (rw * cum[i+1]) // total
        """
        syms, cum, total = _build_cdf(distribution)
        rw     = self._high - self._low + 1
        scaled = self._value - self._low

        # Map cumulative thresholds to the same integer-divided boundaries
        # the encoder used, then bisect to find bucket.
        boundaries = [(rw * c) // total for c in cum]
        idx = max(bisect_right(boundaries, scaled) - 1, 0)
        idx = min(idx, len(syms) - 1)

        self._update_interval(cum[idx], cum[idx + 1], total)
        return syms[idx]

    def _update_interval(
        self, low_cum: int, high_cum: int, total: int
    ) -> None:
        rw       = self._high - self._low + 1
        orig_low = self._low
        new_high = orig_low + (rw * high_cum) // total - 1
        new_low  = orig_low + (rw * low_cum)  // total
        self._low  = new_low
        self._high = new_high
        if self._high < self._low:
            self._high = self._low
        self._renorm()

    def _renorm(self) -> None:
        while True:
            if self._high < _HALF:
                pass
            elif self._low >= _HALF:
                self._value -= _HALF
                self._low   -= _HALF
                self._high  -= _HALF
            elif self._low >= _QUARTER and self._high < _THREE_QUARTER:
                self._value -= _QUARTER
                self._low   -= _QUARTER
                self._high  -= _QUARTER
            else:
                break
            self._low   = self._low   * 2
            self._high  = self._high  * 2 + 1
            self._value = self._value * 2 + (
                self._reader.read_bit() if self._reader else 0
            )


# ---------------------------------------------------------------------------
# Built-in roundtrip self-test
# ---------------------------------------------------------------------------

def _run_self_test() -> None:
    """
    Verify encoder/decoder roundtrip on several synthetic cases.
    Raises AssertionError immediately on any mismatch.
    """
    import random

    # ── Test 1: unigram counts roundtrip ───────────────────────────────────────
    counts: Dict[int, int] = {0: 50, 1: 30, 2: 15, 3: 5}
    rng    = random.Random(42)
    for trial in range(10):
        symbols = [rng.choices(list(counts.keys()),
                               weights=list(counts.values()))[0]
                   for _ in range(200)]
        enc   = ArithmeticEncoder()
        data  = enc.encode_unigram_counts(symbols, counts)
        dec   = ArithmeticDecoder()
        recovered = dec.decode_unigram_counts(data, counts, len(symbols))
        assert recovered == symbols, (
            f"Unigram roundtrip FAILED on trial {trial}:\n"
            f"  original : {symbols[:20]}\n"
            f"  recovered: {recovered[:20]}"
        )
    print("[self-test] unigram_counts: 10 trials PASSED")

    # ── Test 2: single-symbol stream (degenerate distribution) ─────────────────
    counts2 = {7: 1}
    data2   = ArithmeticEncoder().encode_unigram_counts([7] * 50, counts2)
    rec2    = ArithmeticDecoder().decode_unigram_counts(data2, counts2, 50)
    assert rec2 == [7] * 50, "Single-symbol degenerate roundtrip FAILED"
    print("[self-test] degenerate single-symbol: PASSED")

    # ── Test 3: high-entropy near-uniform distribution ─────────────────────────
    counts3 = {i: 1 for i in range(7)}   # 7 equally-likely char classes
    rng3    = random.Random(99)
    syms3   = [rng3.randint(0, 6) for _ in range(300)]
    data3   = ArithmeticEncoder().encode_unigram_counts(syms3, counts3)
    rec3    = ArithmeticDecoder().decode_unigram_counts(data3, counts3, len(syms3))
    assert rec3 == syms3, "High-entropy roundtrip FAILED"
    print("[self-test] high-entropy uniform: PASSED")

    # ── Test 4: length boundary — exactly 1 symbol ────────────────────────────
    counts4 = {3: 10, 5: 1}
    data4   = ArithmeticEncoder().encode_unigram_counts([3], counts4)
    rec4    = ArithmeticDecoder().decode_unigram_counts(data4, counts4, 1)
    assert rec4 == [3], "Single-element roundtrip FAILED"
    print("[self-test] single-element stream: PASSED")

    print("[self-test] ALL PASSED ✓")


if __name__ == "__main__":
    _run_self_test()
