"""Arithmetic encoder/decoder for context-dependent compression."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from typing import Dict, List, Tuple

from compression.config import CHAR_CONTEXT_SIZE

# Fixed precision for float-probability -> integer-count conversion.
# Both encoder and decoder use this constant, so CDF tables are identical.
_PRECISION = 1 << 16  # 65536


# ---------------------------------------------------------------------------
# Bit I/O
# ---------------------------------------------------------------------------

class _BitWriter:
    def __init__(self) -> None:
        self.buffer = bytearray()
        self.current_byte = 0
        self.bits_filled = 0

    def write_bit(self, bit: int) -> None:
        self.current_byte = (self.current_byte << 1) | (1 if bit else 0)
        self.bits_filled += 1
        if self.bits_filled == 8:
            self.buffer.append(self.current_byte & 0xFF)
            self.current_byte = 0
            self.bits_filled = 0

    def flush(self) -> None:
        if self.bits_filled > 0:
            self.current_byte <<= 8 - self.bits_filled
            self.buffer.append(self.current_byte & 0xFF)
            self.current_byte = 0
            self.bits_filled = 0

    def get_bytes(self) -> bytes:
        return bytes(self.buffer)


class _BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.byte_index = 0
        self.bit_index = 0

    def read_bit(self) -> int:
        if self.byte_index >= len(self.data):
            return 0
        current_byte = self.data[self.byte_index]
        bit = (current_byte >> (7 - self.bit_index)) & 1
        self.bit_index += 1
        if self.bit_index == 8:
            self.bit_index = 0
            self.byte_index += 1
        return bit


# ---------------------------------------------------------------------------
# CDF helpers
# ---------------------------------------------------------------------------

def _build_cdf(
    distribution: Dict[int, float]
) -> Tuple[List[int], List[int], int]:
    """
    Convert a float probability distribution into an integer CDF.

    Returns (symbols, cumulative, total) where:
      - symbols[i] is the i-th symbol in sorted order
      - cumulative[i] is the cumulative count *before* symbol i
      - cumulative[i+1] is the cumulative count *after* symbol i
      - total = cumulative[-1]

    Each symbol gets at least 1 count unit, so no zero-width interval.
    The mapping is deterministic given only the distribution dict, so
    encoder and decoder always produce identical tables.
    """
    if not distribution:
        return [0], [0, _PRECISION], _PRECISION

    prob_sum = sum(distribution.values())
    if prob_sum <= 0:
        prob_sum = 1.0

    symbols = sorted(distribution.keys())
    counts = [max(1, round(distribution[s] / prob_sum * _PRECISION)) for s in symbols]

    cumulative = [0]
    for c in counts:
        cumulative.append(cumulative[-1] + c)
    total = cumulative[-1]

    return symbols, cumulative, total


def _build_cumulative_counts(
    counts: Dict[int, int],
) -> Tuple[List[int], List[int], int]:
    if not counts:
        return [0], [0, 1], 1
    symbols = sorted(counts.keys())
    cumulative = [0]
    for sym in symbols:
        # min 1 per symbol — zero-width intervals must never occur
        cumulative.append(cumulative[-1] + max(int(counts[sym]), 1))
    total = cumulative[-1]
    return symbols, cumulative, total


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class ArithmeticEncoder:
    """Encodes symbol streams into a compressed bitstream using arithmetic coding."""

    FULL = 1 << 32
    HALF = FULL >> 1
    QUARTER = HALF >> 1
    THREE_QUARTER = QUARTER * 3

    def __init__(self) -> None:
        self.low = 0
        self.high = (1 << 32) - 1
        self.pending = 0
        self.writer = _BitWriter()

    def encode(
        self,
        char_classes: List[int],
        context_model,
        context_data: Dict,
        encoded_sentences: List[Dict],
    ) -> bytes:
        """Encode char_classes stream into compressed bytes."""
        morph_stream, pos_stream, struct_probs = _build_context_stream(encoded_sentences)
        char_history: deque = deque(maxlen=CHAR_CONTEXT_SIZE)

        length = min(len(char_classes), len(morph_stream), len(pos_stream))
        for idx in range(length):
            symbol = int(char_classes[idx])
            context = {
                "char_history": list(char_history),
                "current_morph_code": morph_stream[idx],
                "current_pos_tag": pos_stream[idx],
                "struct_prob": struct_probs[idx],
            }
            distribution = context_model.probability_distribution(context)
            self._encode_symbol(symbol, distribution)
            char_history.append(symbol)

        self._finalize()
        return self.writer.get_bytes()

    def encode_unigram(
        self, symbols: List[int], distribution: Dict[int, float]
    ) -> bytes:
        """Encode a symbol stream using a fixed unigram distribution."""
        self._reset()
        for symbol in symbols:
            self._encode_symbol(int(symbol), distribution)
        self._finalize()
        return self.writer.get_bytes()

    def encode_unigram_counts(
        self, symbols: List[int], counts: Dict[int, int]
    ) -> bytes:
        """Encode a symbol stream using integer unigram counts."""
        self._reset()
        symbols_sorted, cumulative, total = _build_cumulative_counts(counts)
        symbol_to_index = {sym: idx for idx, sym in enumerate(symbols_sorted)}
        for symbol in symbols:
            idx = symbol_to_index.get(int(symbol))
            if idx is None:
                raise ValueError(f"Symbol {symbol} missing from counts.")
            low_sym  = cumulative[idx]
            high_sym = cumulative[idx + 1]
            range_width = self.high - self.low + 1
            orig_low  = self.low
            self.low  = orig_low + (range_width * low_sym)  // total
            self.high = orig_low + (range_width * high_sym) // total - 1
            if self.high < self.low:
                self.high = self.low
            self._renormalize()
        self._finalize()
        return self.writer.get_bytes()

    def _reset(self) -> None:
        self.low = 0
        self.high = (1 << 32) - 1
        self.pending = 0
        self.writer = _BitWriter()

    def _encode_symbol(self, symbol: int, distribution: Dict[int, float]) -> None:
        syms, cumulative, total = _build_cdf(distribution)
        if symbol not in syms:
            raise ValueError(f"Symbol {symbol} missing from distribution.")
        idx = syms.index(symbol)
        low_sym  = cumulative[idx]
        high_sym = cumulative[idx + 1]
        range_width = self.high - self.low + 1
        orig_low  = self.low
        self.high = orig_low + (range_width * high_sym) // total - 1
        self.low  = orig_low + (range_width * low_sym)  // total
        if self.high < self.low:
            self.high = self.low
        self._renormalize()

    def _output_bit(self, bit: int) -> None:
        self.writer.write_bit(bit)
        for _ in range(self.pending):
            self.writer.write_bit(1 - bit)
        self.pending = 0

    def _renormalize(self) -> None:
        while True:
            if self.high < self.HALF:
                self._output_bit(0)
            elif self.low >= self.HALF:
                self._output_bit(1)
                self.low  -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.pending += 1
                self.low  -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low  = self.low  * 2
            self.high = self.high * 2 + 1

    def _finalize(self) -> None:
        self.pending += 1
        if self.low < self.QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)
        self.writer.flush()


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class ArithmeticDecoder:
    """Decodes arithmetic-coded bitstreams back to symbol streams."""

    FULL = 1 << 32
    HALF = FULL >> 1
    QUARTER = HALF >> 1
    THREE_QUARTER = QUARTER * 3

    def __init__(self) -> None:
        self.reader: _BitReader | None = None
        self.value = 0
        self.low   = 0
        self.high  = (1 << 32) - 1

    def decode(
        self,
        compressed_bytes: bytes,
        context_model,
        encoded_sentences: List[Dict],
        num_symbols: int,
    ) -> List[int]:
        self._init_reader(compressed_bytes)
        morph_stream, pos_stream, struct_probs = _build_context_stream(encoded_sentences)
        char_history: deque = deque(maxlen=CHAR_CONTEXT_SIZE)
        decoded: List[int] = []

        length = min(num_symbols, len(morph_stream), len(pos_stream))
        for idx in range(length):
            context = {
                "char_history":       list(char_history),
                "current_morph_code": morph_stream[idx],
                "current_pos_tag":    pos_stream[idx],
                "struct_prob":        struct_probs[idx],
            }
            distribution = context_model.probability_distribution(context)
            symbol = self._decode_symbol(distribution)
            decoded.append(symbol)
            char_history.append(symbol)

        return decoded

    def decode_unigram(
        self, compressed_bytes: bytes, distribution: Dict[int, float], num_symbols: int
    ) -> List[int]:
        """Decode a symbol stream using a fixed unigram distribution."""
        self._init_reader(compressed_bytes)
        decoded: List[int] = []
        for _ in range(num_symbols):
            symbol = self._decode_symbol(distribution)
            decoded.append(symbol)
        return decoded

    def decode_unigram_counts(
        self, compressed_bytes: bytes, counts: Dict[int, int], num_symbols: int
    ) -> List[int]:
        """Decode a symbol stream using integer unigram counts."""
        self._init_reader(compressed_bytes)
        symbols_sorted, cumulative, total = _build_cumulative_counts(counts)
        decoded: List[int] = []
        for _ in range(num_symbols):
            range_width = self.high - self.low + 1
            scaled = ((self.value - self.low + 1) * total - 1) // range_width
            idx    = max(bisect_right(cumulative, scaled) - 1, 0)
            idx    = min(idx, len(symbols_sorted) - 1)
            symbol = symbols_sorted[idx]
            low_sym  = cumulative[idx]
            high_sym = cumulative[idx + 1]
            orig_low  = self.low
            self.low  = orig_low + (range_width * low_sym)  // total
            self.high = orig_low + (range_width * high_sym) // total - 1
            if self.high < self.low:
                self.high = self.low
            self._renormalize()
            decoded.append(symbol)
        return decoded

    def _init_reader(self, data: bytes) -> None:
        self.reader = _BitReader(data)
        self.value  = 0
        self.low    = 0
        self.high   = (1 << 32) - 1
        for _ in range(32):
            self.value = (self.value << 1) | self._read_bit()

    def _decode_symbol(self, distribution: Dict[int, float]) -> int:
        """
        Decode one symbol using identical integer arithmetic to _encode_symbol.

        Both use _build_cdf() -> (syms, cumulative, total) and then
        compute interval boundaries as (range_width * cum) // total,
        both relative to orig_low captured before any mutation.
        """
        syms, cumulative, total = _build_cdf(distribution)
        range_width = self.high - self.low + 1

        scaled = self.value - self.low
        cdf_scaled = [(range_width * c) // total for c in cumulative]

        idx = max(bisect_right(cdf_scaled, scaled) - 1, 0)
        idx = min(idx, len(syms) - 1)

        symbol   = syms[idx]
        low_sym  = cumulative[idx]
        high_sym = cumulative[idx + 1]

        orig_low  = self.low
        self.high = orig_low + (range_width * high_sym) // total - 1
        self.low  = orig_low + (range_width * low_sym)  // total
        if self.high < self.low:
            self.high = self.low

        self._renormalize()
        return symbol

    def _renormalize(self) -> None:
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.value -= self.HALF
                self.low   -= self.HALF
                self.high  -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.value -= self.QUARTER
                self.low   -= self.QUARTER
                self.high  -= self.QUARTER
            else:
                break
            self.low   = self.low   * 2
            self.high  = self.high  * 2 + 1
            self.value = self.value * 2 + self._read_bit()

    def _read_bit(self) -> int:
        if self.reader is None:
            return 0
        return self.reader.read_bit()


# ---------------------------------------------------------------------------
# Legacy float CDF (kept for any external callers)
# ---------------------------------------------------------------------------

def _build_cumulative(
    distribution: Dict[int, float]
) -> Tuple[List[int], List[float]]:
    """Build a float CDF — kept for legacy callers only."""
    if not distribution:
        return [0], [0.0, 1.0]
    total = sum(distribution.values())
    if total <= 0:
        total = 1.0
    symbols = sorted(distribution.keys())
    cumulative = [0.0]
    running = 0.0
    for sym in symbols:
        running += distribution[sym] / total
        cumulative.append(min(running, 1.0))
    cumulative[-1] = 1.0
    return symbols, cumulative


# kept as alias so any code importing _build_cumulative_int still works
def _build_cumulative_int(
    distribution: Dict[int, float], range_width: int
) -> Tuple[List[int], List[int]]:
    syms, cum, total = _build_cdf(distribution)
    scaled = [(range_width * c) // total for c in cum]
    return syms, scaled


# ---------------------------------------------------------------------------
# Context stream builder
# ---------------------------------------------------------------------------

def _build_context_stream(
    encoded_sentences: List[Dict],
) -> Tuple[List[int], List[str], List[float]]:
    morph_stream:  List[int]   = []
    pos_stream:    List[str]   = []
    struct_probs:  List[float] = []

    for sentence in encoded_sentences:
        char_morph_codes = sentence.get("char_morph_codes", [])
        char_pos_tags    = sentence.get("char_pos_tags",    [])
        n_tags       = int(sentence.get("pos_n_tags",      0))
        huffman_bits = float(sentence.get("pos_huffman_bits", 0.0))
        if n_tags > 0 and huffman_bits > 0:
            struct_prob = 2 ** (-huffman_bits / n_tags)
        else:
            struct_prob = 1.0 / max(len(sentence.get("pos_tags", [])), 1)

        length = max(len(char_morph_codes), len(char_pos_tags))
        for i in range(length):
            morph_stream.append(
                int(char_morph_codes[i]) if i < len(char_morph_codes) else 0
            )
            pos_stream.append(
                str(char_pos_tags[i]) if i < len(char_pos_tags) else "X"
            )
            struct_probs.append(struct_prob)

    return morph_stream, pos_stream, struct_probs
