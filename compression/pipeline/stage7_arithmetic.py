"""Arithmetic encoder/decoder for context-dependent compression."""

from __future__ import annotations

from bisect import bisect_right
from collections import deque
from typing import Dict, List, Tuple

from compression.config import CHAR_CONTEXT_SIZE


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

    def _reset(self) -> None:
        """Reset all mutable coding state so the instance can be reused."""
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
        self._reset()
        morph_stream, pos_stream, struct_probs = _build_context_stream(
            encoded_sentences
        )
        char_history = deque(maxlen=CHAR_CONTEXT_SIZE)

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
            low_sym = cumulative[idx]
            high_sym = cumulative[idx + 1]
            range_width = self.high - self.low + 1
            low_new = self.low + (range_width * low_sym) // total
            high_new = self.low + (range_width * high_sym) // total - 1
            self.low = low_new
            self.high = max(high_new, self.low)
            self._renormalize()
        self._finalize()
        return self.writer.get_bytes()

    def _encode_symbol(self, symbol: int, distribution: Dict[int, float]) -> None:
        symbols, cumulative = _build_cumulative(distribution)
        if symbol not in symbols:
            raise ValueError(f"Symbol {symbol} missing from distribution.")

        idx = symbols.index(symbol)
        low_sym = cumulative[idx]
        high_sym = cumulative[idx + 1]

        range_width = self.high - self.low + 1
        low_new = self.low + int(range_width * low_sym)
        high_new = self.low + int(range_width * high_sym) - 1

        self.low = low_new
        self.high = max(high_new, self.low)

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
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.pending += 1
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low = self.low * 2
            self.high = self.high * 2 + 1

    def _finalize(self) -> None:
        self.pending += 1
        if self.low < self.QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)
        self.writer.flush()


class ArithmeticDecoder:
    """Decodes arithmetic-coded bitstreams back to symbol streams."""

    FULL = 1 << 32
    HALF = FULL >> 1
    QUARTER = HALF >> 1
    THREE_QUARTER = QUARTER * 3

    def __init__(self) -> None:
        self.reader: _BitReader | None = None
        self.value = 0
        self.low = 0
        self.high = (1 << 32) - 1

    def _reset(self, compressed_bytes: bytes) -> None:
        """Reset all mutable decoding state for a fresh decode pass."""
        self.reader = _BitReader(compressed_bytes)
        self.value = 0
        self.low = 0
        self.high = (1 << 32) - 1
        for _ in range(32):
            self.value = (self.value << 1) | self._read_bit()

    def decode(
        self,
        compressed_bytes: bytes,
        context_model,
        encoded_sentences: List[Dict],
        num_symbols: int,
    ) -> List[int]:
        self._reset(compressed_bytes)

        morph_stream, pos_stream, struct_probs = _build_context_stream(
            encoded_sentences
        )
        char_history = deque(maxlen=CHAR_CONTEXT_SIZE)
        decoded: List[int] = []

        length = min(num_symbols, len(morph_stream), len(pos_stream))
        for idx in range(length):
            context = {
                "char_history": list(char_history),
                "current_morph_code": morph_stream[idx],
                "current_pos_tag": pos_stream[idx],
                "struct_prob": struct_probs[idx],
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
        self._reset(compressed_bytes)
        decoded: List[int] = []
        for _ in range(num_symbols):
            symbol = self._decode_symbol(distribution)
            decoded.append(symbol)
        return decoded

    def decode_unigram_counts(
        self, compressed_bytes: bytes, counts: Dict[int, int], num_symbols: int
    ) -> List[int]:
        """Decode a symbol stream using integer unigram counts."""
        self._reset(compressed_bytes)

        symbols_sorted, cumulative, total = _build_cumulative_counts(counts)
        decoded: List[int] = []
        for _ in range(num_symbols):
            range_width = self.high - self.low + 1
            scaled = ((self.value - self.low + 1) * total - 1) // range_width
            idx = max(bisect_right(cumulative, scaled) - 1, 0)
            symbol = symbols_sorted[min(idx, len(symbols_sorted) - 1)]
            low_sym = cumulative[idx]
            high_sym = cumulative[idx + 1]
            low_new = self.low + (range_width * low_sym) // total
            high_new = self.low + (range_width * high_sym) // total - 1
            self.low = low_new
            self.high = max(high_new, self.low)
            self._renormalize()
            decoded.append(symbol)
        return decoded

    def _decode_symbol(self, distribution: Dict[int, float]) -> int:
        symbols, cumulative = _build_cumulative(distribution)
        range_width = self.high - self.low + 1
        position = (self.value - self.low + 1) / range_width

        symbol = symbols[-1]
        low_sym = cumulative[-2]
        high_sym = cumulative[-1]
        for idx, sym in enumerate(symbols):
            if position <= cumulative[idx + 1]:
                symbol = sym
                low_sym = cumulative[idx]
                high_sym = cumulative[idx + 1]
                break

        low_new = self.low + int(range_width * low_sym)
        high_new = self.low + int(range_width * high_sym) - 1

        self.low = low_new
        self.high = max(high_new, self.low)

        self._renormalize()
        return symbol

    def _renormalize(self) -> None:
        while True:
            if self.high < self.HALF:
                pass
            elif self.low >= self.HALF:
                self.value -= self.HALF
                self.low -= self.HALF
                self.high -= self.HALF
            elif self.low >= self.QUARTER and self.high < self.THREE_QUARTER:
                self.value -= self.QUARTER
                self.low -= self.QUARTER
                self.high -= self.QUARTER
            else:
                break
            self.low = self.low * 2
            self.high = self.high * 2 + 1
            self.value = self.value * 2 + self._read_bit()

    def _read_bit(self) -> int:
        if self.reader is None:
            return 0
        return self.reader.read_bit()


def _build_cumulative(distribution: Dict[int, float]) -> Tuple[List[int], List[float]]:
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


def _build_cumulative_counts(
    counts: Dict[int, int],
) -> Tuple[List[int], List[int], int]:
    if not counts:
        return [0], [0, 1], 1
    symbols = sorted(counts.keys())
    cumulative = [0]
    running = 0
    for sym in symbols:
        running += max(int(counts[sym]), 0)
        cumulative.append(running)
    total = cumulative[-1] if cumulative[-1] > 0 else 1
    return symbols, cumulative, total


def _build_context_stream(
    encoded_sentences: List[Dict],
) -> Tuple[List[int], List[str], List[float]]:
    morph_stream: List[int] = []
    pos_stream: List[str] = []
    struct_probs: List[float] = []

    for sentence in encoded_sentences:
        char_morph_codes = sentence.get("char_morph_codes", [])
        char_pos_tags = sentence.get("char_pos_tags", [])
        n_tags = int(sentence.get("pos_n_tags", 0))
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
            pos_stream.append(str(char_pos_tags[i]) if i < len(char_pos_tags) else "X")
            struct_probs.append(struct_prob)

    return morph_stream, pos_stream, struct_probs
