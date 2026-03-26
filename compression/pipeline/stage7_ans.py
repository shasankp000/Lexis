"""CPU rANS encoder/decoder implementation."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Dict, List


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


@dataclass
class FrequencyTable:
    """Quantised probability table for ANS encoding."""

    probabilities: Dict[int, float]
    M: int = 4096

    def __post_init__(self) -> None:
        if not _is_power_of_two(self.M):
            raise ValueError("M must be a power of 2")
        if not self.probabilities:
            raise ValueError("probabilities must not be empty")

        self._symbols = sorted(self.probabilities.keys())
        self._freqs: Dict[int, int] = self._quantise_frequencies()
        self._cumul: Dict[int, int] = {}
        self._build_cumulative()

    def _quantise_frequencies(self) -> Dict[int, int]:
        probs = self.probabilities
        symbols = sorted(probs.keys())
        freqs = {s: max(1, int(round(probs[s] * self.M))) for s in symbols}
        total = sum(freqs.values())

        if total < self.M:
            deficit = self.M - total
            for symbol in sorted(symbols, key=lambda s: probs[s], reverse=True):
                if deficit <= 0:
                    break
                freqs[symbol] += 1
                deficit -= 1
        elif total > self.M:
            surplus = total - self.M
            for symbol in sorted(symbols, key=lambda s: probs[s]):
                if surplus <= 0:
                    break
                if freqs[symbol] > 1:
                    freqs[symbol] -= 1
                    surplus -= 1

        total = sum(freqs.values())
        if total != self.M:
            target = max(symbols, key=lambda s: probs[s])
            freqs[target] += self.M - total

        return freqs

    def _build_cumulative(self) -> None:
        cumul = 0
        for symbol in self._symbols:
            self._cumul[symbol] = cumul
            cumul += self._freqs[symbol]
        self._cumul_end = cumul

    def freq(self, symbol: int) -> int:
        """Return quantised frequency for symbol."""
        return self._freqs[symbol]

    def cumul(self, symbol: int) -> int:
        """Return cumulative frequency (sum of freq[i] for i < symbol)."""
        return self._cumul[symbol]

    def symbol_from_state(self, x: int) -> int:
        """Given ANS state x, return the symbol it decodes to."""
        r = x % self.M
        cumul_values = [self._cumul[symbol] for symbol in self._symbols]
        idx = bisect_right(cumul_values, r) - 1
        return self._symbols[max(idx, 0)]

    @property
    def symbols(self) -> List[int]:
        return list(self._symbols)


class ANSEncoder:
    """CPU rANS encoder. Encodes symbol stream into a single integer."""

    def __init__(self, freq_table: FrequencyTable) -> None:
        self.table = freq_table

    def encode(self, symbols: List[int]) -> int:
        """Encode symbol stream. Returns final ANS state as integer."""
        x = self.table.M
        for symbol in reversed(symbols):
            freq = self.table.freq(symbol)
            cumul = self.table.cumul(symbol)
            x = (x // freq) * self.table.M + cumul + (x % freq)
        return x

    def state_to_bytes(self, state: int) -> bytes:
        """Serialise ANS state integer to bytes."""
        length = max(1, (state.bit_length() + 7) // 8)
        return state.to_bytes(length, "big")


class ANSDecoder:
    """CPU rANS decoder. Reconstructs symbol stream from ANS state integer."""

    def __init__(self, freq_table: FrequencyTable) -> None:
        self.table = freq_table

    def decode(self, state: int, num_symbols: int) -> List[int]:
        """Decode ANS state into symbol stream of length num_symbols."""
        x = state
        symbols: List[int] = []
        for _ in range(num_symbols):
            symbol = self.table.symbol_from_state(x)
            freq = self.table.freq(symbol)
            cumul = self.table.cumul(symbol)
            x = freq * (x // self.table.M) + (x % self.table.M) - cumul
            symbols.append(symbol)
        return symbols

    def bytes_to_state(self, data: bytes) -> int:
        """Deserialise bytes back to ANS state integer."""
        return int.from_bytes(data, "big")


def compress_stream(symbols: List[int], probabilities: Dict[int, float]) -> bytes:
    """Convenience wrapper: symbols + probabilities → compressed bytes."""
    table = FrequencyTable(probabilities)
    encoder = ANSEncoder(table)
    state = encoder.encode(symbols)
    return encoder.state_to_bytes(state)


def decompress_stream(
    data: bytes, probabilities: Dict[int, float], num_symbols: int
) -> List[int]:
    """Convenience wrapper: compressed bytes → symbol stream."""
    table = FrequencyTable(probabilities)
    decoder = ANSDecoder(table)
    state = decoder.bytes_to_state(data)
    return decoder.decode(state, num_symbols)
