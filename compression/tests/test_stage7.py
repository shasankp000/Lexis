from collections import Counter

from compression.pipeline.stage1_normalize import normalize_text
from compression.pipeline.stage7_ans import (
    ANSDecoder,
    ANSEncoder,
    FrequencyTable,
    compress_stream,
    decompress_stream,
)


def _symbolize(text: str) -> tuple[list[int], dict[int, float]]:
    normalized = normalize_text(text).lower()
    symbols = [ord(ch) for ch in normalized]
    counts = Counter(symbols)
    total = sum(counts.values()) or 1
    probabilities = {symbol: count / total for symbol, count in counts.items()}
    return symbols, probabilities


def test_round_trip_short():
    symbols = [0, 1, 0, 1, 1, 0, 0, 1, 0, 1]
    probabilities = {0: 0.5, 1: 0.5}
    data = compress_stream(symbols, probabilities)
    decoded = decompress_stream(data, probabilities, len(symbols))
    assert decoded == symbols


def test_round_trip_long():
    symbols = [0, 1] * 500
    probabilities = {0: 0.6, 1: 0.4}
    data = compress_stream(symbols, probabilities)
    decoded = decompress_stream(data, probabilities, len(symbols))
    assert decoded == symbols


def test_high_prob_symbol_cheaper():
    probabilities = {0: 0.9, 1: 0.1}
    table = FrequencyTable(probabilities)
    encoder = ANSEncoder(table)

    high_prob_state = encoder.encode([0] * 200)
    low_prob_state = encoder.encode([1] * 200)

    assert high_prob_state.bit_length() < low_prob_state.bit_length()


def test_bpb_on_toy_corpus():
    text = "the cat sat on the mat " * 50
    symbols, probabilities = _symbolize(text)

    data = compress_stream(symbols, probabilities)
    bpb = (len(data) * 8) / len(text.encode("utf-8"))

    print(f"ANS bpb (toy corpus): {bpb:.4f}")
    assert bpb < 8.0


def test_state_serialisation():
    probabilities = {0: 0.7, 1: 0.3}
    table = FrequencyTable(probabilities)
    encoder = ANSEncoder(table)
    decoder = ANSDecoder(table)

    state = encoder.encode([0, 1, 0, 1, 0])
    data = encoder.state_to_bytes(state)
    recovered = decoder.bytes_to_state(data)

    assert recovered == state
