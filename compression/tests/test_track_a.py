import re

from compression.alphabet.phonetic_map import (
    char_to_triple,
    compute_deltas,
    get_coords,
)
from compression.pipeline.stage5_encode import CharacterEncoder, encode_factoradic

_SAMPLE_PARAGRAPH = (
    "In the quiet valley, the river wound through fields and forests. "
    "Villagers walked along the banks, speaking softly about the changing seasons. "
    "Their stories carried memories of storms, harvests, and long journeys home. "
    "Each evening, lamps glowed in windows, and the air smelled of bread and rain."
)


def _sample_text(word_target: int = 600) -> str:
    words = re.findall(r"[a-zA-Z]+", _SAMPLE_PARAGRAPH.lower())
    repeats = max(1, word_target // len(words) + 1)
    return " ".join(words * repeats)


def _words_from_text(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


def _sequence_with_markers(words: list[str]) -> list[str]:
    sequence: list[str] = []
    for idx, word in enumerate(words):
        sequence.append("^")
        sequence.extend(list(word))
        sequence.append("$")
        if idx < len(words) - 1:
            sequence.append("_")
    return sequence


def _triples_for_words(words: list[str]) -> list[tuple[int, int, int]]:
    triples: list[tuple[int, int, int]] = []
    for idx, word in enumerate(words):
        triples.append(char_to_triple("^", 0, 1))
        for pos, char in enumerate(word):
            triples.append(char_to_triple(char, pos, len(word)))
        triples.append(char_to_triple("$", 0, 1))
        if idx < len(words) - 1:
            triples.append(char_to_triple("_", 0, 1))
    return triples


def _flat_char_id(char: str) -> int:
    lower = char.lower()
    if "a" <= lower <= "z":
        return ord(lower) - ord("a")
    if lower == "^":
        return 50
    if lower == "$":
        return 51
    if lower == "_":
        return 52
    if lower == ".":
        return 53
    return 54


def test_phonetic_map_complete():
    alphabet = [chr(c) for c in range(ord("a"), ord("z") + 1)]
    specials = ["^", "$", "_", "."]
    for char in alphabet + specials:
        coords = get_coords(char)
        assert isinstance(coords, tuple)
        assert len(coords) == 2


def test_delta_range_improvement():
    text = _sample_text(600)
    words = _words_from_text(text)
    triples = _triples_for_words(words)
    class_deltas, pos_deltas, _ = compute_deltas(triples)

    max_class = max(abs(v) for v in class_deltas[1:]) if len(class_deltas) > 1 else 0
    max_pos = max(abs(v) for v in pos_deltas[1:]) if len(pos_deltas) > 1 else 0
    decomposed_span = max(max_class, max_pos)

    flat_sequence = _sequence_with_markers(words)
    flat_ids = [_flat_char_id(ch) for ch in flat_sequence]
    flat_deltas = [flat_ids[0]] + [b - a for a, b in zip(flat_ids, flat_ids[1:])]
    flat_span = max(abs(v) for v in flat_deltas[1:]) if len(flat_deltas) > 1 else 0

    assert decomposed_span < 12
    assert flat_span > 40


def test_improvement_ratio():
    encoder = CharacterEncoder()
    stats = encoder.stats(_sample_text(600))
    assert stats["improvement_ratio"] > 2.0


def test_morph_role_near_zero():
    text = _sample_text(600)
    words = _words_from_text(text)
    triples = _triples_for_words(words)
    _, _, role_deltas = compute_deltas(triples)
    if len(role_deltas) <= 1:
        assert True
    else:
        mean_role = sum(abs(v) for v in role_deltas[1:]) / (len(role_deltas) - 1)
        assert mean_role < 1.0


def test_round_trip_encode_decode():
    encoder = CharacterEncoder()
    words = _words_from_text(_sample_text(200))
    for word in words[:200]:
        encoded = encoder.encode_word(word)
        decoded = encoder.decode_word(encoded)
        assert decoded == word


def test_factoriadic_small_numbers():
    assert len(encode_factoradic(0)) == 1
    assert len(encode_factoradic(1)) == 2
    assert len(encode_factoradic(2)) == 3


def test_special_symbols_handled():
    encoder = CharacterEncoder()
    for symbol in ["^", "$", "_", "."]:
        coords = get_coords(symbol)
        assert coords[0] == 6
        encoded = encoder.encode_word(symbol)
        decoded = encoder.decode_word(encoded)
        assert decoded == symbol
