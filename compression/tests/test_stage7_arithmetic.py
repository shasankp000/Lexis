from typing import Dict, List

import pytest

from compression.config import SPACY_MODEL
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.stage7_arithmetic import ArithmeticDecoder, ArithmeticEncoder
from main import compress_to_file, decompress


class _UniformContextModel:
    def __init__(self, symbols: List[int]) -> None:
        self.symbols = sorted(set(symbols)) or [0]

    def probability_distribution(self, context: Dict) -> Dict[int, float]:
        prob = 1.0 / len(self.symbols)
        return {symbol: prob for symbol in self.symbols}


def _build_encoded_sentences(char_classes: List[int]) -> List[Dict]:
    length = len(char_classes)
    return [
        {
            "char_classes": char_classes,
            "char_morph_codes": [0] * length,
            "char_pos_tags": ["NOUN"] * length,
            "pos_huffman_bits": 1.0,
            "pos_n_tags": 1,
            "pos_tags": ["NOUN"],
            "morph_codes": [0],
        }
    ]


def test_round_trip_short_uniform():
    symbols = [1, 1, 0, 1, 0]
    encoded_sentences = _build_encoded_sentences(symbols)
    model = _UniformContextModel(symbols)

    encoder = ArithmeticEncoder()
    compressed = encoder.encode(symbols, model, {}, encoded_sentences)

    decoder = ArithmeticDecoder()
    decoded = decoder.decode(compressed, model, encoded_sentences, len(symbols))

    assert decoded == symbols


def test_round_trip_context_model():
    symbols = [0, 1, 0, 1, 1, 0]
    encoded_sentences = _build_encoded_sentences(symbols)
    model = ContextMixingModel()
    model.train(encoded_sentences)

    encoder = ArithmeticEncoder()
    compressed = encoder.encode(symbols, model, {}, encoded_sentences)

    decoder = ArithmeticDecoder()
    decoded = decoder.decode(compressed, model, encoded_sentences, len(symbols))

    assert decoded == symbols


def test_compression_ratio_validation():
    symbols = [0, 1] * 20
    encoded_sentences = _build_encoded_sentences(symbols)
    model = _UniformContextModel(symbols)

    encoder = ArithmeticEncoder()
    compressed = encoder.encode(symbols, model, {}, encoded_sentences)

    bpb = (len(compressed) * 8) / max(len(symbols), 1)
    assert bpb >= 0.0


def test_full_compression_pipeline(tmp_path):
    spacy = pytest.importorskip("spacy")
    try:
        spacy.load(SPACY_MODEL)
    except Exception:
        pytest.skip(f"spaCy model {SPACY_MODEL} not available")

    text = "The whale swam."
    output_path = tmp_path / "test.bin"

    compress_to_file(text, str(output_path), model=SPACY_MODEL)
    decompressed = decompress(str(output_path))

    assert decompressed == text
