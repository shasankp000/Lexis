import math
from typing import Dict, List

from compression.pipeline.stage6_probability import ContextMixingModel, UnigramModel


def _toy_encoded_sentences() -> List[Dict]:
    return [
        {
            "char_classes": [0, 1, 0, 1, 0, 1, 0],
            "char_morph_codes": [1, 1, 1, 1, 1, 1, 1],
            "pos_tags": ["NOUN", "VERB"],
            "morph_codes": [1, 2],
            "char_pos_tags": ["NOUN", "NOUN", "NOUN", "VERB", "VERB", "VERB", "VERB"],
        }
    ]


def test_probability_sums_to_one():
    model = ContextMixingModel()
    model.train(_toy_encoded_sentences())
    context = {
        "char_history": [0],
        "current_morph_code": 1,
        "current_pos_tag": "NOUN",
    }
    dist = model.probability_distribution(context)
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_no_zero_probability():
    model = ContextMixingModel()
    model.train(_toy_encoded_sentences())
    context = {
        "char_history": [1],
        "current_morph_code": 1,
        "current_pos_tag": "VERB",
    }
    dist = model.probability_distribution(context)
    assert all(value > 0.0 for value in dist.values())


def test_bpb_below_naive():
    encoded = _toy_encoded_sentences()

    class DummyPipeline:
        def encode_for_model(self, text: str) -> List[Dict]:
            return encoded

    model = ContextMixingModel()
    model.train(encoded)
    result = model.bpb("abababa", DummyPipeline())
    assert result < 8.0


def test_bpb_better_than_unigram():
    encoded = _toy_encoded_sentences()
    sequence = encoded[0]["char_classes"]
    char_morph_codes = encoded[0]["char_morph_codes"]
    char_pos_tags = encoded[0]["char_pos_tags"]

    context_model = ContextMixingModel()
    context_model.train(encoded)

    context_bits = 0.0
    prev = sequence[0]
    for idx, symbol in enumerate(sequence):
        context = {
            "char_history": [prev],
            "current_morph_code": char_morph_codes[idx],
            "current_pos_tag": char_pos_tags[idx],
        }
        prob = context_model.probability(symbol, context)
        context_bits += -math.log2(prob)
        prev = symbol

    unigram = UnigramModel()
    unigram.train([sequence])
    unigram_bits = sum(-math.log2(unigram.probability(s)) for s in sequence)

    assert context_bits < unigram_bits


def test_model_serialise_round_trip(tmp_path):
    model = ContextMixingModel()
    model.train(_toy_encoded_sentences())

    path = tmp_path / "context_model.msgpack"
    model.serialise(str(path))

    reloaded = ContextMixingModel()
    reloaded.load(str(path))

    context = {
        "char_history": [0],
        "current_morph_code": 1,
        "current_pos_tag": "NOUN",
    }
    dist_original = model.probability_distribution(context)
    dist_loaded = reloaded.probability_distribution(context)

    assert dist_original == dist_loaded
