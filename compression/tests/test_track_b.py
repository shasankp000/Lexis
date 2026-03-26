import pytest

from compression.alphabet.morph_codes import (
    ADVERBIAL,
    IRREGULAR,
    PAST_TENSE,
    PLURAL,
    PRESENT_PART,
    apply_morph,
)
from compression.alphabet.symbol_alphabet import SymbolAlphabet
from compression.config import SPACY_MODEL
from compression.pipeline.stage2_morphology import MorphologicalAnalyser
from compression.pipeline.stage3_syntax import analyse_sentence
from compression.pipeline.stage5_encode import StructuralEncoder

spacy = pytest.importorskip("spacy")
try:
    _NLP = spacy.load(SPACY_MODEL)
except Exception:  # pragma: no cover - environment-specific model availability
    pytest.skip(f"spaCy model {SPACY_MODEL} not available", allow_module_level=True)


def test_morph_common_inflections():
    analyser = MorphologicalAnalyser(use_spacy=True)
    if analyser.nlp is None:
        pytest.skip("spaCy pipeline unavailable")

    assert analyser.analyse("walked") == ("walk", PAST_TENSE)
    assert analyser.analyse("running") == ("run", PRESENT_PART)
    assert analyser.analyse("dogs") == ("dog", PLURAL)
    assert analyser.analyse("went") == ("go", IRREGULAR)

    fallback = MorphologicalAnalyser(use_spacy=False)
    assert fallback.analyse("quickly") == ("quick", ADVERBIAL)


def test_morph_round_trip():
    cases = [
        ("walk", PAST_TENSE, "walked"),
        ("run", PRESENT_PART, "running"),
        ("dog", PLURAL, "dogs"),
        ("quick", ADVERBIAL, "quickly"),
        ("go", IRREGULAR, "went"),
    ]
    for root, code, surface in cases:
        assert apply_morph(root, code) == surface


def test_morph_char_savings():
    analyser = MorphologicalAnalyser(use_spacy=True)
    if analyser.nlp is None:
        analyser = MorphologicalAnalyser(use_spacy=False)

    sentence = "The tired dogs were running quickly home tonight."
    stats = analyser.char_savings(sentence)
    assert stats["pct_saved"] > 10.0


def test_pos_delta_improvement():
    encoder = StructuralEncoder(SymbolAlphabet())
    sentences = [
        "The old man walked slowly home after the long day.",
        "A quick fox jumps over the lazy dog.",
        "She writes letters on Sundays.",
        "The storm broke suddenly and the lights went out.",
        "They finished their meal and left quietly.",
        "My friend bought a new laptop yesterday.",
        "Birds sing loudly in the early morning.",
        "The committee approved the plan without delay.",
        "He studied hard because the exam was difficult.",
        "We watched the movie and discussed the ending.",
    ]

    flat_means = []
    delta_means = []
    for sentence in sentences:
        doc = _NLP(sentence)
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        report = encoder.measure_pos_delta_improvement(pos_tags)
        flat_means.append(float(report.get("flat_mean", 0.0)))
        delta_means.append(float(report.get("delta_mean", 0.0)))

    avg_flat = sum(flat_means) / len(flat_means)
    avg_delta = sum(delta_means) / len(delta_means)
    ratio = (avg_flat / avg_delta) if avg_delta else 0.0
    print(f"POS delta improvement ratio (avg): {ratio:.3f}")

    assert avg_delta < avg_flat


def test_tree_shape_consistent():
    encoder = StructuralEncoder(SymbolAlphabet())
    doc1 = _NLP("The old man walked.")
    doc2 = _NLP("The young girl laughed.")
    syntax1 = analyse_sentence(doc1)
    syntax2 = analyse_sentence(doc2)

    tree_id_1 = encoder.encode_tree_shape(syntax1.tree_shape)
    tree_id_2 = encoder.encode_tree_shape(syntax2.tree_shape)
    assert tree_id_1 == tree_id_2


def test_structural_round_trip():
    encoder = StructuralEncoder(SymbolAlphabet())
    doc = _NLP("The quick fox jumps over the lazy dog.")
    syntax = analyse_sentence(doc)
    encoded = encoder.encode_pos_sequence(syntax.pos_tags)

    encoded_values = encoded["encoded_values"]
    used_delta = encoded["used_delta"]

    decoded = encoder.decode_pos_sequence(encoded_values, used_delta)
    assert decoded == syntax.pos_tags


def test_sentence_meta_round_trip():
    encoder = StructuralEncoder(SymbolAlphabet())
    syntax = analyse_sentence(_NLP("Was the door opened by the man?"))

    meta = encoder.encode_sentence_meta(syntax)
    decoded = encoder.decode_sentence_meta(meta)

    assert decoded["sentence_type"] == syntax.sentence_type
    assert decoded["voice"] == syntax.voice


def test_syntax_sentence_type():
    declarative = analyse_sentence(_NLP("The old man walked slowly home."))
    interrogative = analyse_sentence(_NLP("Was the door opened by the man?"))
    passive = analyse_sentence(_NLP("The door was opened by the man."))

    assert declarative.sentence_type == "DECLARATIVE"
    assert declarative.voice == "ACTIVE"

    assert interrogative.sentence_type == "INTERROGATIVE"
    assert interrogative.voice == "PASSIVE"

    assert passive.voice == "PASSIVE"
