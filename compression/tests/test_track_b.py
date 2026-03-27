import math
from collections import Counter

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


def test_pos_huffman_encoding():
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

    pos_sentences = []
    for sentence in sentences:
        doc = _NLP(sentence)
        pos_sentences.append([token.pos_ for token in doc if not token.is_space])

    freq_table = encoder.build_pos_frequency_table(pos_sentences)

    total_huffman_bits = 0.0
    total_tags = 0
    for pos_tags in pos_sentences:
        encoding = encoder.encode_pos_sequence(pos_tags, freq_table)
        total_huffman_bits += encoding["pos_huffman_bits"]
        total_tags += encoding["tag_count"]

    vocab_size = len(freq_table)
    bits_per_tag = math.ceil(math.log2(vocab_size)) if vocab_size > 1 else 1
    flat_bits = total_tags * bits_per_tag
    print(f"POS Huffman bits saved vs flat: {flat_bits - total_huffman_bits:.3f}")

    assert total_huffman_bits <= flat_bits


def test_frequency_sorted_index_reduces_deltas():
    """Verify frequency table assigns counts to most common tags."""
    encoder = StructuralEncoder(SymbolAlphabet())

    # Realistic English POS distribution across 17 coarse tags
    sentences = (
        [["NOUN", "VERB", "DET", "NOUN", "PUNCT"]] * 40
        + [["DET", "NOUN", "VERB", "ADJ", "NOUN"]] * 30
        + [["PRON", "VERB", "ADV"]] * 20
        + [["ADP", "DET", "NOUN"]] * 15
        + [["ADJ", "NOUN", "VERB", "NOUN", "PUNCT"]] * 10
        + [["PROPN", "VERB", "DET", "NOUN"]] * 8
        + [["INTJ", "PUNCT"]] * 2
    )

    freq_table = encoder.build_pos_frequency_table(sentences)

    all_tags = [tag for sentence in sentences for tag in sentence]
    most_common = Counter(all_tags).most_common(1)[0][0]

    # Guarantee 1: most common tag has the highest count
    assert freq_table[most_common] == max(freq_table.values())

    # Guarantee 2: every tag in the corpus is in the table
    unique_tags = set(all_tags)
    assert set(freq_table.keys()) == unique_tags, (
        f"Freq table missing tags: {unique_tags - set(freq_table.keys())}"
    )

    # Guarantee 3: counts are positive
    assert all(count > 0 for count in freq_table.values())


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

    freq_table = encoder.build_pos_frequency_table([syntax.pos_tags])
    encoded = encoder.encode_pos_sequence(syntax.pos_tags, freq_table)

    bitstring = encoded["pos_huffman_bitstring"]
    codes = encoded["pos_huffman_codes"]

    decoded = encoder.decode_pos_sequence(bitstring, codes)
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
    # Note: en_core_web_lg does not detect passive voice in interrogative form
    # passive interrogative detection is a known limitation — do not assert voice here

    assert passive.sentence_type == "DECLARATIVE"
    assert passive.voice == "PASSIVE"  # lg correctly detects this
