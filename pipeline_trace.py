"""
pipeline_trace.py  —  full per-stage expected vs actual comparison.

Covers stages 1-9 including the real spaCy pipeline on moby500.txt sentence 1.

Run:  python pipeline_trace.py
"""
from __future__ import annotations

import sys
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple

PASS = "\033[92mOK\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SEP  = "\n" + "-" * 70

def label(name, ok, got=None, expected=None):
    tag = PASS if ok else FAIL
    print(f"  [{tag}]  {name}")
    if not ok:
        print(f"          expected : {expected}")
        print(f"          got      : {got}")


# =========================================================================
# STAGE 1  — normalize_text
# =========================================================================
print(SEP)
print("STAGE 1 — normalize_text")
print(SEP)

from compression.pipeline.stage1_normalize import normalize_text

raw = open("moby500.txt", encoding="utf-8").read()
normalized = normalize_text(raw)

label("normalized is non-empty", len(normalized) > 0, len(normalized), ">0")
label("no leading/trailing whitespace", normalized == normalized.strip(),
      repr(normalized[:30]), "stripped")
label("no double spaces", "  " not in normalized, "double space found", "none")
print(f"  first 120 chars : {repr(normalized[:120])}")


# =========================================================================
# STAGE 2  — MorphologicalAnalyser  (first sentence only)
# =========================================================================
print(SEP)
print("STAGE 2 — MorphologicalAnalyser (first sentence)")
print(SEP)

import spacy
nlp = spacy.load("en_core_web_lg")
nlp.max_length = 2_000_000

from compression.pipeline.stage2_morphology import MorphologicalAnalyser

analyser = MorphologicalAnalyser(use_spacy=True)
doc = nlp(normalized)
first_sent = list(doc.sents)[0]
morph_results = analyser.analyse_sentence(first_sent.text)

label("morph results non-empty", len(morph_results) > 0, len(morph_results), ">0")
for item in morph_results[:5]:
    print(f"  token={item[0]!r:15}  root={item[1]!r:15}  code={item[2]}")


# =========================================================================
# STAGE 3  — analyse_sentence (syntax)
# =========================================================================
print(SEP)
print("STAGE 3 — analyse_sentence (syntax)")
print(SEP)

from compression.pipeline.stage3_syntax import analyse_sentence

syntax = analyse_sentence(first_sent)

label("pos_tags non-empty", len(syntax.pos_tags) > 0, len(syntax.pos_tags), ">0")
label("pos_tags count == morph token count",
      len(syntax.pos_tags) == len(morph_results),
      len(syntax.pos_tags), len(morph_results))
print(f"  pos_tags : {syntax.pos_tags}")
print(f"  tree_shape: {syntax.tree_shape!r}")


# =========================================================================
# STAGE 4  — discourse / symbol encoding
# =========================================================================
print(SEP)
print("STAGE 4 — discourse symbol encoding")
print(SEP)

from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import encode_symbols, decode_symbols

disc_analyser = DiscourseAnalyser(use_spacy=True, device="cpu")
stage4_result = disc_analyser.analyse_document(normalized)
compressed_text, symbol_table = encode_symbols(normalized, stage4_result)
restored_text  = decode_symbols(compressed_text, symbol_table)

label("symbol round-trip", restored_text == normalized, repr(restored_text[:80]), repr(normalized[:80]))
print(f"  symbols encoded : {len(symbol_table)}")
print(f"  compressed length : {len(compressed_text)} vs original {len(normalized)}")


# =========================================================================
# STAGE 5A  — encode_sentence_full on first sentence
# =========================================================================
print(SEP)
print("STAGE 5 — encode_sentence_full (first sentence)")
print(SEP)

from compression.pipeline.stage5_encode import (
    CharacterEncoder, StructuralEncoder,
    _expand_morph_codes_for_chars, _expand_pos_tags_for_chars,
    _triples_for_sentence, _char_classes_from_triples,
)
from compression.alphabet.phonetic_map import PHONETIC_CLASSES, compute_deltas
from compression.alphabet.symbol_alphabet import SymbolAlphabet

inverse_map = {coords: ch for ch, coords in PHONETIC_CLASSES.items()}

char_enc  = CharacterEncoder()
sym_alpha = SymbolAlphabet()
struct_enc = StructuralEncoder(sym_alpha)

# Build freq table from first sentence only
freq_table = struct_enc.build_pos_frequency_table([syntax.pos_tags])

encoded = char_enc.encode_sentence_full(morph_results, syntax, struct_enc, freq_table)

roots       = encoded["roots"]
morph_codes = encoded["morph_codes"]
pos_tags    = encoded["pos_tags"]
char_classes     = encoded["char_classes"]
char_morph_codes = encoded["char_morph_codes"]
char_pos_tags    = encoded["char_pos_tags"]
pos_deltas_vals  = encoded["pos_deltas"]

label("char_morph_codes length == char_classes",
      len(char_morph_codes) == len(char_classes),
      len(char_morph_codes), len(char_classes))
label("char_pos_tags length == char_classes",
      len(char_pos_tags) == len(char_classes),
      len(char_pos_tags), len(char_classes))
label("pos_deltas length == char_classes",
      len(pos_deltas_vals) == len(char_classes),
      len(pos_deltas_vals), len(char_classes))

print(f"  roots           : {roots}")
print(f"  morph_codes     : {morph_codes}")
print(f"  pos_tags        : {pos_tags}")
print(f"  char_classes    : {char_classes}")
print(f"  char_morph_codes: {char_morph_codes}")
print(f"  char_pos_tags   : {char_pos_tags}")
print(f"  pos_deltas      : {pos_deltas_vals}")

# verify inverse map round-trip on char_classes + pos_deltas
from main import _cumulative_from_deltas
positions = _cumulative_from_deltas(pos_deltas_vals)
reconstructed_seq = "".join(
    inverse_map.get((cls, pos), "?") for cls, pos in zip(char_classes, positions)
)
expected_seq = "^" + "$_^".join(roots) + "$"
label("class+delta → char sequence",
      reconstructed_seq == expected_seq,
      repr(reconstructed_seq), repr(expected_seq))


# =========================================================================
# STAGE 5C  — metadata reconstruction identical to original streams
# =========================================================================
print(SEP)
print("STAGE 5C — metadata reconstruction vs original streams")
print(SEP)

root_lengths = [len(r) for r in roots]

# replicate _build_encoded_sentences_from_metadata
recon_morph: List[int] = []
recon_pos:   List[str] = []
for token_idx, length in enumerate(root_lengths):
    pt  = pos_tags[token_idx]    if token_idx < len(pos_tags)    else "X"
    mc  = morph_codes[token_idx] if token_idx < len(morph_codes) else 0
    recon_morph.append(0)              # start ^
    recon_morph.extend([mc] * length)
    recon_morph.append(0)              # end $
    recon_pos.append("X")
    recon_pos.extend([pt] * length)
    recon_pos.append("X")
    if token_idx < len(root_lengths) - 1:
        recon_morph.append(0)          # space _
        recon_pos.append("X")

label("recon morph == original",
      recon_morph == char_morph_codes,
      recon_morph[:20], char_morph_codes[:20])
label("recon pos == original",
      recon_pos == char_pos_tags,
      recon_pos[:20], char_pos_tags[:20])

if recon_pos != char_pos_tags:
    for i, (a, b) in enumerate(zip(recon_pos, char_pos_tags)):
        if a != b:
            print(f"  first pos mismatch @ index {i}: recon={a!r}  orig={b!r}")
            break
    print(f"  recon length={len(recon_pos)}  orig length={len(char_pos_tags)}")


# =========================================================================
# STAGE 6  — context model
# =========================================================================
print(SEP)
print("STAGE 6 — ContextMixingModel")
print(SEP)

from compression.pipeline.stage6_probability import ContextMixingModel

fake_sentence = {
    "char_classes":     char_classes,
    "char_morph_codes": char_morph_codes,
    "char_pos_tags":    char_pos_tags,
    "pos_tags":         pos_tags,
    "morph_codes":      morph_codes,
    "roots":            roots,
    "pos_huffman_bits": float(encoded["pos_huffman_bits"]),
    "pos_n_tags":       int(encoded["pos_n_tags"]),
}

cm = ContextMixingModel()
cm.train([fake_sentence])

context = {
    "char_history":        [char_classes[0]],
    "current_morph_code":  char_morph_codes[1],
    "current_pos_tag":     char_pos_tags[1],
    "struct_prob":         0.5,
}
dist = cm.probability_distribution(context)
total = sum(dist.values())
label("prob sums to 1.0", abs(total - 1.0) < 1e-9, total, 1.0)


# =========================================================================
# STAGE 7  — arithmetic round-trip
# =========================================================================
print(SEP)
print("STAGE 7 — arithmetic coder round-trip")
print(SEP)

from compression.pipeline.stage7_arithmetic import ArithmeticEncoder, ArithmeticDecoder

enc = ArithmeticEncoder()
compressed = enc.encode(char_classes, cm, {}, [fake_sentence])

dec = ArithmeticDecoder()
decoded_classes = dec.decode(compressed, cm, [fake_sentence], len(char_classes))

label("decoded length == original",
      len(decoded_classes) == len(char_classes),
      len(decoded_classes), len(char_classes))
label("decoded classes == original",
      decoded_classes == char_classes,
      decoded_classes, char_classes)

if decoded_classes != char_classes:
    for i, (a, b) in enumerate(zip(decoded_classes, char_classes)):
        if a != b:
            print(f"  first class mismatch @ index {i}: decoded={a}, original={b}")
            break


# =========================================================================
# STAGE 7B — pos_deltas round-trip
# =========================================================================
print(SEP)
print("STAGE 7B — pos_deltas round-trip")
print(SEP)

counts = Counter(pos_deltas_vals)
enc2 = ArithmeticEncoder()
pd_compressed = enc2.encode_unigram_counts(pos_deltas_vals, counts)
dec2 = ArithmeticDecoder()
decoded_deltas = dec2.decode_unigram_counts(pd_compressed, counts, len(pos_deltas_vals))

label("pos_deltas round-trip",
      decoded_deltas == pos_deltas_vals,
      decoded_deltas, pos_deltas_vals)


# =========================================================================
# STAGE 8  — _reconstruct_chars
# =========================================================================
print(SEP)
print("STAGE 8 — _reconstruct_chars")
print(SEP)

from main import _reconstruct_chars, _split_roots, _join_words

reconstructed = _reconstruct_chars(
    decoded_classes, decoded_deltas, [len(char_classes)]
)
expected_seq2 = "^" + "$_^".join(roots) + "$"
label("_reconstruct_chars output",
      reconstructed == expected_seq2,
      repr(reconstructed[:60]), repr(expected_seq2[:60]))


# =========================================================================
# STAGE 9  — morph decode
# =========================================================================
print(SEP)
print("STAGE 9 — morph decode")
print(SEP)

from compression.alphabet.morph_codes import apply_morph

roots_out = _split_roots(reconstructed)
label("split roots match", roots_out == roots, roots_out, roots)

words_out = [apply_morph(r, morph_codes[i] if i < len(morph_codes) else 0)
             for i, r in enumerate(roots_out)]
final = _join_words(words_out)
expected_words = first_sent.text.strip()
label("final text",
      final.lower() == expected_words.lower(),
      repr(final), repr(expected_words))

print(SEP)
print("Done.")
