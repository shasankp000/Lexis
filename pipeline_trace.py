"""
pipeline_trace.py  —  full per-stage expected vs actual comparison.

Covers the REAL compress → decompress round-trip path, not just isolated
encoder checks. Every stage is tested including the actual
_build_encoded_sentences_from_metadata path used by decompress.

Run:  python pipeline_trace.py
"""
from __future__ import annotations

import os
import sys
import io
from collections import Counter
from typing import List

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

label("normalized is non-empty",         len(normalized) > 0)
label("no leading/trailing whitespace",  normalized == normalized.strip())
label("no double spaces",                "  " not in normalized)
label("no BOM in normalized text",       "\ufeff" not in normalized,
      got=repr(normalized[:20]), expected="no BOM")
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

label("morph results non-empty", len(morph_results) > 0)
for item in morph_results[:5]:
    print(f"  token={item[0]!r:15}  root={item[1]!r:15}  code={item[2]}")
label("no root contains BOM",
      not any("\ufeff" in item[1] for item in morph_results),
      got=[item[1] for item in morph_results if "\ufeff" in item[1]],
      expected="[]")


# =========================================================================
# STAGE 3  — analyse_sentence
# =========================================================================
print(SEP)
print("STAGE 3 — analyse_sentence (syntax)")
print(SEP)

from compression.pipeline.stage3_syntax import analyse_sentence
syntax = analyse_sentence(first_sent)

label("pos_tags non-empty", len(syntax.pos_tags) > 0)
label("pos_tags count == morph token count",
      len(syntax.pos_tags) == len(morph_results),
      len(syntax.pos_tags), len(morph_results))
print(f"  pos_tags[:5] : {syntax.pos_tags[:5]}")


# =========================================================================
# STAGE 4  — discourse / symbol encoding
# =========================================================================
print(SEP)
print("STAGE 4 — discourse symbol round-trip")
print(SEP)

from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import encode_symbols, decode_symbols

disc_analyser = DiscourseAnalyser(use_spacy=True, device="cpu")
stage4_result = disc_analyser.analyse_document(normalized)
compressed_text, symbol_table = encode_symbols(normalized, stage4_result)
restored_text  = decode_symbols(compressed_text, symbol_table)

label("symbol round-trip restores original",
      restored_text == normalized,
      repr(restored_text[:80]), repr(normalized[:80]))
print(f"  symbols encoded : {len(symbol_table)}")


# =========================================================================
# STAGE 5  — encode_sentence_full (real spaCy output)
# =========================================================================
print(SEP)
print("STAGE 5 — encode_sentence_full")
print(SEP)

from compression.pipeline.stage5_encode import (
    CharacterEncoder, StructuralEncoder,
    _expand_morph_codes_for_chars, _expand_pos_tags_for_chars,
)
from compression.alphabet.phonetic_map import PHONETIC_CLASSES, compute_deltas
from compression.alphabet.symbol_alphabet import SymbolAlphabet

inverse_map = {coords: ch for ch, coords in PHONETIC_CLASSES.items()}

char_enc   = CharacterEncoder()
sym_alpha  = SymbolAlphabet()
struct_enc = StructuralEncoder(sym_alpha)
freq_table = struct_enc.build_pos_frequency_table([syntax.pos_tags])

encoded = char_enc.encode_sentence_full(morph_results, syntax, struct_enc, freq_table)

roots            = encoded["roots"]
morph_codes      = encoded["morph_codes"]
pos_tags         = encoded["pos_tags"]
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
label("no root contains BOM",
      not any("\ufeff" in r for r in roots),
      got=[r for r in roots if "\ufeff" in r], expected="[]")
print(f"  roots[:5]        : {roots[:5]}")

# Verify class+delta → char sequence round-trip
from main import _cumulative_from_deltas
positions = _cumulative_from_deltas(pos_deltas_vals)
reconstructed_seq = "".join(
    inverse_map.get((cls, pos), "?") for cls, pos in zip(char_classes, positions)
)
expected_seq = "^" + "$_^".join(roots) + "$"
label("class+delta → char sequence",
      reconstructed_seq == expected_seq,
      repr(reconstructed_seq[:60]), repr(expected_seq[:60]))


# =========================================================================
# STAGE 5C  — _build_encoded_sentences_from_metadata reconstruction
#             This is what decompress actually uses — must match encoder exactly
# =========================================================================
print(SEP)
print("STAGE 5C — metadata reconstruction (the REAL decompress path)")
print(SEP)

root_lengths = [len(r) for r in roots]

recon_morph: List[int] = []
recon_pos:   List[str] = []
for token_idx, length in enumerate(root_lengths):
    pt = pos_tags[token_idx]    if token_idx < len(pos_tags)    else "X"
    mc = morph_codes[token_idx] if token_idx < len(morph_codes) else 0
    recon_morph.append(0)
    recon_morph.extend([mc] * length)
    recon_morph.append(0)
    recon_pos.append("X")
    recon_pos.extend([pt] * length)
    recon_pos.append("X")
    if token_idx < len(root_lengths) - 1:
        recon_morph.append(0)
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
    print(f"  recon len={len(recon_pos)}  orig len={len(char_pos_tags)}")


# =========================================================================
# STAGE 6  — ContextMixingModel
# =========================================================================
print(SEP)
print("STAGE 6 — ContextMixingModel")
print(SEP)

from compression.pipeline.stage6_probability import ContextMixingModel

# The ENCODER builds fake_sentence using the real encoder outputs
fake_sentence_enc = {
    "char_classes":     char_classes,
    "char_morph_codes": char_morph_codes,
    "char_pos_tags":    char_pos_tags,
    "pos_tags":         pos_tags,
    "morph_codes":      morph_codes,
    "roots":            roots,
    "pos_huffman_bits": float(encoded["pos_huffman_bits"]),
    "pos_n_tags":       int(encoded["pos_n_tags"]),
}

# The DECODER builds fake_sentence using the RECONSTRUCTED streams
fake_sentence_dec = {
    "char_classes":     char_classes,      # same — stored in file
    "char_morph_codes": recon_morph,       # reconstructed
    "char_pos_tags":    recon_pos,         # reconstructed
    "pos_tags":         pos_tags,
    "morph_codes":      morph_codes,
    "roots":            roots,
    "pos_huffman_bits": float(encoded["pos_huffman_bits"]),
    "pos_n_tags":       int(encoded["pos_n_tags"]),
}

cm_enc = ContextMixingModel()
cm_enc.train([fake_sentence_enc])

cm_dec = ContextMixingModel()
cm_dec.train([fake_sentence_dec])

# Compare probability distributions at every position
mismatches = 0
for i in range(len(char_classes)):
    ctx_enc = {
        "char_history":        char_classes[:i],
        "current_morph_code":  char_morph_codes[i],
        "current_pos_tag":     char_pos_tags[i],
        "struct_prob":         0.5,
    }
    ctx_dec = {
        "char_history":        char_classes[:i],
        "current_morph_code":  recon_morph[i],
        "current_pos_tag":     recon_pos[i],
        "struct_prob":         0.5,
    }
    dist_enc = cm_enc.probability_distribution(ctx_enc)
    dist_dec = cm_dec.probability_distribution(ctx_dec)
    if dist_enc != dist_dec:
        mismatches += 1
        if mismatches == 1:
            print(f"  first prob mismatch @ position {i}:")
            print(f"    encoder dist : {dict(list(dist_enc.items())[:4])}...")
            print(f"    decoder dist : {dict(list(dist_dec.items())[:4])}...")
            print(f"    enc morph={char_morph_codes[i]}  dec morph={recon_morph[i]}")
            print(f"    enc pos={char_pos_tags[i]!r}  dec pos={recon_pos[i]!r}")

label("enc vs dec prob distributions match at all positions",
      mismatches == 0,
      f"{mismatches} mismatches", "0 mismatches")


# =========================================================================
# STAGE 7  — arithmetic encode (encoder context) + decode (decoder context)
#            This is the real failure scenario
# =========================================================================
print(SEP)
print("STAGE 7 — arithmetic coder: encode with enc_ctx, decode with dec_ctx")
print(SEP)

from compression.pipeline.stage7_arithmetic import ArithmeticEncoder, ArithmeticDecoder

enc = ArithmeticEncoder()
bitstream = enc.encode(char_classes, cm_enc, {}, [fake_sentence_enc])

dec = ArithmeticDecoder()
decoded_classes = dec.decode(bitstream, cm_dec, [fake_sentence_dec], len(char_classes))

label("decoded length == original",
      len(decoded_classes) == len(char_classes),
      len(decoded_classes), len(char_classes))
label("decoded classes == original",
      decoded_classes == char_classes,
      decoded_classes[:10], char_classes[:10])

if decoded_classes != char_classes:
    for i, (a, b) in enumerate(zip(decoded_classes, char_classes)):
        if a != b:
            print(f"  first class mismatch @ index {i}: decoded={a}, original={b}")
            break


# =========================================================================
# STAGE 7B  — pos_deltas round-trip
# =========================================================================
print(SEP)
print("STAGE 7B — pos_deltas arithmetic round-trip")
print(SEP)

counts = Counter(pos_deltas_vals)
enc2 = ArithmeticEncoder()
pd_bitstream = enc2.encode_unigram_counts(pos_deltas_vals, counts)
dec2 = ArithmeticDecoder()
decoded_deltas = dec2.decode_unigram_counts(pd_bitstream, counts, len(pos_deltas_vals))

label("pos_deltas round-trip",
      decoded_deltas == pos_deltas_vals,
      decoded_deltas[:10], pos_deltas_vals[:10])


# =========================================================================
# STAGE 8  — _reconstruct_chars with DECODED (possibly corrupted) streams
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
label("split roots match", roots_out == roots, roots_out[:5], roots[:5])

words_out = [
    apply_morph(r, morph_codes[i] if i < len(morph_codes) else 0)
    for i, r in enumerate(roots_out)
]
final = _join_words(words_out)
expected_final = first_sent.text.strip()
label("capitalisation restored",
      final.lower() == expected_final.lower(),
      repr(final[:80]), repr(expected_final[:80]))
label("final text exact match",
      final == expected_final,
      repr(final[:80]), repr(expected_final[:80]))


# =========================================================================
# STAGE 10  — full compress_to_file → decompress round-trip on moby500.txt
#             The real end-to-end test
# =========================================================================
print(SEP)
print("STAGE 10 — full compress → decompress round-trip")
print(SEP)

from main import compress_to_file, decompress

compress_to_file(normalized, "/tmp/trace_test.lexis")
decoded_text = decompress("/tmp/trace_test.lexis")

label("decoded length == original",
      len(decoded_text) == len(normalized),
      len(decoded_text), len(normalized))
label("decoded text == original",
      decoded_text == normalized,
      repr(decoded_text[:120]), repr(normalized[:120]))

# Show first character-level mismatch
if decoded_text != normalized:
    for i, (a, b) in enumerate(zip(decoded_text, normalized)):
        if a != b:
            print(f"  first char mismatch @ index {i}:")
            print(f"    decoded  : {repr(decoded_text[max(0,i-10):i+20])}")
            print(f"    original : {repr(normalized[max(0,i-10):i+20])}")
            # find which sentence this index belongs to
            sentences = list(doc.sents)
            pos = 0
            for s_idx, sent in enumerate(sentences):
                sent_len = len(sent.text)
                if pos + sent_len >= i:
                    print(f"  in sentence #{s_idx}: {repr(sent.text[:60])}")
                    print(f"  offset within sentence: {i - pos}")
                    break
                pos += sent_len + 1
            break

print(SEP)
print("Done.")
