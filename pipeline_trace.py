"""
pipeline_trace.py  —  end-to-end per-stage expected vs actual comparison.

Uses a tiny 3-word sentence so you can read every number by eye.

Run:  python pipeline_trace.py
"""
from __future__ import annotations

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


# ---------------------------------------------------------------------------
# Fixed tiny corpus — no spaCy needed for the core checks
# ---------------------------------------------------------------------------
SENTENCE = "call me ishmael"    # 3 roots, all lowercase
ROOTS    = ["call", "me", "ishmael"]
MORPH_CODES = [0, 0, 0]        # identity morph (no inflection change)
POS_TAGS    = ["VERB", "PRON", "NOUN"]


# =========================================================================
# STAGE 5A  — _triples_for_sentence  /  char_classes  /  pos_deltas
# =========================================================================
print(SEP)
print("STAGE 5A — character encoding (phonetic triples)")
print(SEP)

from compression.alphabet.phonetic_map import PHONETIC_CLASSES
from compression.pipeline.stage5_encode import (
    _triples_for_sentence,
    _char_classes_from_triples,
    _expand_morph_codes_for_chars,
    _expand_pos_tags_for_chars,
)
from compression.alphabet.phonetic_map import compute_deltas

triples    = _triples_for_sentence(ROOTS)
expected_chars = "^call$_^me$_^ishmael$"
actual_chars_list = []
inverse_map = {coords: ch for ch, coords in PHONETIC_CLASSES.items()}
for cls, pos, _ in triples:
    ch = inverse_map.get((cls, pos), "?")
    actual_chars_list.append(ch)
actual_chars = "".join(actual_chars_list)
label("sequence with markers", actual_chars == expected_chars,
      actual_chars, expected_chars)

char_classes = _char_classes_from_triples(triples)
print(f"  char_classes      = {char_classes}")

class_deltas, pos_deltas_vals, _ = compute_deltas(triples)
print(f"  class_deltas      = {class_deltas}")
print(f"  pos_deltas        = {pos_deltas_vals}")


# =========================================================================
# STAGE 5B  — _expand_morph_codes / _expand_pos_tags length match
# =========================================================================
print(SEP)
print("STAGE 5B — morph/pos context streams match char_classes length")
print(SEP)

char_morph_codes = _expand_morph_codes_for_chars(ROOTS, MORPH_CODES)
char_pos_tags    = _expand_pos_tags_for_chars(ROOTS, POS_TAGS)

label("morph stream length == char_classes",
      len(char_morph_codes) == len(char_classes),
      len(char_morph_codes), len(char_classes))
label("pos stream length == char_classes",
      len(char_pos_tags) == len(char_classes),
      len(char_pos_tags), len(char_classes))
print(f"  char_morph_codes  = {char_morph_codes}")
print(f"  char_pos_tags     = {char_pos_tags}")


# =========================================================================
# STAGE 5C  — _build_encoded_sentences_from_metadata reconstruction
#             Must produce IDENTICAL char_morph_codes / char_pos_tags
# =========================================================================
print(SEP)
print("STAGE 5C — metadata reconstruction matches original streams")
print(SEP)

root_lengths = [len(r) for r in ROOTS]

# Replicate _build_encoded_sentences_from_metadata logic exactly
recon_morph: List[int] = []
recon_pos:   List[str] = []
for token_idx, length in enumerate(root_lengths):
    pos_tag    = POS_TAGS[token_idx]    if token_idx < len(POS_TAGS)    else "X"
    morph_code = MORPH_CODES[token_idx] if token_idx < len(MORPH_CODES) else 0
    recon_morph.append(0)           # start marker (BASE = 0)
    recon_morph.extend([morph_code] * length)
    recon_morph.append(0)           # end marker
    recon_pos.append("X")           # start marker
    recon_pos.extend([pos_tag] * length)
    recon_pos.append("X")           # end marker
    if token_idx < len(root_lengths) - 1:
        recon_morph.append(0)       # space marker
        recon_pos.append("X")      # space marker

label("recon morph stream == original",
      recon_morph == char_morph_codes,
      recon_morph, char_morph_codes)
label("recon pos stream == original",
      recon_pos == char_pos_tags,
      recon_pos, char_pos_tags)

# If they differ, show a diff
if recon_morph != char_morph_codes:
    for i, (a, b) in enumerate(zip(recon_morph, char_morph_codes)):
        if a != b:
            print(f"  first morph mismatch @ index {i}: recon={a}, orig={b}")
            break
if recon_pos != char_pos_tags:
    for i, (a, b) in enumerate(zip(recon_pos, char_pos_tags)):
        if a != b:
            print(f"  first pos mismatch   @ index {i}: recon={a}, orig={b}")
            break


# =========================================================================
# STAGE 6  — ContextMixingModel train + probability_distribution sanity
# =========================================================================
print(SEP)
print("STAGE 6 — context model probabilities sum to 1.0 per symbol")
print(SEP)

from compression.pipeline.stage6_probability import ContextMixingModel

# Build a minimal fake encoded_sentences that matches the real structure
fake_sentence = {
    "char_classes":    char_classes,
    "char_morph_codes": char_morph_codes,
    "char_pos_tags":   char_pos_tags,
    "pos_tags":        POS_TAGS,
    "morph_codes":     MORPH_CODES,
    "roots":           ROOTS,
    "pos_huffman_bits": 5.0,
    "pos_n_tags":       3,
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
label("prob distribution sums to 1.0", abs(total - 1.0) < 1e-9, total, 1.0)
label("all symbols in 0-6", all(0 <= s <= 6 for s in dist), list(dist.keys()), "0-6")


# =========================================================================
# STAGE 7  — ArithmeticEncoder → ArithmeticDecoder round-trip
#            Using the EXACT same context streams both sides
# =========================================================================
print(SEP)
print("STAGE 7 — arithmetic coder round-trip (context-aware)")
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

# Show first mismatch if any
if decoded_classes != char_classes:
    for i, (a, b) in enumerate(zip(decoded_classes, char_classes)):
        if a != b:
            print(f"  first class mismatch @ index {i}: decoded={a}, original={b}")
            ch_orig = inverse_map.get((b, 0), "?")
            print(f"  (original class {b} corresponds to chars: "
                  f"{[c for c,(cl,_) in PHONETIC_CLASSES.items() if cl==b][:5]})")
            break


# =========================================================================
# STAGE 7B — pos_deltas round-trip
# =========================================================================
print(SEP)
print("STAGE 7B — pos_deltas arithmetic round-trip")
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
# STAGE 8  — _reconstruct_chars  (class_stream + pos_deltas → text)
# =========================================================================
print(SEP)
print("STAGE 8 — _reconstruct_chars")
print(SEP)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from main import _reconstruct_chars

# Use the *decoded* streams (as decompress would)
single_sentence_count = [len(char_classes)]
reconstructed = _reconstruct_chars(decoded_classes, decoded_deltas, single_sentence_count)
expected_char_stream = "^call$_^me$_^ishmael$"
label("_reconstruct_chars output",
      reconstructed == expected_char_stream,
      repr(reconstructed), repr(expected_char_stream))


# =========================================================================
# STAGE 9 — morph decode  (roots + morph_codes → words)
# =========================================================================
print(SEP)
print("STAGE 9 — morph decode")
print(SEP)

from compression.alphabet.morph_codes import apply_morph
from main import _split_roots, _join_words

roots_out = _split_roots(reconstructed)
label("_split_roots", roots_out == ROOTS, roots_out, ROOTS)

words_out = [apply_morph(r, MORPH_CODES[i] if i < len(MORPH_CODES) else 0)
             for i, r in enumerate(roots_out)]
final = _join_words(words_out)
expected_final = "Call me ishmael"
label("final text", final.lower() == SENTENCE, final, SENTENCE)

print(SEP)
print("Done.")
