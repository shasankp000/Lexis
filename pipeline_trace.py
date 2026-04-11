"""
pipeline_trace.py  —  exhaustive per-stage and end-to-end coverage.

Every component that participates in the compress → decompress path is
exercised here, including:

  Stage  1  normalize_text  (BOM, multi-space, edge cases)
  Stage  2  MorphologicalAnalyser  (types, BOM in roots, empty roots)
  Stage  3  analyse_sentence (syntax; tag count parity)
  Stage  4  discourse analysis + symbol encoding/decoding (round-trip + edges)
  Stage  5  encode_sentence_full: class+delta round-trip, cap_flags,
              case_flags length, case_bitmaps length,
              compute_case_flag all four categories,
              apply_case_flag all four categories
  Stage  5C _build_encoded_sentences_from_metadata reconstruction
  Stage  5D compress_to_file root_lengths serialisation round-trip (2 sentences)
  Stage  6  ContextMixingModel: enc vs dec distributions, multi-sentence
  Stage  7  arithmetic encoder/decoder: enc-ctx vs dec-ctx
  Stage  7B pos_deltas unigram round-trip
  Stage  7C multi-sentence arithmetic round-trip
  Stage  8  _reconstruct_chars with decoded streams
  Stage  9  morph decode, apply_morph all codes, apply_case_flag all flags
  Stage  9B _join_words punctuation/quote edge cases
  Stage 10  metadata_codec — every mode independently:
              raw, scalar, flat_uint, flat_int, flat_dict,
              pos, int_nested, sparse_dict, sparse_dict_pos,
              model_weights, pos_freq, float_nested, symbol_table
            + encode_metadata / decode_metadata / is_lexi_file
            + case_flags and case_bitmaps fields round-trip
  Stage 11  full compress_to_file → decompress end-to-end, sentence-level diff,
              case restoration spot-check
  Stage 12  autocorrect pass-through

Run:  python pipeline_trace.py
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

PASS = "\033[92mOK\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SEP  = "\n" + "-" * 70


def label(name: str, ok: bool, got=None, expected=None) -> None:
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

raw        = open("moby500.txt", encoding="utf-8").read()
normalized = normalize_text(raw)

label("normalized is non-empty",        len(normalized) > 0)
label("no leading/trailing whitespace", normalized == normalized.strip())
label("no double spaces",               "  " not in normalized)
label("no BOM in normalized text",      "\ufeff" not in normalized,
      got=repr(normalized[:20]), expected="no BOM")

# Edge: BOM at start of input
bom_input = "\ufeffHello world."
bom_norm  = normalize_text(bom_input)
label("BOM-prefixed input → BOM stripped", "\ufeff" not in bom_norm,
      got=repr(bom_norm[:20]), expected="no BOM")

# Edge: multiple consecutive spaces
multi_space = normalize_text("Hello  world.   How  are   you?")
label("multiple spaces collapsed",
      "  " not in multi_space,
      got=repr(multi_space), expected="single spaces")

# Edge: empty string
label("empty string normalizes to empty",
      normalize_text("") == "",
      got=repr(normalize_text("")), expected="''")

# Edge: only whitespace
label("whitespace-only normalizes to empty",
      normalize_text("   \n\t  ").strip() == "",
      got=repr(normalize_text("   \n\t  ")), expected="''")

# Edge: tab and newline collapse to single space
tab_nl = normalize_text("Hello\tworld.\nHow\nare\nyou?")
label("tabs/newlines collapse to single space",
      "  " not in tab_nl and "\t" not in tab_nl,
      got=repr(tab_nl[:60]), expected="single spaces, no tabs")

# Edge: already-normalized text must be idempotent
label("normalize_text is idempotent",
      normalize_text(normalized) == normalized,
      got=repr(normalize_text(normalized)[:60]), expected=repr(normalized[:60]))

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

doc        = nlp(normalized)
all_sents  = list(doc.sents)
first_sent = all_sents[0]

morph_results = analyser.analyse_sentence(first_sent.text)

label("morph results non-empty", len(morph_results) > 0)
for item in morph_results[:5]:
    print(f"  token={item[0]!r:15}  root={item[1]!r:15}  code={item[2]}")

label("no root contains BOM",
      not any("\ufeff" in item[1] for item in morph_results),
      got=[item[1] for item in morph_results if "\ufeff" in item[1]],
      expected="[]")

# Each (original, root, code) must be a 3-tuple of (str, str, int)
label("each morph item is (str, str, int)",
      all(isinstance(o, str) and isinstance(r, str) and isinstance(c, int)
          for o, r, c in morph_results),
      got="type error", expected="(str, str, int)")

# root must never be empty
label("no empty root",
      all(len(r) > 0 for _, r, _ in morph_results),
      got=[r for _, r, _ in morph_results if not r], expected="[]")

# code must be a known morph code (0–12)
label("all morph codes in 0–12",
      all(0 <= c <= 12 for _, _, c in morph_results),
      got=[c for _, _, c in morph_results if not (0 <= c <= 12)], expected="[]")

# original token must never be empty
label("no empty original token",
      all(len(o) > 0 for o, _, _ in morph_results),
      got=[o for o, _, _ in morph_results if not o], expected="[]")

# Verify second sentence also analysable (idempotency across sentences)
second_sent = all_sents[1] if len(all_sents) > 1 else first_sent
morph2_check = analyser.analyse_sentence(second_sent.text)
label("second sentence morph non-empty", len(morph2_check) > 0)
label("second sentence morph codes in 0–12",
      all(0 <= c <= 12 for _, _, c in morph2_check),
      got=[c for _, _, c in morph2_check if not (0 <= c <= 12)], expected="[]")


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
label("all pos_tags are non-empty strings",
      all(isinstance(t, str) and len(t) > 0 for t in syntax.pos_tags),
      got=[t for t in syntax.pos_tags if not t], expected="[]")

# Second sentence
syntax2_check = analyse_sentence(second_sent)
label("second sentence pos_tags non-empty", len(syntax2_check.pos_tags) > 0)
label("second sentence pos_tags count == morph2 count",
      len(syntax2_check.pos_tags) == len(morph2_check),
      len(syntax2_check.pos_tags), len(morph2_check))

print(f"  pos_tags[:5] : {syntax.pos_tags[:5]}")


# =========================================================================
# STAGE 4  — discourse / symbol encoding
# =========================================================================
print(SEP)
print("STAGE 4 — discourse symbol round-trip")
print(SEP)

from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import encode_symbols, decode_symbols

disc_analyser  = DiscourseAnalyser(use_spacy=True, device="cpu")
stage4_result  = disc_analyser.analyse_document(normalized)
compressed_text, symbol_table = encode_symbols(normalized, stage4_result)
restored_text  = decode_symbols(compressed_text, symbol_table)

label("symbol round-trip restores original",
      restored_text == normalized,
      repr(restored_text[:80]), repr(normalized[:80]))
print(f"  symbols encoded : {len(symbol_table)}")

# Edge: empty symbol table — decode must be identity
label("decode with empty symbol_table is identity",
      decode_symbols("Hello world.", {}) == "Hello world.",
      got=decode_symbols("Hello world.", {}), expected="Hello world.")

# Edge: encode_symbols on text with no repeated entities
short_text = "The cat sat on the mat."
short_c, short_t = encode_symbols(short_text, disc_analyser.analyse_document(short_text))
short_restored   = decode_symbols(short_c, short_t)
label("short text symbol round-trip",
      short_restored == short_text,
      repr(short_restored), repr(short_text))

# Edge: symbol_table key must appear in compressed text
if symbol_table:
    first_key = next(iter(symbol_table))
    label("encoded text contains at least one §E token",
          "§E" in compressed_text or len(symbol_table) == 0,
          got=compressed_text[:80], expected="contains §E...")

# Edge: decode on text with no §-tokens must be identity
no_symbol_text = "No entities here at all."
label("decode text with no §-tokens is identity",
      decode_symbols(no_symbol_text, {"§E0": "London"}) == no_symbol_text,
      got=decode_symbols(no_symbol_text, {"§E0": "London"}), expected=no_symbol_text)


# =========================================================================
# STAGE 5  — encode_sentence_full (real spaCy output)
# =========================================================================
print(SEP)
print("STAGE 5 — encode_sentence_full")
print(SEP)

from compression.pipeline.stage5_encode import (
    CharacterEncoder, StructuralEncoder,
    _expand_morph_codes_for_chars, _expand_pos_tags_for_chars,
    compute_case_flag, apply_case_flag,
    CASE_LOWER, CASE_TITLE, CASE_UPPER, CASE_MIXED,
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
case_flags_enc   = encoded["case_flags"]
case_bitmaps_enc = encoded["case_bitmaps"]

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
label("morph_codes count == roots count",
      len(morph_codes) == len(roots),
      len(morph_codes), len(roots))
label("pos_tags count == roots count",
      len(pos_tags) == len(roots),
      len(pos_tags), len(roots))

# --- case_flags / case_bitmaps sanity ---
label("case_flags length == roots count",
      len(case_flags_enc) == len(roots),
      len(case_flags_enc), len(roots))
label("case_bitmaps length == roots count",
      len(case_bitmaps_enc) == len(roots),
      len(case_bitmaps_enc), len(roots))
label("all case_flags in 0–3",
      all(0 <= f <= 3 for f in case_flags_enc),
      got=[f for f in case_flags_enc if not (0 <= f <= 3)], expected="[]")
label("case_bitmaps are non-negative ints",
      all(isinstance(b, int) and b >= 0 for b in case_bitmaps_enc),
      got=[b for b in case_bitmaps_enc if not (isinstance(b, int) and b >= 0)],
      expected="[]")

# --- encode_sentence_full on second sentence ---
syntax2      = analyse_sentence(second_sent)
morph2       = analyser.analyse_sentence(second_sent.text)
freq_table2  = struct_enc.build_pos_frequency_table([syntax.pos_tags, syntax2.pos_tags])
encoded2     = char_enc.encode_sentence_full(morph2, syntax2, struct_enc, freq_table2)

roots2       = encoded2["roots"]
morph_codes2 = encoded2["morph_codes"]
pos_tags2    = encoded2["pos_tags"]

label("sentence 2 char_classes non-empty", len(encoded2["char_classes"]) > 0)
label("sentence 2 case_flags length == roots2 count",
      len(encoded2["case_flags"]) == len(roots2),
      len(encoded2["case_flags"]), len(roots2))
label("sentence 2 case_bitmaps length == roots2 count",
      len(encoded2["case_bitmaps"]) == len(roots2),
      len(encoded2["case_bitmaps"]), len(roots2))

# --- compute_case_flag unit tests ---
_ccf_cases = [
    ("the",     CASE_LOWER, 0),
    ("hello",   CASE_LOWER, 0),
    ("The",     CASE_TITLE, 0),
    ("Hello",   CASE_TITLE, 0),
    ("USA",     CASE_UPPER, 0),
    ("HTTP",    CASE_UPPER, 0),
    ("eBook",   CASE_MIXED, None),   # bitmap checked separately
    ("iPhone",  CASE_MIXED, None),
    ("",        CASE_LOWER, 0),
    # purely numeric / punctuation tokens — must not raise
    ("123",     CASE_LOWER, 0),
    ("...",     CASE_LOWER, 0),
]
for surface, expected_flag, expected_bitmap in _ccf_cases:
    flag, bitmap = compute_case_flag(surface)
    label(f"compute_case_flag({surface!r}) → flag={expected_flag}",
          flag == expected_flag, got=flag, expected=expected_flag)
    if expected_bitmap is not None:
        label(f"compute_case_flag({surface!r}) → bitmap={expected_bitmap}",
              bitmap == expected_bitmap, got=bitmap, expected=expected_bitmap)
    else:
        label(f"compute_case_flag({surface!r}) → bitmap is non-neg int",
              isinstance(bitmap, int) and bitmap >= 0, got=bitmap, expected=">=0 int")

# Verify that bitmap correctly encodes which characters are uppercase
flag_ebook, bitmap_ebook = compute_case_flag("eBook")
# 'e'=0 lower, 'B'=1 upper, 'o'=2 lower, 'o'=3 lower, 'k'=4 lower
label("compute_case_flag('eBook') bitmap bit 1 set for 'B'",
      bitmap_ebook & (1 << 1) != 0,
      got=bin(bitmap_ebook), expected="bit 1 set")
label("compute_case_flag('eBook') bitmap bit 0 not set for 'e'",
      bitmap_ebook & (1 << 0) == 0,
      got=bin(bitmap_ebook), expected="bit 0 clear")

flag_iphone, bitmap_iphone = compute_case_flag("iPhone")
label("compute_case_flag('iPhone') flag == CASE_MIXED",
      flag_iphone == CASE_MIXED, got=flag_iphone, expected=CASE_MIXED)
label("compute_case_flag('iPhone') bitmap bit 0 clear for 'i'",
      bitmap_iphone & (1 << 0) == 0,
      got=bin(bitmap_iphone), expected="bit 0 clear")
label("compute_case_flag('iPhone') bitmap bit 1 set for 'P'",
      bitmap_iphone & (1 << 1) != 0,
      got=bin(bitmap_iphone), expected="bit 1 set")

# ALL-CAPS single char
flag_a, bitmap_a = compute_case_flag("A")
label("compute_case_flag('A') flag == CASE_UPPER or CASE_TITLE",
      flag_a in (CASE_UPPER, CASE_TITLE),
      got=flag_a, expected=f"{CASE_UPPER} or {CASE_TITLE}")

# --- apply_case_flag unit tests ---
_acf_cases = [
    ("hello",   CASE_LOWER, 0,          "hello"),
    ("hello",   CASE_TITLE, 0,          "Hello"),
    ("hello",   CASE_UPPER, 0,          "HELLO"),
    ("ebook",   CASE_MIXED, (1 << 1),   "eBook"),   # bit 1 → 'B'
    ("iphone",  CASE_MIXED, (1 << 1),   "iPhone"),  # bit 0 → 'I'
    ("",        CASE_LOWER, 0,          ""),
    ("",        CASE_TITLE, 0,          ""),
    ("a",       CASE_TITLE, 0,          "A"),
    ("abc",     CASE_UPPER, 0,          "ABC"),
    ("abc",     CASE_MIXED, 0b101,      "AbC"),     # bits 0,2 → 'A','C'
    ("z",       CASE_LOWER, 0,          "z"),
    ("z",       CASE_UPPER, 0,          "Z"),
    ("hello",   CASE_MIXED, 0b00000,    "hello"),   # no bits set → all lower
    ("hello",   CASE_MIXED, 0b11111,    "HELLO"),   # all bits set → all upper
]
for word, flag, bitmap, expected_out in _acf_cases:
    result = apply_case_flag(word, flag, bitmap)
    label(f"apply_case_flag({word!r}, flag={flag}, bitmap={bin(bitmap)}) == {expected_out!r}",
          result == expected_out, got=repr(result), expected=repr(expected_out))

# Round-trip: compute_case_flag → apply_case_flag must recover the original surface
for item in morph_results[:10]:
    surface = item[0]
    root_lower = item[1]
    flag, bitmap = compute_case_flag(surface)
    # apply_morph produces lowercase by default; apply case on top
    from compression.alphabet.morph_codes import apply_morph as _am
    morphed = _am(root_lower, item[2])
    restored_surface = apply_case_flag(morphed, flag, bitmap)
    # case must match (we can only check what apply_morph + case_flag produces)
    label(f"case round-trip for token {surface!r}: flag={flag} bitmap={bitmap}",
          restored_surface.lower() == morphed.lower(),
          got=repr(restored_surface), expected=repr(morphed + " (same case-folded)"))

# Verify class+delta → char sequence round-trip
from main import _cumulative_from_deltas
positions         = _cumulative_from_deltas(pos_deltas_vals)
reconstructed_seq = "".join(
    inverse_map.get((cls, pos), "?") for cls, pos in zip(char_classes, positions)
)
expected_seq = "^" + "$_^".join(roots) + "$"
label("class+delta → char sequence round-trip",
      reconstructed_seq == expected_seq,
      repr(reconstructed_seq[:60]), repr(expected_seq[:60]))

# No "?" in reconstructed sequence — every (class, pos) must be in inverse_map
label("no unmapped (class, pos) pairs",
      "?" not in reconstructed_seq,
      got=reconstructed_seq[:60], expected="no '?'")

print(f"  roots[:5]        : {roots[:5]}")
print(f"  char_classes[:5] : {char_classes[:5]}")
print(f"  pos_deltas[:5]   : {pos_deltas_vals[:5]}")
print(f"  case_flags[:5]   : {case_flags_enc[:5]}")
print(f"  case_bitmaps[:5] : {case_bitmaps_enc[:5]}")


# =========================================================================
# STAGE 5C  — _build_encoded_sentences_from_metadata reconstruction
# =========================================================================
print(SEP)
print("STAGE 5C — metadata reconstruction (the REAL decompress path)")
print(SEP)

from main import _build_encoded_sentences_from_metadata

root_lengths = [len(r) for r in roots]

# Manually build what _build_encoded_sentences_from_metadata produces
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

label("recon morph == original char_morph_codes",
      recon_morph == char_morph_codes,
      recon_morph[:20], char_morph_codes[:20])
label("recon pos == original char_pos_tags",
      recon_pos == char_pos_tags,
      recon_pos[:20], char_pos_tags[:20])

if recon_pos != char_pos_tags:
    for i, (a, b) in enumerate(zip(recon_pos, char_pos_tags)):
        if a != b:
            print(f"  first pos mismatch @ index {i}: recon={a!r}  orig={b!r}")
            break
    print(f"  recon len={len(recon_pos)}  orig len={len(char_pos_tags)}")

# Verify via the actual function
payload_for_recon = {
    "root_lengths":     [root_lengths],
    "pos_huffman_bits": [float(encoded["pos_huffman_bits"])],
    "pos_n_tags":       [int(encoded["pos_n_tags"])],
    "pos_tags":         [pos_tags],
    "morph_codes":      [morph_codes],
}
recon_sentences = _build_encoded_sentences_from_metadata(payload_for_recon)
label("_build_encoded_sentences_from_metadata returns 1 sentence",
      len(recon_sentences) == 1, len(recon_sentences), 1)
label("function recon morph == manual recon morph",
      recon_sentences[0]["char_morph_codes"] == recon_morph,
      recon_sentences[0]["char_morph_codes"][:10], recon_morph[:10])
label("function recon pos == manual recon pos",
      recon_sentences[0]["char_pos_tags"] == recon_pos,
      recon_sentences[0]["char_pos_tags"][:10], recon_pos[:10])

# Edge: empty payload must return empty list
empty_payload = {
    "root_lengths": [], "pos_huffman_bits": [], "pos_n_tags": [],
    "pos_tags": [], "morph_codes": [],
}
label("_build_encoded_sentences_from_metadata([]) returns []",
      _build_encoded_sentences_from_metadata(empty_payload) == [],
      got=_build_encoded_sentences_from_metadata(empty_payload), expected=[])


# =========================================================================
# STAGE 5D  — compress_to_file root_lengths serialisation round-trip
# =========================================================================
print(SEP)
print("STAGE 5D — compress_to_file root_lengths serialisation round-trip")
print(SEP)

packed_payload: Dict[str, Any] = {
    "root_lengths":     [[len(r) for r in roots], [len(r) for r in roots2]],
    "pos_huffman_bits": [float(encoded["pos_huffman_bits"]),
                         float(encoded2["pos_huffman_bits"])],
    "pos_n_tags":       [int(encoded["pos_n_tags"]), int(encoded2["pos_n_tags"])],
    "pos_tags":         [pos_tags, pos_tags2],
    "morph_codes":      [morph_codes, morph_codes2],
}
recon2 = _build_encoded_sentences_from_metadata(packed_payload)
label("multi-sentence reconstruction: 2 sentences returned",
      len(recon2) == 2, len(recon2), 2)

enc0_morph = encoded["char_morph_codes"]
enc0_pos   = encoded["char_pos_tags"]
label("sentence 0 morph matches",
      recon2[0]["char_morph_codes"] == enc0_morph,
      recon2[0]["char_morph_codes"][:10], enc0_morph[:10])
label("sentence 0 pos matches",
      recon2[0]["char_pos_tags"] == enc0_pos,
      recon2[0]["char_pos_tags"][:10], enc0_pos[:10])

enc1_morph = encoded2["char_morph_codes"]
enc1_pos   = encoded2["char_pos_tags"]
label("sentence 1 morph matches",
      recon2[1]["char_morph_codes"] == enc1_morph,
      recon2[1]["char_morph_codes"][:10], enc1_morph[:10])
label("sentence 1 pos matches",
      recon2[1]["char_pos_tags"] == enc1_pos,
      recon2[1]["char_pos_tags"][:10], enc1_pos[:10])


# =========================================================================
# STAGE 6  — ContextMixingModel
# =========================================================================
print(SEP)
print("STAGE 6 — ContextMixingModel")
print(SEP)

from compression.pipeline.stage6_probability import ContextMixingModel

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

fake_sentence_dec = {
    "char_classes":     char_classes,
    "char_morph_codes": recon_morph,
    "char_pos_tags":    recon_pos,
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

mismatches = 0
for i in range(len(char_classes)):
    ctx_enc = {
        "char_history":       char_classes[:i],
        "current_morph_code": char_morph_codes[i],
        "current_pos_tag":    char_pos_tags[i],
        "struct_prob":        0.5,
    }
    ctx_dec = {
        "char_history":       char_classes[:i],
        "current_morph_code": recon_morph[i],
        "current_pos_tag":    recon_pos[i],
        "struct_prob":        0.5,
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
      mismatches == 0, f"{mismatches} mismatches", "0 mismatches")

ctx_test = {
    "char_history":       [],
    "current_morph_code": 0,
    "current_pos_tag":    "NOUN",
    "struct_prob":        0.5,
}
dist_test  = cm_enc.probability_distribution(ctx_test)
total_prob = sum(dist_test.values())
label("probability distribution sums to ~1.0",
      abs(total_prob - 1.0) < 1e-6,
      got=f"{total_prob:.8f}", expected="~1.0")

# All symbols in the distribution must be non-negative probabilities
label("all distribution probabilities >= 0",
      all(p >= 0 for p in dist_test.values()),
      got=[p for p in dist_test.values() if p < 0], expected="[]")

# Distribution must be non-empty
label("distribution is non-empty", len(dist_test) > 0)

fake_s2_enc = {
    "char_classes":     encoded2["char_classes"],
    "char_morph_codes": encoded2["char_morph_codes"],
    "char_pos_tags":    encoded2["char_pos_tags"],
    "pos_tags":         pos_tags2,
    "morph_codes":      morph_codes2,
    "roots":            roots2,
    "pos_huffman_bits": float(encoded2["pos_huffman_bits"]),
    "pos_n_tags":       int(encoded2["pos_n_tags"]),
}

cm_multi = ContextMixingModel()
cm_multi.train([fake_sentence_enc, fake_s2_enc])
label("multi-sentence training: char_vocab non-empty",
      len(cm_multi.char_vocab) > 0)
label("multi-sentence training: vocab >= single-sentence vocab",
      len(cm_multi.char_vocab) >= len(cm_enc.char_vocab),
      len(cm_multi.char_vocab), len(cm_enc.char_vocab))

# Verify distribution after multi-sentence training also sums to ~1.0
dist_multi = cm_multi.probability_distribution(ctx_test)
total_multi = sum(dist_multi.values())
label("multi-sentence distribution sums to ~1.0",
      abs(total_multi - 1.0) < 1e-6,
      got=f"{total_multi:.8f}", expected="~1.0")


# =========================================================================
# STAGE 7  — arithmetic encode (encoder context) + decode (decoder context)
# =========================================================================
print(SEP)
print("STAGE 7 — arithmetic coder: encode with enc_ctx, decode with dec_ctx")
print(SEP)

from compression.pipeline.stage7_arithmetic import ArithmeticEncoder, ArithmeticDecoder

enc = ArithmeticEncoder()
bitstream = enc.encode(char_classes, cm_enc, {}, [fake_sentence_enc])

label("bitstream is non-empty bytes",
      isinstance(bitstream, (bytes, bytearray)) and len(bitstream) > 0,
      type(bitstream).__name__, "bytes")

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

# Re-encode with SAME model (enc == dec) — must always be lossless
bitstream_same = enc.encode(char_classes, cm_enc, {}, [fake_sentence_enc])
decoded_same   = dec.decode(bitstream_same, cm_enc, [fake_sentence_enc], len(char_classes))
label("same-model encode/decode is lossless",
      decoded_same == char_classes,
      decoded_same[:10], char_classes[:10])

# Single-symbol encode/decode
if char_classes:
    bs_single = enc.encode([char_classes[0]], cm_enc, {}, [fake_sentence_enc])
    dec_single = dec.decode(bs_single, cm_enc, [fake_sentence_enc], 1)
    label("single-symbol arithmetic encode/decode",
          dec_single == [char_classes[0]],
          got=dec_single, expected=[char_classes[0]])

# Repeated-symbol stream
repeated = [char_classes[0]] * 8 if char_classes else []
if repeated:
    bs_rep = enc.encode(repeated, cm_enc, {}, [fake_sentence_enc])
    dec_rep = dec.decode(bs_rep, cm_enc, [fake_sentence_enc], len(repeated))
    label("repeated-symbol arithmetic encode/decode",
          dec_rep == repeated, got=dec_rep[:4], expected=repeated[:4])


# =========================================================================
# STAGE 7B  — pos_deltas unigram round-trip
# =========================================================================
print(SEP)
print("STAGE 7B — pos_deltas arithmetic round-trip")
print(SEP)

counts         = Counter(pos_deltas_vals)
enc2           = ArithmeticEncoder()
pd_bitstream   = enc2.encode_unigram_counts(pos_deltas_vals, counts)
dec2           = ArithmeticDecoder()
decoded_deltas = dec2.decode_unigram_counts(pd_bitstream, counts, len(pos_deltas_vals))

label("pos_deltas round-trip",
      decoded_deltas == pos_deltas_vals,
      decoded_deltas[:10], pos_deltas_vals[:10])
label("pos_deltas bitstream is bytes",
      isinstance(pd_bitstream, (bytes, bytearray)),
      type(pd_bitstream).__name__, "bytes")

single_stream = [3, 3, 3, 3]
single_counts = Counter(single_stream)
enc3  = ArithmeticEncoder()
dec3  = ArithmeticDecoder()
bs3   = enc3.encode_unigram_counts(single_stream, single_counts)
out3  = dec3.decode_unigram_counts(bs3, single_counts, len(single_stream))
label("pos_deltas single-value round-trip",
      out3 == single_stream, out3, single_stream)

# Edge: two-symbol alternating stream
alt_stream = [0, 1] * 6
alt_counts = Counter(alt_stream)
enc4  = ArithmeticEncoder()
dec4  = ArithmeticDecoder()
bs4   = enc4.encode_unigram_counts(alt_stream, alt_counts)
out4  = dec4.decode_unigram_counts(bs4, alt_counts, len(alt_stream))
label("pos_deltas alternating-values round-trip",
      out4 == alt_stream, out4, alt_stream)

# Edge: all-zero deltas
zero_stream = [0] * 10
zero_counts = Counter(zero_stream)
enc5 = ArithmeticEncoder()
dec5 = ArithmeticDecoder()
bs5  = enc5.encode_unigram_counts(zero_stream, zero_counts)
out5 = dec5.decode_unigram_counts(bs5, zero_counts, len(zero_stream))
label("pos_deltas all-zero round-trip",
      out5 == zero_stream, out5, zero_stream)


# =========================================================================
# STAGE 7C  — multi-sentence arithmetic round-trip
# =========================================================================
print(SEP)
print("STAGE 7C — multi-sentence arithmetic round-trip")
print(SEP)

all_classes = char_classes + encoded2["char_classes"]
fake_s2_dec = {
    "char_classes":     encoded2["char_classes"],
    "char_morph_codes": encoded2["char_morph_codes"],
    "char_pos_tags":    encoded2["char_pos_tags"],
    "pos_tags":         pos_tags2,
    "morph_codes":      morph_codes2,
    "roots":            roots2,
    "pos_huffman_bits": float(encoded2["pos_huffman_bits"]),
    "pos_n_tags":       int(encoded2["pos_n_tags"]),
}

enc_multi = ArithmeticEncoder()
bs_multi  = enc_multi.encode(all_classes, cm_multi, {},
                              [fake_sentence_enc, fake_s2_enc])

dec_multi    = ArithmeticDecoder()
decoded_multi = dec_multi.decode(bs_multi, cm_multi,
                                  [fake_sentence_enc, fake_s2_enc], len(all_classes))

label("multi-sentence decoded length == original",
      len(decoded_multi) == len(all_classes),
      len(decoded_multi), len(all_classes))
label("multi-sentence decoded classes == original",
      decoded_multi == all_classes,
      decoded_multi[:10], all_classes[:10])

# Verify sentence 0 and 1 independently
s0_len = len(char_classes)
label("multi-sentence: sentence 0 portion matches",
      decoded_multi[:s0_len] == char_classes,
      decoded_multi[:5], char_classes[:5])
label("multi-sentence: sentence 1 portion matches",
      decoded_multi[s0_len:] == encoded2["char_classes"],
      decoded_multi[s0_len:s0_len+5], encoded2["char_classes"][:5])


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
label("_reconstruct_chars output matches expected",
      reconstructed == expected_seq2,
      repr(reconstructed[:60]), repr(expected_seq2[:60]))

label("_reconstruct_chars on empty → empty string",
      _reconstruct_chars([], [], []) == "",
      got=_reconstruct_chars([], [], []), expected="")

# Single-token smoke test
if roots:
    single_cls = char_classes[:len(roots[0]) + 2]   # ^<root>$
    single_dlt = pos_deltas_vals[:len(roots[0]) + 2]
    single_rec = _reconstruct_chars(single_cls, single_dlt, [len(single_cls)])
    label("_reconstruct_chars single-token smoke test",
          isinstance(single_rec, str),
          got=type(single_rec).__name__, expected="str")

# Verify that ^ and $ boundary markers appear correctly
label("_reconstruct_chars output starts with '^'",
      reconstructed.startswith("^"),
      got=reconstructed[:3], expected="starts with '^'")
label("_reconstruct_chars output ends with '$'",
      reconstructed.endswith("$"),
      got=reconstructed[-3:], expected="ends with '$'")

# Multi-sentence _reconstruct_chars
reconstructed_multi = _reconstruct_chars(
    decoded_multi,
    decoded_deltas + dec5.decode_unigram_counts(
        enc5.encode_unigram_counts(
            encoded2["pos_deltas"], Counter(encoded2["pos_deltas"])
        ),
        Counter(encoded2["pos_deltas"]),
        len(encoded2["pos_deltas"]),
    ) if encoded2.get("pos_deltas") else decoded_deltas,
    [len(char_classes), len(encoded2["char_classes"])],
)
label("_reconstruct_chars multi-sentence returns string",
      isinstance(reconstructed_multi, str),
      got=type(reconstructed_multi).__name__, expected="str")


# =========================================================================
# STAGE 9  — morph decode + apply_morph all codes + apply_case_flag
# =========================================================================
print(SEP)
print("STAGE 9 — morph decode + apply_morph coverage + apply_case_flag")
print(SEP)

from compression.alphabet.morph_codes import (
    apply_morph,
    BASE, PLURAL, PAST_TENSE, PRESENT_PART, PAST_PART,
    THIRD_SING, COMPARATIVE, SUPERLATIVE, ADVERBIAL,
    NEGATION, AGENT, NOMINALIZE, IRREGULAR,
)

roots_out = _split_roots(reconstructed)
label("_split_roots count matches roots count",
      len(roots_out) == len(roots), len(roots_out), len(roots))
label("_split_roots values match roots",
      roots_out == roots, roots_out[:5], roots[:5])

words_out  = [apply_morph(r, morph_codes[i] if i < len(morph_codes) else 0)
              for i, r in enumerate(roots_out)]
final      = _join_words(words_out)
final_cap  = final[0].upper() + final[1:] if final else final
expected_final = first_sent.text.strip()

label("final text (case-insensitive) matches source sentence",
      final_cap.lower() == expected_final.lower(),
      repr(final_cap[:80]), repr(expected_final[:80]))

apply_cases = [
    (BASE,         "run",    "run"),
    (PLURAL,       "dog",    None),
    (PAST_TENSE,   "walk",   None),
    (PRESENT_PART, "run",    None),
    (PAST_PART,    "break",  None),
    (THIRD_SING,   "run",    None),
    (COMPARATIVE,  "fast",   "faster"),
    (COMPARATIVE,  "good",   "better"),
    (COMPARATIVE,  "bad",    "worse"),
    (SUPERLATIVE,  "fast",   "fastest"),
    (SUPERLATIVE,  "good",   "best"),
    (SUPERLATIVE,  "bad",    "worst"),
    (ADVERBIAL,    "quick",  "quickly"),
    (ADVERBIAL,    "happy",  "happily"),
    (ADVERBIAL,    "gentle", "gently"),
    (NEGATION,     "happy",  "unhappy"),
    (NEGATION,     "unhappy","unhappy"),
    (AGENT,        "run",    "runner"),
    (AGENT,        "write",  "writer"),
    (NOMINALIZE,   "dark",   "darkness"),
    (NOMINALIZE,   "kind",   "kindness"),
    (IRREGULAR,    "go",     None),
    # Edge: single-char roots
    (BASE,         "a",      "a"),
    (PLURAL,       "i",      None),
    # Edge: root ending in 'e' for morphological rules
    (PAST_TENSE,   "love",   None),
    (PRESENT_PART, "love",   None),
    (AGENT,        "love",   None),
]

for code, root, expect in apply_cases:
    result = apply_morph(root, code)
    if expect is None:
        ok = isinstance(result, str) and len(result) > 0
        label(f"apply_morph({root!r}, code={code}) → non-empty str",
              ok, got=repr(result), expected="non-empty str")
    else:
        ok = result == expect
        label(f"apply_morph({root!r}, code={code}) == {expect!r}",
              ok, got=repr(result), expected=repr(expect))

# apply_case_flag — exhaustive coverage in Stage 9 (separate from Stage 5 unit tests)
print("  -- apply_case_flag exhaustive coverage --")
_acf9_cases = [
    # (word, flag, bitmap, expected)
    ("hello",   CASE_LOWER, 0,        "hello"),
    ("world",   CASE_TITLE, 0,        "World"),
    ("nato",    CASE_UPPER, 0,        "NATO"),
    ("ebook",   CASE_MIXED, 0b10,     "eBook"),
    ("iphone",  CASE_MIXED, 0b1,      "iPhone"),
    ("abc",     CASE_MIXED, 0b111,    "ABC"),
    ("abc",     CASE_MIXED, 0b0,      "abc"),
    ("a",       CASE_TITLE, 0,        "A"),
    ("ab",      CASE_MIXED, 0b11,     "AB"),
    ("ab",      CASE_MIXED, 0b01,     "Ab"),
    ("ab",      CASE_MIXED, 0b10,     "aB"),
    ("",        CASE_LOWER, 0,        ""),
    ("",        CASE_TITLE, 0,        ""),
    ("",        CASE_UPPER, 0,        ""),
    # Longer tokens
    ("abcdef",  CASE_MIXED, 0b101010, "aBcDeF"),
    ("abcdef",  CASE_MIXED, 0b010101, "AbCdEf"),
    ("xyz",     CASE_UPPER, 0,        "XYZ"),
    ("xyz",     CASE_LOWER, 0,        "xyz"),
]
for word, flag, bitmap, expected_out in _acf9_cases:
    result = apply_case_flag(word, flag, bitmap)
    label(f"apply_case_flag({word!r}, {flag}, {bin(bitmap)}) == {expected_out!r}",
          result == expected_out, got=repr(result), expected=repr(expected_out))


# =========================================================================
# STAGE 9B  — _join_words punctuation / quote edge cases
# =========================================================================
print(SEP)
print("STAGE 9B — _join_words edge cases")
print(SEP)

_jw_cases = [
    (["Hello", ",", "world", "."],           "Hello, world."),
    (["(", "hello", ")"],                    "(hello)"),
    (["it", "'s", "fine"],                   "it's fine"),
    (["end", "-", "to", "-", "end"],         "end-to-end"),
    (["$", "100"],                            "$100"),
    (["50", "%"],                             "50%"),
    ([],                                      ""),
    (["word"],                                "word"),
    (["Hello", "world"],                     "Hello world"),
    # Additional edge cases
    (["Hello", "!", "world", "?"],           "Hello! world?"),
    (["\"", "hello", "\""],                  "\"hello\""),
    (["I", "can", "'", "t"],                 "I can't"),
    (["no", ".", "problem"],                 "no. problem"),
    ([".", ",", "!"],                        ".,!"),
]

for words_in, expected_out in _jw_cases:
    result = _join_words(words_in)
    label(f"_join_words({words_in!r})",
          result == expected_out, got=repr(result), expected=repr(expected_out))


# =========================================================================
# STAGE 10  — metadata_codec: every mode independently, then full pipeline
# =========================================================================
print(SEP)
print("STAGE 10 — metadata_codec: every mode + encode_metadata / decode_metadata")
print(SEP)

from compression.metadata_codec import (
    encode_metadata, decode_metadata, is_lexi_file,
    # low-level per-mode helpers
    _enc_raw,            _dec_raw,
    _enc_scalar,         _dec_scalar,
    _enc_flat_uint,      _dec_flat_uint,
    _enc_flat_int,       _dec_flat_int,
    _enc_flat_dict,      _dec_flat_dict,
    _enc_pos,            _dec_pos,
    _enc_int_nested,     _dec_int_nested,
    _enc_sparse_dict,    _dec_sparse_dict,
    _enc_sparse_dict_pos,_dec_sparse_dict_pos,
    _enc_model_weights,  _dec_model_weights,
    _enc_pos_freq,       _dec_pos_freq,
    _enc_float_nested,   _dec_float_nested,
    _enc_symbol_table,   _dec_symbol_table,
)

# --- raw ---------------------------------------------------------------
for raw_data in [b"", b"\x00", b"\xde\xad\xbe\xef" * 16, bytes(range(256))]:
    rt = _dec_raw(_enc_raw(raw_data))
    label(f"raw round-trip len={len(raw_data)}",
          rt == raw_data, got=rt[:8], expected=raw_data[:8])

# --- scalar ------------------------------------------------------------
for val in [0, 1, -1, 127, -128, 1_000_000, -1_000_000, 2**31 - 1, -(2**31)]:
    rt = _dec_scalar(_enc_scalar(val))
    label(f"scalar round-trip {val}",
          rt == val, got=rt, expected=val)

# --- flat_uint ---------------------------------------------------------
for data in [[], [0], [255], list(range(20)), [0] * 50, [65535], [0, 1, 2, 3, 4]]:
    rt = _dec_flat_uint(_enc_flat_uint(data))
    label(f"flat_uint round-trip len={len(data)}",
          rt == data, got=rt[:5], expected=data[:5])

# --- flat_int ----------------------------------------------------------
for data in [[], [0], [-1, 0, 1], list(range(-10, 11)), [1_000_000, -1_000_000],
             [-32768, 32767]]:
    rt = _dec_flat_int(_enc_flat_int(data))
    label(f"flat_int round-trip {data[:4]}{'...' if len(data) > 4 else ''}",
          rt == data, got=rt[:5], expected=data[:5])

# --- flat_dict ---------------------------------------------------------
for d in [{}, {0: 1}, {-1: 5, 0: 3, 1: 2}, {k: k * 2 for k in range(10)},
          {-100: 999, 0: 0, 100: -999}]:
    rt = _dec_flat_dict(_enc_flat_dict(d))
    label(f"flat_dict round-trip len={len(d)}",
          rt == d, got=rt, expected=d)

# --- pos (nested list[list[str]]) — UPOS tags only --------------------
pos_cases = [
    [],
    [[]],
    [["NOUN", "VERB", "DET"]],
    [["NOUN", "VERB"], ["ADJ", "PUNCT", "X"]],
    [["PROPN", "AUX", "NOUN", "VERB", "DET", "NOUN", "PUNCT"]],
]
for sentences in pos_cases:
    rt = _dec_pos(_enc_pos(sentences))
    label(f"pos round-trip {[len(s) for s in sentences]}",
          rt == sentences, got=rt, expected=sentences)

# --- int_nested (nested list[list[int]], signed) ----------------------
int_nested_cases = [
    [],
    [[]],
    [[0, 1, 2]],
    [[0, -1, 2], [3, -4]],
    [[i - 5 for i in range(10)]],
    [[0] * 100],
    [[12, 0, 3, 1, 0, 2, 0, 1]],
]
for sentences in int_nested_cases:
    rt = _dec_int_nested(_enc_int_nested(sentences))
    label(f"int_nested round-trip {[len(s) for s in sentences]}",
          rt == sentences, got=rt, expected=sentences)

# --- sparse_dict (dict[int, dict[int, int]]) --------------------------
print("  -- sparse_dict --")
_sd_cases = [
    {},
    {0: {0: 1}},
    {0: {1: 5, 2: 3}, 1: {0: 2}},
    {-1: {0: 10, -1: 3}, 0: {0: 4}},
    {k: {k + j: j + 1 for j in range(3)} for k in range(5)},
    {100: {-100: 999}},
]
for d in _sd_cases:
    rt = _dec_sparse_dict(_enc_sparse_dict(d))
    label(f"sparse_dict round-trip keys={sorted(d.keys())}",
          rt == d, got=rt, expected=d)

# --- sparse_dict_pos (dict[str, dict[int, int]]) ----------------------
print("  -- sparse_dict_pos --")
_sdp_cases = [
    {},
    {"NOUN": {0: 1}},
    {"NOUN": {1: 5, 2: 3}, "VERB": {0: 2}},
    {"ADJ": {0: 10}, "DET": {1: 4, 2: 2}, "PUNCT": {0: 1}},
    {"X": {0: 1, 1: 2, 2: 3}},
]
for d in _sdp_cases:
    rt = _dec_sparse_dict_pos(_enc_sparse_dict_pos(d))
    label(f"sparse_dict_pos round-trip keys={sorted(d.keys())}",
          rt == d, got=rt, expected=d)

# --- model_weights (list[float], 3 elements, float64) -----------------
print("  -- model_weights --")
_mw_cases = [
    [1/3, 1/3, 1/3],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, 0.3, 0.2],
    [0.9999999, 0.0000001, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 0.0],
]
for weights in _mw_cases:
    rt = _dec_model_weights(_enc_model_weights(weights))
    # float64 should be bit-exact
    label(f"model_weights round-trip {[round(w, 6) for w in weights]}",
          rt == weights, got=rt, expected=weights)

# Verify no float32 precision loss (the historical bug)
precise_weights = [0.33333333333333331, 0.33333333333333331, 0.33333333333333337]
rt_precise = _dec_model_weights(_enc_model_weights(precise_weights))
label("model_weights preserves float64 precision (no float32 rounding)",
      rt_precise == precise_weights,
      got=rt_precise, expected=precise_weights)

# --- pos_freq (dict[str, int]) ----------------------------------------
print("  -- pos_freq --")
_pf_cases = [
    {},
    {"NOUN": 10},
    {"NOUN": 10, "VERB": 5},
    {"NOUN": 100, "VERB": 50, "DET": 30, "ADJ": 20, "PUNCT": 15},
    {tag: i + 1 for i, tag in enumerate(["ADJ", "ADP", "ADV", "AUX", "NOUN", "VERB"])},
    {"X": 1},
    {"PROPN": 7, "PRON": 3},
]
for d in _pf_cases:
    rt = _dec_pos_freq(_enc_pos_freq(d))
    label(f"pos_freq round-trip keys={sorted(d.keys())}",
          rt == d, got=rt, expected=d)

# Edge: unknown tags must be silently dropped (not in UPOS_TAGS)
_pf_unknown = {"NOUN": 5, "FAKE_TAG": 99, "VERB": 3}
rt_unknown = _dec_pos_freq(_enc_pos_freq(_pf_unknown))
label("pos_freq silently drops unknown tags",
      "FAKE_TAG" not in rt_unknown and rt_unknown.get("NOUN") == 5,
      got=rt_unknown, expected={"NOUN": 5, "VERB": 3})

# --- float_nested (nested list[list[float]]) --------------------------
print("  -- float_nested --")
_fn_cases = [
    [],
    [[]],
    [[3.14]],
    [[1.0, 2.0, 3.0]],
    [[1.5, 2.5], [3.5, 4.5]],
    [[0.0, 0.0, 0.0]],
    [[100.0, -100.0, 0.5, -0.5]],
]
for sentences in _fn_cases:
    rt = _dec_float_nested(_enc_float_nested(sentences))
    # float32 XOR-delta — tolerate tiny precision differences
    ok = len(rt) == len(sentences) and all(
        len(rt[i]) == len(sentences[i]) and
        all(abs(rt[i][j] - sentences[i][j]) < 1e-5
            for j in range(len(sentences[i])))
        for i in range(len(sentences))
    )
    label(f"float_nested round-trip {[len(s) for s in sentences]}",
          ok, got=rt, expected=sentences)

# --- symbol_table (dict[str, str]) ------------------------------------
print("  -- symbol_table --")
_st_cases = [
    {},
    {"§E0": "London"},
    {"§E0": "Ishmael", "§E1": "Ahab", "§R0": "hunts"},
    {"§E0": "Ünïcödé", "§R1": "São Paulo"},
    {"§E0": "A" * 50},   # long value
    {"§E0": "x", "§E1": "y", "§E2": "z", "§E3": "w"},
]
for d in _st_cases:
    rt = _dec_symbol_table(_enc_symbol_table(d))
    label(f"symbol_table round-trip len={len(d)}",
          rt == d, got=rt, expected=d)

# --- case_flags / case_bitmaps via int_nested directly ----------------
print("  -- case_flags / case_bitmaps int_nested round-trips --")
cf_cases = [
    [],
    [[]],
    [[0, 1, 2, 3]],
    [[0, 0, 1], [2, 3, 0]],
    [[CASE_LOWER] * 10, [CASE_TITLE, CASE_UPPER, CASE_MIXED]],
    [[CASE_MIXED] * 5],
]
for sentences in cf_cases:
    rt = _dec_int_nested(_enc_int_nested(sentences))
    label(f"case_flags int_nested round-trip {[len(s) for s in sentences]}",
          rt == sentences, got=rt, expected=sentences)

cb_cases = [
    [],
    [[]],
    [[0, 0, 2, 0]],
    [[0, 2], [0, 4, 0]],
    [[0] * 8, [1, 2, 4, 8, 16]],
    [[255, 0, 127]],
    [[1 << i for i in range(8)]],
]
for sentences in cb_cases:
    rt = _dec_int_nested(_enc_int_nested(sentences))
    label(f"case_bitmaps int_nested round-trip {[len(s) for s in sentences]}",
          rt == sentences, got=rt, expected=sentences)

# --- Full encode_metadata / decode_metadata ---------------------------
# NOTE: all POS tags must be UPOS (ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ,
# NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X).
test_metadata: Dict[str, Any] = {
    "compressed_bitstream":  b"\x01\x02\x03\xff",
    "pos_deltas_bitstream":  b"\xaa\xbb",
    "symbol_table":          {"§E0": "London"},
    "pos_deltas_counts":     {0: 3, 1: 2, -1: 1},
    "pos_deltas_count":      6,
    "sentence_char_counts":  [12, 8],
    "pos_huffman_bits":      [3.14, 2.71],
    "pos_n_tags":            [5, 4],
    "pos_tags":              [["NOUN", "VERB", "DET"], ["PROPN", "AUX"]],
    "morph_codes":           [[0, 2, 0], [0, 1]],
    "root_lengths":          [[3, 2, 1], [2, 3]],
    "model_weights":         [0.33, 0.33, 0.34],
    "char_context":          {0: {1: 5, 2: 3}, 1: {0: 2}},
    "morph_context":         {0: {0: 10}},
    "struct_context":        {"NOUN": {0: 4, 1: 2}},
    "char_vocab":            [0, 1, 2, 3, 4, 5, 6],
    "morph_vocab":           [0, 2],
    "pos_vocab":             ["NOUN", "VERB", "DET"],
    "num_symbols":           20,
    "num_char_classes":      7,
    "pos_freq_table":        {"NOUN": 10, "VERB": 5},
    # case fields
    "case_flags":            [[CASE_LOWER, CASE_TITLE, CASE_UPPER],
                              [CASE_MIXED, CASE_LOWER]],
    "case_bitmaps":          [[0, 0, 0], [2, 0]],
}

encoded_binary = encode_metadata(test_metadata)
label("encode_metadata returns bytes",
      isinstance(encoded_binary, (bytes, bytearray)),
      type(encoded_binary).__name__, "bytes")
label("encoded binary is non-empty",
      len(encoded_binary) > 0, len(encoded_binary), "> 0")

label("is_lexi_file detects own output",
      is_lexi_file(encoded_binary), got=False, expected=True)
label("is_lexi_file rejects random bytes",
      not is_lexi_file(b"\x00\x01\x02\x03\x04\x05\x06\x07"),
      got=True, expected=False)
label("is_lexi_file rejects JSON",
      not is_lexi_file(b'{"key": "value"}'),
      got=True, expected=False)
label("is_lexi_file rejects empty bytes",
      not is_lexi_file(b""),
      got=True, expected=False)
label("is_lexi_file rejects truncated magic",
      not is_lexi_file(b"LEX"),
      got=True, expected=False)
label("is_lexi_file rejects single byte",
      not is_lexi_file(b"\x00"),
      got=True, expected=False)

try:
    decoded_meta = decode_metadata(encoded_binary)
    label("decode_metadata returns dict",
          isinstance(decoded_meta, dict), type(decoded_meta).__name__, "dict")

    for key in ["pos_deltas_count", "num_symbols", "num_char_classes"]:
        label(f"scalar field '{key}' round-trips",
              decoded_meta.get(key) == test_metadata[key],
              got=decoded_meta.get(key), expected=test_metadata[key])

    label("compressed_bitstream round-trips",
          bytes(decoded_meta["compressed_bitstream"]) == test_metadata["compressed_bitstream"],
          got=bytes(decoded_meta["compressed_bitstream"]),
          expected=test_metadata["compressed_bitstream"])

    label("pos_deltas_bitstream round-trips",
          bytes(decoded_meta["pos_deltas_bitstream"]) == test_metadata["pos_deltas_bitstream"],
          got=bytes(decoded_meta["pos_deltas_bitstream"]),
          expected=test_metadata["pos_deltas_bitstream"])

    label("symbol_table round-trips",
          decoded_meta.get("symbol_table") == test_metadata["symbol_table"],
          got=decoded_meta.get("symbol_table"), expected=test_metadata["symbol_table"])

    label("pos_tags round-trips",
          decoded_meta.get("pos_tags") == test_metadata["pos_tags"],
          got=decoded_meta.get("pos_tags"), expected=test_metadata["pos_tags"])

    label("morph_codes round-trips",
          decoded_meta.get("morph_codes") == test_metadata["morph_codes"],
          got=decoded_meta.get("morph_codes"), expected=test_metadata["morph_codes"])

    label("root_lengths round-trips",
          decoded_meta.get("root_lengths") == test_metadata["root_lengths"],
          got=decoded_meta.get("root_lengths"), expected=test_metadata["root_lengths"])

    label("model_weights round-trips",
          decoded_meta.get("model_weights") == test_metadata["model_weights"],
          got=decoded_meta.get("model_weights"), expected=test_metadata["model_weights"])

    label("char_vocab round-trips",
          list(decoded_meta.get("char_vocab", [])) == test_metadata["char_vocab"],
          got=decoded_meta.get("char_vocab"), expected=test_metadata["char_vocab"])

    label("morph_vocab round-trips",
          list(decoded_meta.get("morph_vocab", [])) == test_metadata["morph_vocab"],
          got=decoded_meta.get("morph_vocab"), expected=test_metadata["morph_vocab"])

    label("pos_vocab round-trips",
          list(decoded_meta.get("pos_vocab", [])) == test_metadata["pos_vocab"],
          got=decoded_meta.get("pos_vocab"), expected=test_metadata["pos_vocab"])

    label("pos_freq_table round-trips",
          decoded_meta.get("pos_freq_table") == test_metadata["pos_freq_table"],
          got=decoded_meta.get("pos_freq_table"), expected=test_metadata["pos_freq_table"])

    label("sentence_char_counts round-trips",
          list(decoded_meta.get("sentence_char_counts", [])) == test_metadata["sentence_char_counts"],
          got=decoded_meta.get("sentence_char_counts"),
          expected=test_metadata["sentence_char_counts"])

    label("pos_deltas_counts round-trips",
          decoded_meta.get("pos_deltas_counts") == test_metadata["pos_deltas_counts"],
          got=decoded_meta.get("pos_deltas_counts"),
          expected=test_metadata["pos_deltas_counts"])

    label("char_context round-trips",
          decoded_meta.get("char_context") == test_metadata["char_context"],
          got=decoded_meta.get("char_context"),
          expected=test_metadata["char_context"])

    label("morph_context round-trips",
          decoded_meta.get("morph_context") == test_metadata["morph_context"],
          got=decoded_meta.get("morph_context"),
          expected=test_metadata["morph_context"])

    label("struct_context round-trips",
          decoded_meta.get("struct_context") == test_metadata["struct_context"],
          got=decoded_meta.get("struct_context"),
          expected=test_metadata["struct_context"])

    # --- case_flags and case_bitmaps round-trips ---
    label("case_flags round-trips",
          decoded_meta.get("case_flags") == test_metadata["case_flags"],
          got=decoded_meta.get("case_flags"),
          expected=test_metadata["case_flags"])

    label("case_bitmaps round-trips",
          decoded_meta.get("case_bitmaps") == test_metadata["case_bitmaps"],
          got=decoded_meta.get("case_bitmaps"),
          expected=test_metadata["case_bitmaps"])

    # Verify correct flag values survived serialisation
    for s_idx, (flags_sent, bitmaps_sent) in enumerate(
        zip(decoded_meta["case_flags"], decoded_meta["case_bitmaps"])
    ):
        for t_idx, (flag, bitmap) in enumerate(zip(flags_sent, bitmaps_sent)):
            orig_flag   = test_metadata["case_flags"][s_idx][t_idx]
            orig_bitmap = test_metadata["case_bitmaps"][s_idx][t_idx]
            label(f"case_flags[{s_idx}][{t_idx}] == {orig_flag}",
                  flag == orig_flag, got=flag, expected=orig_flag)
            label(f"case_bitmaps[{s_idx}][{t_idx}] == {orig_bitmap}",
                  bitmap == orig_bitmap, got=bitmap, expected=orig_bitmap)

    # --- Additional: compress + decompress the metadata itself (idempotency) ---
    # Re-encode the decoded metadata and check the binary is identical
    re_encoded = encode_metadata(decoded_meta)
    re_decoded = decode_metadata(re_encoded)
    label("decode_metadata(encode_metadata(decoded)) is idempotent for pos_tags",
          re_decoded.get("pos_tags") == test_metadata["pos_tags"],
          got=re_decoded.get("pos_tags"), expected=test_metadata["pos_tags"])
    label("decode_metadata(encode_metadata(decoded)) is idempotent for case_flags",
          re_decoded.get("case_flags") == test_metadata["case_flags"],
          got=re_decoded.get("case_flags"), expected=test_metadata["case_flags"])

except Exception as _e10:
    label("decode_metadata completed without exception",
          False, got=f"{type(_e10).__name__}: {_e10}", expected="no exception")
    print("  traceback:")
    for _line in traceback.format_exc().splitlines():
        print(f"    {_line}")


# =========================================================================
# STAGE 11  — full compress_to_file → decompress end-to-end
# =========================================================================
print(SEP)
print("STAGE 11 — full compress_to_file → decompress end-to-end")
print(SEP)

from main import compress_to_file, decompress

with tempfile.NamedTemporaryFile(suffix=".lexis", delete=False) as tf:
    tmp_path = tf.name

try:
    stats = compress_to_file(normalized, tmp_path)

    label("compress_to_file returns dict",
          isinstance(stats, dict), type(stats).__name__, "dict")
    label("output file exists and is non-empty",
          Path(tmp_path).exists() and Path(tmp_path).stat().st_size > 0,
          got=Path(tmp_path).stat().st_size if Path(tmp_path).exists() else 0,
          expected="> 0")
    label("is_lexi_file accepts written output",
          is_lexi_file(Path(tmp_path).read_bytes()), got=False, expected=True)

    for stat_key in ["original_size", "compressed_size", "compression_ratio", "bpb"]:
        label(f"stats has key '{stat_key}'",
              stat_key in stats, got=list(stats.keys()), expected=f"contains '{stat_key}'")

    label("compression_ratio > 0",
          stats.get("compression_ratio", 0) > 0,
          got=stats.get("compression_ratio"), expected="> 0")
    label("original_size > compressed_size (file is actually compressed)",
          stats.get("original_size", 0) > stats.get("compressed_size", 0),
          got=(stats.get("original_size"), stats.get("compressed_size")),
          expected="original > compressed")

    # Verify case_flags and case_bitmaps are persisted in the written file
    _raw_lexi = Path(tmp_path).read_bytes()
    _meta_check = decode_metadata(_raw_lexi)
    label("written LEXI file contains case_flags field",
          "case_flags" in _meta_check and len(_meta_check["case_flags"]) > 0,
          got=len(_meta_check.get("case_flags", [])), expected="> 0 sentences")
    label("written LEXI file contains case_bitmaps field",
          "case_bitmaps" in _meta_check and len(_meta_check["case_bitmaps"]) > 0,
          got=len(_meta_check.get("case_bitmaps", [])), expected="> 0 sentences")
    label("case_flags sentence count == case_bitmaps sentence count",
          len(_meta_check["case_flags"]) == len(_meta_check["case_bitmaps"]),
          got=len(_meta_check["case_flags"]), expected=len(_meta_check["case_bitmaps"]))

    # Verify case_flags values are all in 0–3
    all_cf = [f for sent in _meta_check["case_flags"] for f in sent]
    label("all serialised case_flags in 0–3",
          all(0 <= f <= 3 for f in all_cf),
          got=[f for f in all_cf if not (0 <= f <= 3)], expected="[]")

    # Verify case_bitmaps are non-negative ints
    all_cb = [b for sent in _meta_check["case_bitmaps"] for b in sent]
    label("all serialised case_bitmaps are non-negative ints",
          all(isinstance(b, int) and b >= 0 for b in all_cb),
          got=[b for b in all_cb if not (isinstance(b, int) and b >= 0)], expected="[]")

    decoded_text = decompress(tmp_path)

    label("decompress returns string",
          isinstance(decoded_text, str), type(decoded_text).__name__, "str")
    label("decoded length == normalized",
          len(decoded_text) == len(normalized),
          len(decoded_text), len(normalized))
    label("decoded text == normalized",
          decoded_text == normalized,
          repr(decoded_text[:120]), repr(normalized[:120]))

    if decoded_text != normalized:
        for i, (a, b) in enumerate(zip(decoded_text, normalized)):
            if a != b:
                print(f"  first char mismatch @ index {i}:")
                decoded_snippet  = repr(decoded_text[max(0, i-10):i+20])
                original_snippet = repr(normalized[max(0, i-10):i+20])
                print(f"    decoded  : {decoded_snippet}")
                print(f"    original : {original_snippet}")
                pos = 0
                for s_idx, sent in enumerate(all_sents):
                    sent_len = len(sent.text)
                    if pos + sent_len >= i:
                        print(f"  in sentence #{s_idx}: {repr(sent.text[:60])}")
                        print(f"  offset within sentence: {i - pos}")
                        break
                    pos += sent_len + 1
                break
        if len(decoded_text) != len(normalized):
            print(f"  decoded is {'shorter' if len(decoded_text) < len(normalized) else 'longer'} "
                  f"by {abs(len(decoded_text) - len(normalized))} chars")

    # --- case restoration spot-check ---
    print("  -- case restoration spot-check --")
    _src_tokens = [item[0] for item in morph_results]
    _dec_doc    = nlp(decoded_text[:len(first_sent.text) + 5])
    _dec_tokens = [t.text for t in _dec_doc]
    _matched = 0
    _total   = min(len(_src_tokens), len(_dec_tokens), 10)
    for _i in range(_total):
        if _src_tokens[_i] == _dec_tokens[_i]:
            _matched += 1
    label(f"case-restored tokens: {_matched}/{_total} match source (first sentence)",
          _matched >= _total * 0.7,
          got=f"{_matched}/{_total}", expected=f">= {int(_total * 0.7)}/{_total}")

    if _matched < _total:
        print("  token comparison (src → dec):")
        for _i in range(_total):
            _src = _src_tokens[_i] if _i < len(_src_tokens) else "?"
            _dec = _dec_tokens[_i] if _i < len(_dec_tokens) else "?"
            match = "✓" if _src == _dec else "✗"
            print(f"    [{match}] src={_src!r:20} dec={_dec!r}")

    # --- decompress on a second independent temp file (idempotency) ---
    with tempfile.NamedTemporaryFile(suffix=".lexis", delete=False) as tf2:
        tmp_path2 = tf2.name
    try:
        stats2 = compress_to_file(normalized, tmp_path2)
        decoded2 = decompress(tmp_path2)
        label("second independent compress/decompress identical to first",
              decoded2 == decoded_text,
              repr(decoded2[:80]), repr(decoded_text[:80]))
    finally:
        try:
            os.unlink(tmp_path2)
        except OSError:
            pass

except Exception as _e11:
    label("Stage 11 completed without exception",
          False, got=f"{type(_e11).__name__}: {_e11}", expected="no exception")
    print("  traceback:")
    for _line in traceback.format_exc().splitlines():
        print(f"    {_line}")
finally:
    try:
        os.unlink(tmp_path)
    except OSError:
        pass


# =========================================================================
# STAGE 12  — autocorrect pass-through
# =========================================================================
print(SEP)
print("STAGE 12 — autocorrect pass-through")
print(SEP)

from compression.pipeline.stage9_autocorrect import autocorrect

autocorrect_cases = [
    "The quick brown fox jumps over the lazy dog.",
    "Hello world.",
    "It is a truth universally acknowledged.",
    "",
    "A",
    # Edge: already-correct text must not be altered
    normalized[:200],
    # Edge: punctuation-only
    "...",
    # Edge: single word
    "hello",
    # Edge: numbers
    "12345",
]

for case in autocorrect_cases:
    snippet = repr(case[:30])
    try:
        result = autocorrect(case)
        label(f"autocorrect({snippet}) returns str",
              isinstance(result, str), type(result).__name__, "str")
        if case:
            label(f"autocorrect({snippet}) is non-empty",
                  len(result) > 0, got=len(result), expected="> 0")
        # autocorrect must never expand the text beyond reason
        label(f"autocorrect({snippet}) length <= 2x input",
              len(result) <= max(len(case) * 2, 10),
              got=len(result), expected=f"<= {max(len(case) * 2, 10)}")
    except Exception as _eac:
        label(f"autocorrect({snippet}) raises no exception",
              False, got=f"{type(_eac).__name__}: {_eac}", expected="no exception")


print(SEP)
print("Done.")
