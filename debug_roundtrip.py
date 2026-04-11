"""
Drop-in diagnostic: run after compress_to_file to verify every layer
of the decode pipeline independently.

Usage:
    python debug_roundtrip.py
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from compression.metadata_codec import decode_metadata, is_lexi_file
from compression.pipeline.stage7_arithmetic import ArithmeticDecoder, _build_context_stream
from compression.alphabet.phonetic_map import PHONETIC_CLASSES
from compression.alphabet.morph_codes import apply_morph


def _cumulative_from_deltas(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    values = [deltas[0]]
    for d in deltas[1:]:
        values.append(values[-1] + d)
    return values


def _flatten(nested):
    return [x for sub in nested for x in sub]


def run(path: str = "test.lexis") -> None:
    raw = Path(path).read_bytes()
    assert is_lexi_file(raw), "Not a LEXI file"
    p = decode_metadata(raw)

    # ── 1. Stream length audit ─────────────────────────────────────────────
    print("=== Stream length audit ===")
    print(f"  num_symbols            : {p['num_symbols']}")
    print(f"  sum(sentence_char_counts): {sum(p['sentence_char_counts'])}")
    print(f"  len(sentence_char_counts): {len(p['sentence_char_counts'])}")
    print(f"  len(pos_tags)            : {len(p['pos_tags'])}")
    print(f"  len(morph_codes)         : {len(p['morph_codes'])}")
    print(f"  len(root_lengths)        : {len(p['root_lengths'])}")
    for i, rl in enumerate(p['root_lengths']):
        n_chars = sum(rl) + 2 * len(rl) + max(len(rl) - 1, 0)
        print(f"  sentence {i}: root_lengths={rl}  -> expected char_count={n_chars}  stored={p['sentence_char_counts'][i]}")

    # ── 2. Rebuild encoded_sentences (same logic as main.py) ──────────────
    print("\n=== Reconstructing encoded_sentences ===")
    root_lengths  = p["root_lengths"]
    pos_bits      = p["pos_huffman_bits"]
    pos_n_tags    = p["pos_n_tags"]
    pos_tags_meta = p["pos_tags"]
    morph_codes   = p["morph_codes"]

    encoded: List[Dict[str, Any]] = []
    for idx, lengths in enumerate(root_lengths):
        sentence_pos   = pos_tags_meta[idx] if idx < len(pos_tags_meta) else []
        sentence_morph = morph_codes[idx]   if idx < len(morph_codes)   else []
        char_pos_tags:    List[str] = []
        char_morph_codes: List[int] = []

        for token_idx, length in enumerate(lengths):
            pos_tag    = sentence_pos[token_idx]   if token_idx < len(sentence_pos)   else "X"
            morph_code = sentence_morph[token_idx] if token_idx < len(sentence_morph) else 0
            char_pos_tags.append("X");      char_morph_codes.append(0)
            char_pos_tags.extend([pos_tag] * length)
            char_morph_codes.extend([morph_code] * length)
            char_pos_tags.append("X");      char_morph_codes.append(0)
            if token_idx < len(lengths) - 1:
                char_pos_tags.append("X");  char_morph_codes.append(0)

        enc = {
            "char_morph_codes": char_morph_codes,
            "char_pos_tags":    char_pos_tags,
            "pos_huffman_bits": float(pos_bits[idx]) if idx < len(pos_bits) else 0.0,
            "pos_n_tags":       int(pos_n_tags[idx]) if idx < len(pos_n_tags) else 0,
            "pos_tags":         sentence_pos,
        }
        encoded.append(enc)
        print(f"  sentence {idx}: len(char_morph_codes)={len(char_morph_codes)}  len(char_pos_tags)={len(char_pos_tags)}  stored_char_count={p['sentence_char_counts'][idx]}")

    # ── 3. Context stream vs num_symbols ──────────────────────────────────
    morph_stream, pos_stream, _ = _build_context_stream(encoded)
    print(f"\n=== Context stream ===")
    print(f"  len(morph_stream) : {len(morph_stream)}")
    print(f"  len(pos_stream)   : {len(pos_stream)}")
    print(f"  num_symbols       : {p['num_symbols']}")
    decode_length = min(p['num_symbols'], len(morph_stream), len(pos_stream))
    print(f"  decode will run   : {decode_length} steps")
    if decode_length < p['num_symbols']:
        print(f"  *** WARNING: decode will stop {p['num_symbols'] - decode_length} symbols early! ***")

    # ── 4. Build context model and decode ─────────────────────────────────
    from main import _build_context_model, _reconstruct_chars, _split_roots, _flatten as fl, _join_words
    from compression.pipeline.stage9_autocorrect import autocorrect

    context_model = _build_context_model(p)
    decoder = ArithmeticDecoder()
    char_classes = decoder.decode(
        p["compressed_bitstream"], context_model, encoded, p["num_symbols"]
    )
    print(f"\n=== Decoded char_classes (first 20): {char_classes[:20]}")

    # ── 5. pos_deltas decode ──────────────────────────────────────────────
    counts = {int(k): int(v) for k, v in p["pos_deltas_counts"].items()}
    pos_deltas = ArithmeticDecoder().decode_unigram_counts(
        bytes(p["pos_deltas_bitstream"]), counts, p["pos_deltas_count"]
    )
    print(f"=== pos_deltas (first 20): {pos_deltas[:20]}")

    sentence_counts = [int(c) for c in p["sentence_char_counts"]]
    char_stream = _reconstruct_chars(char_classes, pos_deltas, sentence_counts)
    print(f"=== char_stream: {repr(char_stream)}")

    roots = _split_roots(char_stream)
    print(f"=== roots: {roots}")

    morph_codes_flat = fl(p["morph_codes"])
    words = [
        apply_morph(root, morph_codes_flat[i] if i < len(morph_codes_flat) else 0)
        for i, root in enumerate(roots)
    ]
    print(f"=== words: {words}")
    result = _join_words(words)
    result = result[0].upper() + result[1:] if result else result
    print(f"=== final: {repr(autocorrect(result))}")


if __name__ == "__main__":
    run()
