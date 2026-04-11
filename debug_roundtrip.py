"""
Drop-in diagnostic: audits context model round-trip through the LEXI codec.

Usage:
    python debug_roundtrip.py             # uses moby500.lexis + moby500.txt
    python debug_roundtrip.py foo.lexis   # uses foo.lexis + foo.txt
"""
from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from compression.metadata_codec import decode_metadata, is_lexi_file
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.stage7_arithmetic import ArithmeticDecoder, _build_context_stream


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_lexi_model(p: Dict[str, Any]) -> ContextMixingModel:
    """Reconstruct ContextMixingModel from decoded LEXI metadata."""
    m = ContextMixingModel()
    m.char_context  = defaultdict(Counter, {
        int(k): Counter({int(ck): int(cv) for ck, cv in v.items()})
        for k, v in p.get("char_context", {}).items()
    })
    m.morph_context = defaultdict(Counter, {
        int(k): Counter({int(ck): int(cv) for ck, cv in v.items()})
        for k, v in p.get("morph_context", {}).items()
    })
    m.struct_context = defaultdict(Counter, {
        str(k): Counter({int(ck): int(cv) for ck, cv in v.items()})
        for k, v in p.get("struct_context", {}).items()
    })
    m.weights    = list(p.get("model_weights", [1/3, 1/3, 1/3]))
    m.char_vocab  = [int(v) for v in p.get("char_vocab",  list(range(7)))]
    m.morph_vocab = [int(v) for v in p.get("morph_vocab", [0])]
    m.pos_vocab   = [str(v) for v in p.get("pos_vocab",   [])]
    return m


def _build_live_model(text: str):
    """Build context model by running the live pipeline (no LEXI round-trip)."""
    from main import _encode_for_model
    encoded_sentences, _ = _encode_for_model(text)
    model = ContextMixingModel()
    model.train(encoded_sentences)
    return model, encoded_sentences


def _rebuild_encoded_sentences(p: Dict[str, Any]) -> List[Dict]:
    root_lengths  = p["root_lengths"]
    pos_bits      = p["pos_huffman_bits"]
    pos_n_tags    = p["pos_n_tags"]
    pos_tags_meta = p["pos_tags"]
    morph_codes   = p["morph_codes"]
    encoded = []
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
        encoded.append({
            "char_morph_codes": char_morph_codes,
            "char_pos_tags":    char_pos_tags,
            "pos_huffman_bits": float(pos_bits[idx]) if idx < len(pos_bits) else 0.0,
            "pos_n_tags":       int(pos_n_tags[idx]) if idx < len(pos_n_tags) else 0,
            "pos_tags":         sentence_pos,
        })
    return encoded


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run(path: str = "moby500.lexis") -> None:
    raw = Path(path).read_bytes()
    assert is_lexi_file(raw), f"Not a LEXI file: {path}"
    p = decode_metadata(raw)

    txt_path = path.replace(".lexis", ".txt")
    text = Path(txt_path).read_text(encoding="utf-8")

    # --- build both models ---
    print("Building live model (re-running pipeline on original text)...")
    live_model, _ = _build_live_model(text)
    lexi_model    = _build_lexi_model(p)
    encoded       = _rebuild_encoded_sentences(p)

    # === Audit 1: char_context ===
    print("\n=== char_context ===")
    mismatches = 0
    for cls in sorted(set(live_model.char_context) | set(lexi_model.char_context)):
        live_c = dict(live_model.char_context.get(cls, Counter()))
        lexi_c = dict(lexi_model.char_context.get(cls, Counter()))
        if live_c != lexi_c:
            mismatches += 1
            print(f"  MISMATCH class {cls}:")
            print(f"    live={live_c}")
            print(f"    lexi={lexi_c}")
    if mismatches == 0:
        print(f"  All {len(live_model.char_context)} entries OK")

    # === Audit 2: morph_context ===
    print("\n=== morph_context ===")
    mismatches = 0
    for k in sorted(set(live_model.morph_context) | set(lexi_model.morph_context)):
        live_c = dict(live_model.morph_context.get(k, Counter()))
        lexi_c = dict(lexi_model.morph_context.get(k, Counter()))
        if live_c != lexi_c:
            mismatches += 1
            print(f"  MISMATCH morph {k}: live={live_c}  lexi={lexi_c}")
    if mismatches == 0:
        print(f"  All {len(live_model.morph_context)} entries OK")

    # === Audit 3: weights ===
    print("\n=== model_weights ===")
    print(f"  live: {[round(w,6) for w in live_model.weights]}")
    print(f"  lexi: {[round(w,6) for w in lexi_model.weights]}")
    weight_ok = all(abs(a-b) < 0.001 for a, b in zip(live_model.weights, lexi_model.weights))
    print(f"  Match within 0.001: {weight_ok}")

    # === Audit 4: vocabs ===
    print("\n=== vocabs ===")
    print(f"  char_vocab   live={live_model.char_vocab}  lexi={lexi_model.char_vocab}  match={live_model.char_vocab==lexi_model.char_vocab}")
    print(f"  morph_vocab  live={live_model.morph_vocab}  lexi={lexi_model.morph_vocab}  match={live_model.morph_vocab==lexi_model.morph_vocab}")

    # === Audit 5: decode with live vs lexi model ===
    print("\n=== arithmetic decode: live model ===")
    cc_live = ArithmeticDecoder().decode(
        bytes(p["compressed_bitstream"]), live_model, encoded, p["num_symbols"]
    )
    print(f"  char_classes[:20]: {cc_live[:20]}")

    print("\n=== arithmetic decode: lexi model ===")
    cc_lexi = ArithmeticDecoder().decode(
        bytes(p["compressed_bitstream"]), lexi_model, encoded, p["num_symbols"]
    )
    print(f"  char_classes[:20]: {cc_lexi[:20]}")

    if cc_live == cc_lexi:
        print("\n  Models produce IDENTICAL decode -> problem is downstream of arithmetic coder")
    else:
        first_diff = next(i for i, (a, b) in enumerate(zip(cc_live, cc_lexi)) if a != b)
        print(f"\n  FIRST DIVERGENCE at symbol index {first_diff}")
        print(f"    live={cc_live[first_diff]}  lexi={cc_lexi[first_diff]}")
        print(f"  live stream (around divergence): {cc_live[max(0,first_diff-2):first_diff+5]}")
        print(f"  lexi stream (around divergence): {cc_lexi[max(0,first_diff-2):first_diff+5]}")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "moby500.lexis"
    run(path)
