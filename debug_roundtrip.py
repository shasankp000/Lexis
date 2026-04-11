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

from compression.metadata_codec import decode_metadata, encode_metadata, is_lexi_file
from compression.pipeline.stage7_arithmetic import ArithmeticDecoder, _build_context_stream
from compression.alphabet.phonetic_map import PHONETIC_CLASSES


def _cumulative_from_deltas(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    values = [deltas[0]]
    for d in deltas[1:]:
        values.append(values[-1] + d)
    return values


def _build_context_model_from_meta(p):
    """Reconstruct ContextMixingModel from decoded metadata."""
    from compression.pipeline.stage6_probability import ContextMixingModel
    m = ContextMixingModel()
    m.char_context  = defaultdict(Counter, {int(k): Counter({int(ck): int(cv) for ck, cv in v.items()}) for k, v in p.get("char_context", {}).items()})
    m.morph_context = defaultdict(Counter, {int(k): Counter({int(ck): int(cv) for ck, cv in v.items()}) for k, v in p.get("morph_context", {}).items()})
    m.struct_context= defaultdict(Counter, {str(k): Counter({int(ck): int(cv) for ck, cv in v.items()}) for k, v in p.get("struct_context", {}).items()})
    m.weights    = list(p.get("model_weights", [1/3, 1/3, 1/3]))
    m.char_vocab  = list(p.get("char_vocab", list(range(7))))
    m.morph_vocab = list(p.get("morph_vocab", [0]))
    m.pos_vocab   = list(p.get("pos_vocab", []))
    return m


def _live_context_model(text: str):
    """Build context model directly from the live pipeline (no LEXI round-trip)."""
    from main import _run_pipeline
    encoded_sentences, context_model, _ = _run_pipeline(text)
    return context_model, encoded_sentences


def run(path: str = "moby500.lexis") -> None:
    raw = Path(path).read_bytes()
    assert is_lexi_file(raw), "Not a LEXI file"
    p = decode_metadata(raw)

    # --- Rebuild live context model from original text for comparison ---
    text = open(path.replace(".lexis", ".txt")).read()
    live_model, live_sentences = _live_context_model(text)

    # --- Rebuild context model from LEXI metadata ---
    lexi_model = _build_context_model_from_meta(p)

    # === Audit 1: char_context ===
    print("=== char_context audit ===")
    for cls in range(7):
        live_c  = dict(live_model.char_context.get(cls, Counter()))
        lexi_c  = dict(lexi_model.char_context.get(cls, Counter()))
        if live_c != lexi_c:
            print(f"  MISMATCH class {cls}:")
            print(f"    live: {live_c}")
            print(f"    lexi: {lexi_c}")
        else:
            print(f"  class {cls}: OK ({len(live_c)} entries)")

    # === Audit 2: morph_context ===
    print("\n=== morph_context audit ===")
    all_morph_keys = set(live_model.morph_context) | set(lexi_model.morph_context)
    mismatch_morph = 0
    for k in sorted(all_morph_keys):
        live_c = dict(live_model.morph_context.get(k, Counter()))
        lexi_c = dict(lexi_model.morph_context.get(k, Counter()))
        if live_c != lexi_c:
            mismatch_morph += 1
            print(f"  MISMATCH morph {k}: live={live_c}  lexi={lexi_c}")
    if mismatch_morph == 0:
        print(f"  All {len(all_morph_keys)} morph_context entries OK")

    # === Audit 3: weights ===
    print("\n=== model_weights ===")
    print(f"  live: {live_model.weights}")
    print(f"  lexi: {lexi_model.weights}")

    # === Audit 4: vocabs ===
    print("\n=== vocabs ===")
    print(f"  char_vocab  live={live_model.char_vocab}  lexi={lexi_model.char_vocab}")
    print(f"  morph_vocab live={live_model.morph_vocab}  lexi={lexi_model.morph_vocab}")

    # === Audit 5: first-symbol probability comparison ===
    print("\n=== first symbol probability (context=empty) ===")
    context = {"char_history": [], "current_morph_code": 0, "current_pos_tag": "X", "struct_prob": 1.0}
    live_dist = live_model.probability_distribution(context)
    lexi_dist = lexi_model.probability_distribution(context)
    print("  live:", {k: round(v, 4) for k, v in sorted(live_dist.items())})
    print("  lexi:", {k: round(v, 4) for k, v in sorted(lexi_dist.items())})

    # === Audit 6: full decode with live vs lexi model ===
    from compression.pipeline.stage7_arithmetic import ArithmeticDecoder
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
        encoded.append({
            "char_morph_codes": char_morph_codes,
            "char_pos_tags":    char_pos_tags,
            "pos_huffman_bits": float(pos_bits[idx]) if idx < len(pos_bits) else 0.0,
            "pos_n_tags":       int(pos_n_tags[idx]) if idx < len(pos_n_tags) else 0,
            "pos_tags":         sentence_pos,
        })

    print("\n=== decode with LIVE model (should be correct) ===")
    cc_live = ArithmeticDecoder().decode(
        bytes(p["compressed_bitstream"]), live_model, encoded, p["num_symbols"]
    )
    print(f"  char_classes[:20]: {cc_live[:20]}")

    print("\n=== decode with LEXI model (current behaviour) ===")
    cc_lexi = ArithmeticDecoder().decode(
        bytes(p["compressed_bitstream"]), lexi_model, encoded, p["num_symbols"]
    )
    print(f"  char_classes[:20]: {cc_lexi[:20]}")

    if cc_live != cc_lexi:
        first_diff = next(i for i, (a, b) in enumerate(zip(cc_live, cc_lexi)) if a != b)
        print(f"  FIRST DIVERGENCE at symbol index {first_diff}")
        print(f"    live={cc_live[first_diff]}  lexi={cc_lexi[first_diff]}")
    else:
        print("  Models produce identical decode — problem is downstream of arithmetic coder")


if __name__ == "__main__":
    run()
