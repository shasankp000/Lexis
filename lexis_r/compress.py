"""Lexis-R compressor.

Writes a self-contained .lexisr msgpack file, post-compressed with zstd
level 19.  The trained ContextMixingModel is serialised directly into
the payload (zlib-compressed) so the decompressor can load it without
re-training.

Usage
-----
    from lexis_r.compress import compress
    stats = compress(text, "output.lexisr")
"""

from __future__ import annotations

import os
import tempfile
import zlib
from pathlib import Path
from typing import Any, cast, Dict, List, Tuple

import msgpack

from compression.alphabet.phonetic_map import PHONETIC_CLASSES
from compression.alphabet.symbol_alphabet import SymbolAlphabet
from compression.pipeline.stage1_normalize import normalize_text
from compression.pipeline.stage1b_word_subs import encode_word_subs, merge_symbol_tables
from compression.pipeline.stage1c_symbol_slots import extract_symbols, pack_slot_map
from compression.pipeline.stage2_morphology import MorphologicalAnalyser
from compression.pipeline.stage3_syntax import analyse_sentence
from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import encode_symbols
from compression.pipeline.stage5_encode import CharacterEncoder, StructuralEncoder
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.utils import chunk_text
from lexis_r.arithmetic import ArithmeticEncoder
from lexis_r import huffman
from lexis_r.lz77_pos import pack_pos_tags_lz77
from lexis_r.payload import (
    POS_TO_IDX,
    pack_huffman_bits,
    pack_pos_freq_table,
    pack_token_array,
    pack_u8_list,
    pack_vlq_list,
)
from lexis_r.zstd_wrap import compress_payload

try:
    from tqdm import tqdm  # type: ignore
except ImportError:
    def tqdm(it, **kw):  # type: ignore
        return it


# ---------------------------------------------------------------------------
# spaCy loader
# ---------------------------------------------------------------------------

def _get_nlp(model: str | None = None):
    from compression.config import SPACY_MAX_LENGTH, SPACY_MODEL
    model_name = model or SPACY_MODEL
    try:
        import spacy
        gpu_fn = getattr(spacy, "prefer_gpu", None) or getattr(spacy, "require_gpu", None)
        if callable(gpu_fn):
            print("[spaCy] GPU enabled" if gpu_fn() else "[spaCy] Running on CPU")
        else:
            print("[spaCy] Running on CPU")
        nlp = spacy.load(model_name)
        nlp.max_length = SPACY_MAX_LENGTH
        print(f"[spaCy] Loaded model: {model_name}")
        return nlp
    except Exception as exc:
        raise RuntimeError(f"spaCy model '{model_name}' not available: {exc}") from exc


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------

def _encode_sentences(
    text: str, model: str | None = None
) -> tuple[list[dict], dict[str, int]]:
    nlp            = _get_nlp(model)
    analyser       = MorphologicalAnalyser(use_spacy=True, model_name=model)
    sym_alphabet   = SymbolAlphabet()
    struct_encoder = StructuralEncoder(sym_alphabet)
    char_encoder   = CharacterEncoder()
    chunks         = list(chunk_text(text))

    sentence_data: list[tuple] = []
    pos_sentences: list[list[str]] = []

    for i, chunk in enumerate(chunks):
        doc   = nlp(chunk)
        sents = list(doc.sents)
        for sent in tqdm(sents, desc=f"Chunk {i+1}/{len(chunks)}", unit="sent"):
            syntax     = analyse_sentence(sent)
            morphology = analyser.analyse_sentence(sent.text)
            sentence_data.append((morphology, syntax))
            pos_sentences.append(syntax.pos_tags)

    freq_table = struct_encoder.build_pos_frequency_table(pos_sentences)

    encoded: list[dict] = []
    for morphology, syntax in tqdm(
        sentence_data, desc="Stage 5 — encoding", unit="sent"
    ):
        encoded.append(
            char_encoder.encode_sentence_full(
                morphology, syntax, struct_encoder, freq_table
            )
        )

    return encoded, freq_table


def _serialise_model(model: ContextMixingModel) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".lcm", delete=False) as tf:
        tmp_path = tf.name
    try:
        model.serialise(tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Sentinel stripping  (per-sentence)
# ---------------------------------------------------------------------------

def _sentinel_layout(lengths: List[int]) -> List[bool]:
    layout: List[bool] = []
    for t_idx, length in enumerate(lengths):
        layout.append(True)
        layout.extend([False] * length)
        layout.append(True)
        if t_idx < len(lengths) - 1:
            layout.append(False)
    return layout


def _strip_sentinel_deltas_per_sentence(
    pos_deltas_nested:   List[List[int]],
    root_lengths_nested: List[List[int]],
) -> Tuple[List[List[int]], List[int]]:
    content_nested: List[List[int]] = []
    for s_idx, sent_deltas in enumerate(pos_deltas_nested):
        lengths = root_lengths_nested[s_idx] if s_idx < len(root_lengths_nested) else []
        layout  = _sentinel_layout(lengths)
        content = [d for d, is_sent in zip(sent_deltas, layout) if not is_sent]
        content_nested.append(content)
    content_counts = [len(c) for c in content_nested]
    return content_nested, content_counts


# ---------------------------------------------------------------------------
# Public compress function
# ---------------------------------------------------------------------------

def compress(
    text: str,
    output_path: str,
    model: str | None = None,
) -> Dict[str, Any]:
    normalized = normalize_text(text)

    print("[Stage 4+5] Running discourse analysis...")
    disc_analyser = DiscourseAnalyser(use_spacy=True)
    stage4_result = disc_analyser.analyse_document(normalized)
    discourse_compressed, entity_table = encode_symbols(normalized, stage4_result)
    orig_len = len(normalized)
    disc_len = len(discourse_compressed)
    print(
        f"[Stage 4+5] {orig_len} -> {disc_len} chars "
        f"({100*(orig_len-disc_len)/orig_len:.2f}% reduction)"
    )

    print("[Stage 1b] Frequency-based word substitution...")
    after_word_subs, word_table = encode_word_subs(discourse_compressed)
    if word_table:
        print(f"[Stage 1b] Substituted {len(word_table)} word types.")
    else:
        print("[Stage 1b] No words met frequency threshold.")

    symbol_table = merge_symbol_tables(entity_table, word_table)

    # Stage 1c: strip all §-tokens, record char offsets in clean text.
    # The clean_text is what gets encoded; char offsets let us re-splice
    # symbols into the final joined string after decoding.
    print("[Stage 1c] Extracting symbol slots...")
    clean_text, slot_map = extract_symbols(after_word_subs)
    print(f"[Stage 1c] {len(slot_map)} symbols extracted. "
          f"Chars: {len(after_word_subs)} -> {len(clean_text)} "
          f"(saved {len(after_word_subs) - len(clean_text)} chars)")

    encoded_sentences, pos_freq_table = _encode_sentences(clean_text, model=model)

    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)
    model_bytes = _serialise_model(context_model)
    model_bytes_compressed = zlib.compress(model_bytes, level=9)
    print(
        f"[Stage 7] context_model_data: {len(model_bytes)} -> "
        f"{len(model_bytes_compressed)} bytes "
        f"({100*(len(model_bytes)-len(model_bytes_compressed))/len(model_bytes):.1f}% reduction)"
    )

    char_classes:          List[int]       = []
    pos_deltas_nested:     List[List[int]] = []
    sentence_char_counts:  List[int]       = []
    pos_huffman_bits_list: List[float]     = []
    pos_n_tags_list:       List[int]       = []
    pos_tags_nested:       List[List[str]] = []
    morph_codes_nested:    List[List[int]] = []
    root_lengths_nested:   List[List[int]] = []

    for sent in encoded_sentences:
        cc = sent.get("char_classes", [])
        char_classes.extend(cc)
        pos_deltas_nested.append(sent.get("pos_deltas", []))
        sentence_char_counts.append(len(cc))
        pos_huffman_bits_list.append(float(sent.get("pos_huffman_bits", 0.0)))
        pos_n_tags_list.append(int(sent.get("pos_n_tags", 0)))
        pos_tags_nested.append(sent.get("pos_tags", []))
        morph_codes_nested.append(sent.get("morph_codes", []))
        root_lengths_nested.append([len(r) for r in sent.get("roots", [])])

    print("[Stage 7] Stripping sentinel deltas (per-sentence)...")
    content_nested, content_counts = _strip_sentinel_deltas_per_sentence(
        pos_deltas_nested, root_lengths_nested
    )
    content_flat = [d for sent_c in content_nested for d in sent_c]
    total_orig   = sum(len(s) for s in pos_deltas_nested)
    total_strip  = sum(content_counts)
    print(f"[Stage 7] pos_deltas: {total_orig} -> {total_strip} "
          f"(stripped {total_orig - total_strip} sentinels across {len(pos_deltas_nested)} sentences)")

    print("[Stage 7] Arithmetic encoding char stream...")
    enc              = ArithmeticEncoder()
    compressed_bytes = enc.encode(char_classes, context_model, encoded_sentences)

    print("[Stage 7] Huffman encoding pos_deltas...")
    huff_table_bytes, huff_bitstream = huffman.encode(content_flat)

    print("[Stage 7] LZ77 encoding pos_tags...")
    lz77_n_tokens  = sum(pos_n_tags_list)
    lz77_pos_bytes = pack_pos_tags_lz77(pos_tags_nested, POS_TO_IDX)
    print(f"[Stage 7] pos_tags: {lz77_n_tokens} tokens -> {len(lz77_pos_bytes)} bytes")

    rl_flat   = [l for sent in root_lengths_nested for l in sent]
    rl_counts = [len(sent) for sent in root_lengths_nested]
    rl_huff_table, rl_huff_stream = huffman.encode(rl_flat)
    print(
        f"[Stage 7] root_lengths: {len(rl_flat)} values -> "
        f"{len(rl_huff_stream)} B huffman stream (+{len(rl_huff_table)} B table)"
    )

    mc_data, mc_bits = pack_token_array(morph_codes_nested, 4)

    # Store clean_text in payload so decompressor can use it as
    # the scaling reference for char-offset splice.
    payload: Dict[str, Any] = {
        "lexis_variant":               "reconstructed",
        "symbol_table":                symbol_table,
        "slot_map":                    pack_slot_map(slot_map),
        "slot_clean_text":             clean_text,
        "context_model_data":          model_bytes_compressed,
        "compressed_bitstream":        compressed_bytes,
        "pos_deltas_huffman_table":    huff_table_bytes,
        "pos_deltas_huffman_stream":   huff_bitstream,
        "pos_deltas_total_count":      total_strip,
        "pos_deltas_content_counts":   pack_vlq_list(content_counts),
        "packed_pos_tags_lz77":        lz77_pos_bytes,
        "pos_tags_token_count":        lz77_n_tokens,
        "packed_sentence_char_counts": pack_vlq_list(sentence_char_counts),
        "packed_pos_huffman_bits":     pack_huffman_bits(pos_huffman_bits_list),
        "packed_pos_n_tags":           pack_u8_list(pos_n_tags_list),
        "packed_morph_codes":          (mc_data, mc_bits),
        "root_lengths_huffman_table":  rl_huff_table,
        "root_lengths_huffman_stream": rl_huff_stream,
        "root_lengths_total_count":    len(rl_flat),
        "root_lengths_sent_counts":    pack_vlq_list(rl_counts),
        "packed_pos_freq_table":       pack_pos_freq_table(pos_freq_table),
        "num_symbols":                 len(char_classes),
    }

    packed       = msgpack.packb(payload, use_bin_type=True)
    packed_bytes = cast(bytes, packed)

    print("[Stage 8] zstd compressing payload (level=19)...")
    final_bytes = compress_payload(packed_bytes, level=19)
    print(
        f"[Stage 8] msgpack payload: {len(packed_bytes)} -> "
        f"{len(final_bytes)} bytes "
        f"({100*(len(packed_bytes)-len(final_bytes))/len(packed_bytes):.1f}% reduction)"
    )

    Path(output_path).write_bytes(final_bytes)

    original_size   = len(text.encode("utf-8"))
    compressed_size = len(compressed_bytes)
    print(f"[Lexis-R] Wrote {output_path}")
    print(f"[Lexis-R] char-stream bpb  : {compressed_size * 8 / original_size:.4f}")
    print(f"[Lexis-R] full-payload bpb : {len(final_bytes) * 8 / original_size:.4f}")
    print(f"[Lexis-R] payload size     : {len(final_bytes)} bytes")

    return {
        "original_size":      original_size,
        "compressed_size":    compressed_size,
        "msgpack_size":       len(packed_bytes),
        "payload_size":       len(final_bytes),
        "bpb":                compressed_size * 8 / original_size if original_size else 0.0,
        "full_payload_bpb":   len(final_bytes) * 8 / original_size if original_size else 0.0,
        "discourse_symbols":  len(symbol_table),
    }
