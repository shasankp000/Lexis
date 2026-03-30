"""Lexis-R compressor.

Writes a self-contained .lexisr msgpack file.  The trained
ContextMixingModel is serialised directly into the payload so the
decompressor can load it byte-for-exact-byte — no re-training needed.

Usage
-----
    from lexis_r.compress import compress
    stats = compress(text, "output.lexisr")
"""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
from typing import Any, cast, Dict, List

import msgpack

from compression.alphabet.symbol_alphabet import SymbolAlphabet
from compression.pipeline.stage1_normalize import normalize_text
from compression.pipeline.stage2_morphology import MorphologicalAnalyser
from compression.pipeline.stage3_syntax import analyse_sentence
from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import encode_symbols
from compression.pipeline.stage5_encode import CharacterEncoder, StructuralEncoder
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.utils import chunk_text
from lexis_r.arithmetic import ArithmeticEncoder
from lexis_r.payload import (
    pack_deltas_counts,
    pack_huffman_bits,
    pack_pos_freq_table,
    pack_pos_tags,
    pack_root_lengths,
    pack_token_array,
    pack_u8_list,
)

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
    """Run stages 2-5 and return (encoded_sentences, pos_freq_table)."""
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
    """Serialise a trained ContextMixingModel to raw bytes (via temp file)."""
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".lcm", delete=False) as tf:
        tmp_path = tf.name
    try:
        model.serialise(tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Public compress function
# ---------------------------------------------------------------------------

def compress(
    text: str,
    output_path: str,
    model: str | None = None,
) -> Dict[str, Any]:
    """
    Compress text into a Lexis-R .lexisr file.

    The trained ContextMixingModel is serialised and stored verbatim
    in the payload so the decompressor can load it exactly.

    Returns a dict of compression statistics.
    """
    # ─ Stage 1: normalise ──────────────────────────────────────────────────
    normalized = normalize_text(text)

    # ─ Stage 4+5: discourse + symbol encoding ──────────────────────────────
    print("[Stage 4+5] Running discourse analysis...")
    disc_analyser = DiscourseAnalyser(use_spacy=True)
    stage4_result = disc_analyser.analyse_document(normalized)
    discourse_compressed, symbol_table = encode_symbols(normalized, stage4_result)
    orig_len = len(normalized)
    disc_len = len(discourse_compressed)
    print(
        f"[Stage 4+5] {orig_len} → {disc_len} chars "
        f"({100*(orig_len-disc_len)/orig_len:.2f}% reduction)"
    )

    # ─ Stages 2-5: morphology + syntax + character encoding ────────────────
    encoded_sentences, pos_freq_table = _encode_sentences(normalized, model=model)

    # ─ Stage 6: train context model ────────────────────────────────────────
    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)
    model_bytes = _serialise_model(context_model)   # <── stored in payload

    # ─ Gather per-sentence streams ─────────────────────────────────────────
    char_classes:          List[int]       = []
    char_pos_deltas:       List[int]       = []
    sentence_char_counts:  List[int]       = []
    pos_huffman_bits_list: List[float]     = []
    pos_n_tags_list:       List[int]       = []
    pos_tags_nested:       List[List[str]] = []
    morph_codes_nested:    List[List[int]] = []
    root_lengths_nested:   List[List[int]] = []

    for sent in encoded_sentences:
        cc = sent.get("char_classes", [])
        char_classes.extend(cc)
        char_pos_deltas.extend(sent.get("pos_deltas", []))
        sentence_char_counts.append(len(cc))
        pos_huffman_bits_list.append(float(sent.get("pos_huffman_bits", 0.0)))
        pos_n_tags_list.append(int(sent.get("pos_n_tags", 0)))
        pos_tags_nested.append(sent.get("pos_tags", []))
        morph_codes_nested.append(sent.get("morph_codes", []))
        root_lengths_nested.append([len(r) for r in sent.get("roots", [])])

    # ─ Stage 7: arithmetic encode char stream ──────────────────────────────
    print("[Stage 7] Arithmetic encoding...")
    enc              = ArithmeticEncoder()
    compressed_bytes = enc.encode(char_classes, context_model, encoded_sentences)

    # ─ Encode pos-delta stream with its own unigram coder ──────────────────
    pos_delta_counts = Counter(char_pos_deltas)
    pos_delta_bytes  = enc.encode_unigram_counts(
        char_pos_deltas,
        {int(k): int(v) for k, v in pos_delta_counts.items()},
    )

    # ─ Pack structural metadata ────────────────────────────────────────────
    pt_data, pt_bits = pack_pos_tags(pos_tags_nested)
    mc_data, mc_bits = pack_token_array(morph_codes_nested, 4)
    rl_vlq           = pack_root_lengths(root_lengths_nested)

    payload: Dict[str, Any] = {
        "lexis_variant":               "reconstructed",
        "symbol_table":                symbol_table,
        "context_model_data":          model_bytes,     # <── serialised model
        "compressed_bitstream":        compressed_bytes,
        "pos_deltas_bitstream":        pos_delta_bytes,
        "packed_pos_deltas_counts":    pack_deltas_counts(
            {int(k): int(v) for k, v in pos_delta_counts.items()}
        ),
        "pos_deltas_count":            len(char_pos_deltas),
        "packed_sentence_char_counts": pack_u8_list(sentence_char_counts),
        "packed_pos_huffman_bits":     pack_huffman_bits(pos_huffman_bits_list),
        "packed_pos_n_tags":           pack_u8_list(pos_n_tags_list),
        "packed_pos_tags":             (pt_data, pt_bits),
        "packed_morph_codes":          (mc_data, mc_bits),
        "packed_root_lengths_vlq":     rl_vlq,
        "packed_pos_freq_table":       pack_pos_freq_table(pos_freq_table),
        "num_symbols":                 len(char_classes),
    }

    packed       = msgpack.packb(payload, use_bin_type=True)
    packed_bytes = cast(bytes, packed)
    Path(output_path).write_bytes(packed_bytes)

    original_size   = len(text.encode("utf-8"))
    compressed_size = len(compressed_bytes)
    print(f"[Lexis-R] Wrote {output_path}")
    print(f"[Lexis-R] char-stream bpb  : {compressed_size * 8 / original_size:.4f}")
    print(f"[Lexis-R] full-payload bpb : {len(packed_bytes) * 8 / original_size:.4f}")
    print(f"[Lexis-R] payload size     : {len(packed_bytes)} bytes")

    return {
        "original_size":    original_size,
        "compressed_size":  compressed_size,
        "payload_size":     len(packed_bytes),
        "bpb":              compressed_size * 8 / original_size if original_size else 0.0,
        "full_payload_bpb": len(packed_bytes) * 8 / original_size if original_size else 0.0,
        "discourse_symbols": len(symbol_table),
    }
