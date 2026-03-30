"""Pipeline entry point and CLI wiring.

Two compression variants are supported:

  Lexis-E  (--variant embedded, default)
      The trained context model (char_context, morph_context, struct_context,
      weights, vocabs) is serialised into the .lexis msgpack payload alongside
      the arithmetic bitstream and structural metadata. Decompression is a
      fast single pass that requires only msgpack — no spaCy re-analysis.
      Trade-off: larger artifact (full-payload bpb ~9.4 on FineWeb-10BT
      after compact encoding, down from ~24.6).

  Lexis-R  (--variant reconstructed)
      The context model is omitted from the payload. At decompress time it
      is reconstructed by re-training a fresh ContextMixingModel on the
      encoded_sentences rebuilt from the stored pos_tags / morph_codes /
      root_lengths arrays. Decompression requires spaCy.
      Trade-off: smaller artifact, slower decompression.

A 'lexis_variant' key is stamped in every payload produced by this module
so the decompressor can identify the format without heuristics. Files
produced before this key was introduced are treated as Lexis-E.

Metadata encoding (all variants)
---------------------------------
All structural metadata is stored in compact binary form rather than
verbose msgpack nested dicts/lists:

  pos_tags        5-bit/token bit-pack  (8-bit sentence-length prefix)
  morph_codes     4-bit/token bit-pack  (same scheme)
  root_lengths    VLQ bytes  (no upper bound — fixes 4-bit clamping bug)
  char_context    flat 7x7   VLQ bytes  (row-major, no keys)
  morph_context   flat 13x7  VLQ bytes
  struct_context  flat 18x7  VLQ bytes + zlib
  pos_huffman_bits  uint16 x100 quantisation (0.01-bit precision)
  pos_n_tags        uint8 list
  sentence_char_counts  uint8 list
  pos_freq_table    18 x uint32 fixed-order flat array
  model_weights     3 x float32 struct
  char/morph_vocab  uint8 flat arrays
  pos_vocab         packed 5-bit ids
  pos_deltas_counts sorted (int8, uint16) pairs
  huffman_codes     DROPPED (reconstructible from pos_freq_table)
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, cast

import msgpack

try:
    from tqdm import tqdm  # type: ignore
except Exception:

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


from compression.alphabet.morph_codes import apply_morph
from compression.alphabet.phonetic_map import PHONETIC_CLASSES
from compression.alphabet.symbol_alphabet import SymbolAlphabet
from compression.config import (
    CHAR_CONTEXT_SIZE,
    MORPH_CONTEXT_SIZE,
    SPACY_MAX_LENGTH,
    SPACY_MODEL,
    STRUCT_CONTEXT_SIZE,
)
from compression.pipeline.stage1_normalize import normalize_text
from compression.pipeline.stage2_morphology import MorphologicalAnalyser
from compression.pipeline.stage3_syntax import analyse_sentence
from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import (
    decode_symbols,
    encode_symbols,
)
from compression.pipeline.stage5_encode import CharacterEncoder, StructuralEncoder
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.stage7_arithmetic import ArithmeticDecoder, ArithmeticEncoder
from compression.pipeline.stage9_autocorrect import autocorrect
from compression.pipeline.utils import chunk_text
from compression.root_lengths_vlq import pack_root_lengths, unpack_root_lengths

# Variant constants
VARIANT_EMBEDDED = "embedded"           # Lexis-E: context model stored in payload
VARIANT_RECONSTRUCTED = "reconstructed" # Lexis-R: context model re-derived at decompress

# Fixed POS vocab (spaCy universal tagset — order is the implicit schema)
_POS_VOCAB = [
    "ADJ", "ADP", "ADV", "AUX", "CONJ", "CCONJ", "DET", "INTJ",
    "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X",
]
_POS_TO_IDX = {tag: i for i, tag in enumerate(_POS_VOCAB)}
_CHAR_CLASSES  = list(range(7))   # char_context outer keys
_MORPH_CODES_R = list(range(13))  # morph_context outer keys (0=BASE … 12=IRREGULAR)


# ---------------------------------------------------------------------------
# Compact binary encoders/decoders
# ---------------------------------------------------------------------------

# ── VLQ (base-128, identical to protobuf varint) ────────────────────────────

def _vlq_encode(value: int) -> bytes:
    assert value >= 0
    groups: list[int] = []
    while True:
        groups.append(value & 0x7F)
        value >>= 7
        if value == 0:
            break
    groups.reverse()
    return bytes([(g | 0x80) if i < len(groups) - 1 else g
                  for i, g in enumerate(groups)])


def _vlq_decode_stream(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    while True:
        b = data[offset]; offset += 1
        value = (value << 7) | (b & 0x7F)
        if not (b & 0x80):
            break
    return value, offset


# ── Token arrays: 5-bit (pos_tags) or 4-bit (morph_codes) ───────────────────
# Layout: [ 8-bit sentence length | N-bit val | N-bit val | ... | next sentence ]
# NOTE: root_lengths now uses VLQ (see compression/root_lengths_vlq.py) so
#       that roots longer than 15 characters are not silently truncated.

def _pack_token_array(sentences: list[list[int]], bits_per_val: int) -> tuple[bytes, int]:
    """Pack nested int list into a compact bitstream. Returns (bytes, n_bits)."""
    bits = ""
    for sentence in sentences:
        bits += format(len(sentence), '08b')
        for val in sentence:
            bits += format(val, f'0{bits_per_val}b')
    n_bits = len(bits)
    n_bytes = (n_bits + 7) // 8
    padded = bits + "0" * (n_bytes * 8 - n_bits)
    data = bytes(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8))
    return data, n_bits


def _unpack_token_array(data: bytes, n_bits: int, bits_per_val: int) -> list[list[int]]:
    """Inverse of _pack_token_array."""
    bits = "".join(format(b, '08b') for b in data)[:n_bits]
    sentences: list[list[int]] = []
    i = 0
    while i + 8 <= len(bits):
        length = int(bits[i:i + 8], 2)
        i += 8
        vals = [
            int(bits[i + j * bits_per_val: i + (j + 1) * bits_per_val], 2)
            for j in range(length)
        ]
        i += length * bits_per_val
        sentences.append(vals)
    return sentences


# ── POS tags: 5-bit/token (18-label vocab) ───────────────────────────────────

def _pack_pos_tags(sentences: list[list[str]]) -> tuple[bytes, int]:
    int_sentences = [[_POS_TO_IDX.get(tag, 17) for tag in s] for s in sentences]
    return _pack_token_array(int_sentences, 5)


def _unpack_pos_tags(data: bytes, n_bits: int) -> list[list[str]]:
    int_sentences = _unpack_token_array(data, n_bits, 5)
    return [[_POS_VOCAB[i] for i in s] for s in int_sentences]


# ── Context matrices: flat VLQ row-major ─────────────────────────────────────

def _pack_context_matrix(ctx: dict, row_keys: list, n_cols: int) -> bytes:
    """Flatten dict-of-dict to VLQ byte stream, row-major. No keys stored."""
    parts = bytearray()
    for key in row_keys:
        row = ctx.get(key, {})
        for col in range(n_cols):
            parts += _vlq_encode(row.get(col, 0))
    return bytes(parts)


def _unpack_context_matrix(data: bytes, row_keys: list, n_cols: int) -> dict:
    ctx: dict = {}
    offset = 0
    for key in row_keys:
        row: dict = {}
        for col in range(n_cols):
            val, offset = _vlq_decode_stream(data, offset)
            if val > 0:
                row[col] = val
        if row:
            ctx[key] = row
    return ctx


# ── pos_huffman_bits: uint16 x100 quantisation ───────────────────────────────

def _pack_huffman_bits(values: list[float]) -> bytes:
    """1B count + 2B per value (scaled x100, 0.01-bit precision)."""
    n = len(values)
    quantized = [min(round(v * 100), 65535) for v in values]
    return bytes([n]) + b"".join(struct.pack(">H", q) for q in quantized)


def _unpack_huffman_bits(data: bytes) -> list[float]:
    n = data[0]
    return [struct.unpack(">H", data[1 + i * 2: 3 + i * 2])[0] / 100.0 for i in range(n)]


# ── uint8 lists (pos_n_tags, sentence_char_counts) ───────────────────────────

def _pack_u8_list(values: list[int]) -> bytes:
    """1B count + 1B per value. Values must fit uint8."""
    clamped = [min(v, 255) for v in values]
    return bytes([len(clamped)] + clamped)


def _unpack_u8_list(data: bytes) -> list[int]:
    n = data[0]
    return list(data[1: n + 1])


# ── pos_freq_table: 18 x uint32 fixed-order flat array ───────────────────────

def _pack_pos_freq_table(table: dict[str, int]) -> bytes:
    """18 uint32 values in _POS_VOCAB order."""
    return b"".join(struct.pack(">I", table.get(tag, 0)) for tag in _POS_VOCAB)


def _unpack_pos_freq_table(data: bytes) -> dict[str, int]:
    return {
        tag: struct.unpack(">I", data[i * 4: i * 4 + 4])[0]
        for i, tag in enumerate(_POS_VOCAB)
        if struct.unpack(">I", data[i * 4: i * 4 + 4])[0] > 0
    }


# ── model_weights: 3 x float32 ───────────────────────────────────────────────

def _pack_weights(weights: list[float]) -> bytes:
    return struct.pack(">3f", *weights[:3])


def _unpack_weights(data: bytes) -> list[float]:
    return list(struct.unpack(">3f", data))


# ── uint8 vocab arrays (char_vocab, morph_vocab) ─────────────────────────────

def _pack_int_vocab(vocab: list[int]) -> bytes:
    return bytes([len(vocab)] + [v & 0xFF for v in vocab])


def _unpack_int_vocab(data: bytes) -> list[int]:
    n = data[0]
    return list(data[1: n + 1])


# ── pos_vocab: packed 5-bit ids ───────────────────────────────────────────────

def _pack_pos_vocab(vocab: list[str]) -> tuple[bytes, int]:
    ids = [_POS_TO_IDX.get(tag, 17) for tag in vocab]
    return _pack_token_array([ids], 5)  # single "sentence"


def _unpack_pos_vocab(data: bytes, n_bits: int) -> list[str]:
    sentences = _unpack_token_array(data, n_bits, 5)
    ids = sentences[0] if sentences else []
    return [_POS_VOCAB[i] for i in ids]


# ── pos_deltas_counts: sorted (int8 + uint16) pairs ──────────────────────────

def _pack_deltas_counts(counts: dict[int, int]) -> bytes:
    """n (1B) + sorted pairs of (int8 delta, uint16 count)."""
    items = sorted(counts.items())
    n = len(items)
    out = bytearray([n])
    for delta, count in items:
        out += struct.pack(">bH", max(-128, min(127, delta)), min(count, 65535))
    return bytes(out)


def _unpack_deltas_counts(data: bytes) -> dict[int, int]:
    n = data[0]
    result: dict[int, int] = {}
    for i in range(n):
        delta, count = struct.unpack(">bH", data[1 + i * 3: 4 + i * 3])
        result[delta] = count
    return result


# ---------------------------------------------------------------------------
# Internal helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _get_nlp(model: str | None = None):
    model_name = model or SPACY_MODEL
    try:
        import spacy

        gpu_fn = getattr(spacy, "prefer_gpu", None)
        if gpu_fn is None:
            gpu_fn = getattr(spacy, "require_gpu", None)

        if callable(gpu_fn):
            if gpu_fn():
                print("[spaCy] GPU enabled")
            else:
                print("[spaCy] Running on CPU")
        else:
            print("[spaCy] Running on CPU")

        nlp = spacy.load(model_name)
        nlp.max_length = SPACY_MAX_LENGTH
        print(f"[spaCy] Loaded model: {model_name}")
        return nlp
    except Exception as exc:
        raise RuntimeError(f"spaCy model '{model_name}' not available: {exc}") from exc


def _run_discourse(text: str, device: str = "cpu") -> tuple[str, dict]:
    analyser = DiscourseAnalyser(use_spacy=True, device=device)
    stage4_result = analyser.analyse_document(text)
    compressed, symbol_table = encode_symbols(text, stage4_result)
    return compressed, symbol_table


def _encode_for_model(
    text: str, model: str | None = None
) -> tuple[list[dict], dict[str, int]]:
    nlp = _get_nlp(model)
    analyser = MorphologicalAnalyser(use_spacy=True, model_name=model)
    symbol_alphabet = SymbolAlphabet()
    structural_encoder = StructuralEncoder(symbol_alphabet)
    char_encoder = CharacterEncoder()

    chunks = list(chunk_text(text))

    sentence_data: list[tuple] = []
    pos_sentences: list[list[str]] = []

    for i, chunk in enumerate(chunks):
        doc = nlp(chunk)
        sents = list(doc.sents)
        for sent in tqdm(sents, desc=f"Chunk {i + 1}/{len(chunks)}", unit="sent"):
            syntax = analyse_sentence(sent)
            morphology = analyser.analyse_sentence(sent.text)
            sentence_data.append((morphology, syntax))
            pos_sentences.append(syntax.pos_tags)

    freq_table = structural_encoder.build_pos_frequency_table(pos_sentences)

    encoded_sentences: list[dict] = []
    for morphology, syntax in tqdm(
        sentence_data, desc="Stage 5 \u2014 Encoding sentences", unit="sent"
    ):
        encoded_sentences.append(
            char_encoder.encode_sentence_full(
                morphology, syntax, structural_encoder, freq_table
            )
        )

    return encoded_sentences, freq_table


def _summarise_pos_huffman(results: list[dict], freq_table: dict[str, int]) -> dict:
    total = len(results)
    total_huffman_bits = sum(float(r.get("pos_huffman_bits", 0.0)) for r in results)
    total_tags = sum(int(r.get("tag_count", 0)) for r in results)
    avg_bits_per_sentence = (total_huffman_bits / total) if total else 0.0
    avg_bits_per_tag = (total_huffman_bits / total_tags) if total_tags else 0.0

    vocab_size = len(freq_table)
    bits_per_tag = math.ceil(math.log2(vocab_size)) if vocab_size > 1 else 1
    flat_bits_baseline = total_tags * bits_per_tag
    bits_saved_vs_flat = flat_bits_baseline - total_huffman_bits

    huffman_codes = results[0].get("pos_huffman_codes", {}) if results else {}

    return {
        "total_sentences": total,
        "total_huffman_bits": total_huffman_bits,
        "avg_bits_per_sentence": avg_bits_per_sentence,
        "avg_bits_per_tag": avg_bits_per_tag,
        "huffman_codes": huffman_codes,
        "flat_bits_baseline": float(flat_bits_baseline),
        "bits_saved_vs_flat": float(bits_saved_vs_flat),
    }


class _EncodedPipeline:
    def __init__(self, encoded_sentences: list[dict]) -> None:
        self._encoded_sentences = encoded_sentences

    def encode_for_model(self, text: str) -> list[dict]:
        return self._encoded_sentences


def _cumulative_from_deltas(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    values = [deltas[0]]
    for delta in deltas[1:]:
        values.append(values[-1] + delta)
    return values


def _reconstruct_chars(
    class_stream: List[int],
    pos_deltas: List[int],
    sentence_char_counts: List[int],
) -> str:
    inverse_map = {coords: char for char, coords in PHONETIC_CLASSES.items()}
    chars: List[str] = []
    idx = 0

    for count in sentence_char_counts:
        segment_classes = class_stream[idx: idx + count]
        segment_deltas = pos_deltas[idx: idx + count]
        positions = _cumulative_from_deltas(segment_deltas)
        for cls, pos in zip(segment_classes, positions):
            char = inverse_map.get((cls, pos))
            if char is not None:
                chars.append(char)
        idx += count

    if idx < len(class_stream):
        segment_classes = class_stream[idx:]
        segment_deltas = pos_deltas[idx:]
        positions = _cumulative_from_deltas(segment_deltas)
        for cls, pos in zip(segment_classes, positions):
            char = inverse_map.get((cls, pos))
            if char is not None:
                chars.append(char)

    return "".join(chars)


def _split_roots(char_stream: str) -> List[str]:
    roots: List[str] = []
    current: List[str] = []
    for char in char_stream:
        if char == "^":
            current = []
        elif char == "$":
            if current:
                roots.append("".join(current))
            current = []
        else:
            current.append(char)
    if current:
        roots.append("".join(current))
    return roots


def _flatten(nested: List[List[Any]]) -> List[Any]:
    return [item for sub in nested for item in sub]


_ATTACH_LEFT  = set(".,;:!?)'-\u2014%-/")
_ATTACH_RIGHT = set("($#/")
_OPEN_QUOTE_AFTER = set("!?(\u2014 ")


def _join_words(words: list[str]) -> str:
    if not words:
        return ""

    parts: list[str] = [words[0]]
    for word in words[1:]:
        if not word:
            continue
        if word == "*":
            if parts[-1].endswith("*"):
                parts[-1] += word
            else:
                parts.append(" " + word)
            continue
        if parts[-1].endswith(("-", "/")):
            parts[-1] += word
            continue
        if parts[-1] and parts[-1][-1] in _ATTACH_RIGHT:
            parts[-1] += word
            continue
        first_char = word[0]
        if first_char == '"':
            prev_last = parts[-1][-1] if parts[-1] else ""
            if prev_last in _OPEN_QUOTE_AFTER or not parts[-1]:
                parts.append(' "')
            else:
                parts[-1] += '"'
            continue
        if first_char == "'":
            parts[-1] += word
        elif first_char in _ATTACH_LEFT:
            parts[-1] += word
        else:
            parts.append(" " + word)

    result = "".join(parts)
    result = result.replace("( ", "(").replace(" )", ")")
    result = result.replace("[ ", "[").replace(" ]", "]")
    result = result.replace(" \u2014 ", "\u2014")
    return result.strip()


def _build_context_model_from_sentences(encoded_sentences: List[Dict]) -> ContextMixingModel:
    model = ContextMixingModel()
    model.train(encoded_sentences)
    return model


def _build_context_model_from_payload(payload: Dict[str, Any]) -> ContextMixingModel:
    """
    Reconstruct ContextMixingModel from payload.
    Supports both new compact encoding (packed_ keys) and legacy verbose encoding.
    """
    model = ContextMixingModel()
    char_context:   Dict = defaultdict(Counter)
    morph_context:  Dict = defaultdict(Counter)
    struct_context: Dict = defaultdict(Counter)

    # ── char_context ──────────────────────────────────────────────────────────
    if "packed_char_context" in payload:
        raw = _unpack_context_matrix(
            bytes(payload["packed_char_context"]), _CHAR_CLASSES, 7
        )
        for k, row in raw.items():
            char_context[k] = Counter(row)
    else:
        for key, counter in payload.get("char_context", {}).items():
            char_context[int(key)] = Counter(
                {int(k): int(v) for k, v in dict(counter).items()}
            )

    # ── morph_context ─────────────────────────────────────────────────────────
    if "packed_morph_context" in payload:
        raw = _unpack_context_matrix(
            bytes(payload["packed_morph_context"]), _MORPH_CODES_R, 7
        )
        for k, row in raw.items():
            morph_context[k] = Counter(row)
    else:
        for key, counter in payload.get("morph_context", {}).items():
            morph_context[int(key)] = Counter(
                {int(k): int(v) for k, v in dict(counter).items()}
            )

    # ── struct_context ────────────────────────────────────────────────────────
    if "packed_struct_context" in payload:
        raw_bytes = bytes(payload["packed_struct_context"])
        decompressed = zlib.decompress(raw_bytes)
        raw = _unpack_context_matrix(decompressed, _POS_VOCAB, 7)
        for k, row in raw.items():
            struct_context[str(k)] = Counter(row)
    else:
        for key, counter in payload.get("struct_context", {}).items():
            struct_context[str(key)] = Counter(
                {int(k): int(v) for k, v in dict(counter).items()}
            )

    model.char_context   = char_context
    model.morph_context  = morph_context
    model.struct_context = struct_context

    # ── weights ───────────────────────────────────────────────────────────────
    if "packed_model_weights" in payload:
        model.weights = _unpack_weights(bytes(payload["packed_model_weights"]))
    else:
        model.weights = list(payload.get("model_weights", model.weights))

    # ── vocabs ────────────────────────────────────────────────────────────────
    if "packed_char_vocab" in payload:
        model.char_vocab = _unpack_int_vocab(bytes(payload["packed_char_vocab"]))
    else:
        model.char_vocab = [int(v) for v in payload.get("char_vocab", model.char_vocab)]

    if "packed_morph_vocab" in payload:
        model.morph_vocab = _unpack_int_vocab(bytes(payload["packed_morph_vocab"]))
    else:
        model.morph_vocab = [int(v) for v in payload.get("morph_vocab", model.morph_vocab)]

    if "packed_pos_vocab" in payload:
        pv_data, pv_bits = payload["packed_pos_vocab"]
        model.pos_vocab = _unpack_pos_vocab(bytes(pv_data), pv_bits)
    else:
        model.pos_vocab = [str(v) for v in payload.get("pos_vocab", model.pos_vocab)]

    return model


def _build_encoded_sentences_from_metadata(payload: Dict[str, Any]) -> List[Dict]:
    # ── pos_tags ──────────────────────────────────────────────────────────────
    if "packed_pos_tags" in payload:
        pt_data, pt_bits = payload["packed_pos_tags"]
        pos_tags = _unpack_pos_tags(bytes(pt_data), pt_bits)
    else:
        pos_tags = payload.get("pos_tags", [])

    # ── morph_codes ───────────────────────────────────────────────────────────
    if "packed_morph_codes" in payload:
        mc_data, mc_bits = payload["packed_morph_codes"]
        morph_codes_nested = _unpack_token_array(bytes(mc_data), mc_bits, 4)
    else:
        morph_codes_nested = payload.get("morph_codes", [])

    # ── root_lengths: prefer VLQ (no clamping), fall back to legacy 4-bit ─────
    if "packed_root_lengths_vlq" in payload:
        root_lengths = unpack_root_lengths(bytes(payload["packed_root_lengths_vlq"]))
    elif "packed_root_lengths" in payload:
        rl_data, rl_bits = payload["packed_root_lengths"]
        root_lengths = _unpack_token_array(bytes(rl_data), rl_bits, 4)
    else:
        root_lengths = payload.get("root_lengths", [])

    # ── pos_huffman_bits ──────────────────────────────────────────────────────
    if "packed_pos_huffman_bits" in payload:
        pos_bits = _unpack_huffman_bits(bytes(payload["packed_pos_huffman_bits"]))
    else:
        pos_bits = payload.get("pos_huffman_bits", [])

    # ── pos_n_tags ────────────────────────────────────────────────────────────
    if "packed_pos_n_tags" in payload:
        pos_n_tags = _unpack_u8_list(bytes(payload["packed_pos_n_tags"]))
    else:
        pos_n_tags = payload.get("pos_n_tags", [])

    encoded: List[Dict] = []
    for idx, lengths in enumerate(root_lengths):
        sentence_pos   = pos_tags[idx]           if idx < len(pos_tags)           else []
        sentence_morph = morph_codes_nested[idx] if idx < len(morph_codes_nested) else []
        char_pos_tags:    List[str] = []
        char_morph_codes: List[int] = []

        for token_idx, length in enumerate(lengths):
            pos_tag    = sentence_pos[token_idx]   if token_idx < len(sentence_pos)   else "X"
            morph_code = sentence_morph[token_idx] if token_idx < len(sentence_morph) else 0
            char_pos_tags.append("X")
            char_morph_codes.append(0)
            char_pos_tags.extend([pos_tag] * length)
            char_morph_codes.extend([morph_code] * length)
            char_pos_tags.append("X")
            char_morph_codes.append(0)
            if token_idx < len(lengths) - 1:
                char_pos_tags.append("X")
                char_morph_codes.append(0)

        encoded.append(
            {
                "char_morph_codes": char_morph_codes,
                "char_pos_tags":    char_pos_tags,
                "pos_huffman_bits": float(pos_bits[idx]) if idx < len(pos_bits) else 0.0,
                "pos_n_tags":       int(pos_n_tags[idx]) if idx < len(pos_n_tags) else 0,
                "pos_tags":         sentence_pos,
                "morph_codes":      sentence_morph,
            }
        )
    return encoded


# ---------------------------------------------------------------------------
# Core compression entry point
# ---------------------------------------------------------------------------

def compress(text: str, output_path: str, model: str | None = None) -> Dict:
    """Run full available pipeline on text (JSON analysis output). Return stats."""
    normalized = normalize_text(text)

    print("[Stage 4+5] Running discourse analysis and symbol encoding...")
    discourse_compressed, symbol_table = _run_discourse(normalized)
    print(f"[Stage 4+5] Symbols encoded: {len(symbol_table)}")

    analyser = MorphologicalAnalyser(use_spacy=True, model_name=model)
    morphology = analyser.analyse_sentence(normalized)

    encoder = CharacterEncoder()
    stats = encoder.stats(normalized)

    encoded_sentences, pos_freq_table = _encode_for_model(normalized, model=model)
    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)
    bpb_value = context_model.bpb(
        normalized, _EncodedPipeline(encoded_sentences)
    )

    payload = {
        "symbol_table":  symbol_table,
        "morphology": [
            {"original": original, "root": root, "code": code}
            for original, root, code in morphology
        ],
        "stats": stats,
        "stage6": {
            "bpb": bpb_value,
            "encoded_sentences": len(encoded_sentences),
            "char_vocab":  context_model.char_vocab,
            "morph_vocab": context_model.morph_vocab,
            "pos_vocab":   context_model.pos_vocab,
            "pos_freq_table": pos_freq_table,
            "weights": context_model.weights,
            "char_context_size":   CHAR_CONTEXT_SIZE,
            "morph_context_size":  MORPH_CONTEXT_SIZE,
            "struct_context_size": STRUCT_CONTEXT_SIZE,
            "global_char_distribution": context_model.global_char_distribution(),
        },
    }

    Path(output_path).write_text(json.dumps(payload, indent=2))
    return payload


def compress_to_file(
    text: str,
    output_path: str,
    model: str | None = None,
    store_model: bool = True,
) -> Dict:
    """
    Full compression pipeline with arithmetic coding.

    Parameters
    ----------
    store_model : bool
        True  → Lexis-E: serialise compact context model into payload.
        False → Lexis-R: omit context model; re-trained at decompress time.
    """
    normalized = normalize_text(text)

    print("[Stage 4+5] Running discourse analysis and symbol encoding...")
    discourse_compressed, symbol_table = _run_discourse(normalized)
    print(f"[Stage 4+5] Symbols encoded: {len(symbol_table)}")
    orig_len = len(normalized)
    disc_len = len(discourse_compressed)
    print(
        f"[Stage 4+5] Text length: {orig_len} \u2192 {disc_len} "
        f"({100 * (orig_len - disc_len) / orig_len:.2f}% reduction)"
    )

    encoded_sentences, pos_freq_table = _encode_for_model(normalized, model=model)

    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)

    char_classes:          List[int]        = []
    char_pos_deltas:       List[int]        = []
    sentence_char_counts:  List[int]        = []
    pos_huffman_bits_list: List[float]      = []
    pos_n_tags_list:       List[int]        = []
    pos_tags_nested:       List[List[str]]  = []
    morph_codes_nested:    List[List[int]]  = []
    root_lengths_nested:   List[List[int]]  = []

    for sentence in encoded_sentences:
        char_classes.extend(sentence.get("char_classes", []))
        char_pos_deltas.extend(sentence.get("pos_deltas", []))
        sentence_char_counts.append(len(sentence.get("char_classes", [])))
        pos_huffman_bits_list.append(float(sentence.get("pos_huffman_bits", 0.0)))
        pos_n_tags_list.append(int(sentence.get("pos_n_tags", 0)))
        pos_tags_nested.append(sentence.get("pos_tags", []))
        morph_codes_nested.append(sentence.get("morph_codes", []))
        root_lengths_nested.append([len(root) for root in sentence.get("roots", [])])

    # ── Arithmetic encode ─────────────────────────────────────────────────────
    encoder = ArithmeticEncoder()
    compressed_bytes = encoder.encode(
        char_classes, context_model, {}, encoded_sentences
    )
    pos_delta_counts  = Counter(char_pos_deltas)
    pos_deltas_stream = encoder.encode_unigram_counts(char_pos_deltas, pos_delta_counts)

    # ── Pack all token arrays ─────────────────────────────────────────────────
    pt_data,  pt_bits  = _pack_pos_tags(pos_tags_nested)
    mc_data,  mc_bits  = _pack_token_array(morph_codes_nested, 4)
    # root_lengths: use VLQ (no upper bound) — old 4-bit key kept for compat
    rl_vlq             = pack_root_lengths(root_lengths_nested)
    rl_data,  rl_bits  = _pack_token_array(root_lengths_nested, 4)  # legacy fallback

    # ── Pack sentence-level arrays ────────────────────────────────────────────
    phb_data = _pack_huffman_bits(pos_huffman_bits_list)
    pnt_data = _pack_u8_list(pos_n_tags_list)
    scc_data = _pack_u8_list(sentence_char_counts)

    # ── Pack pos_freq_table and pos_deltas_counts ─────────────────────────────
    pft_data = _pack_pos_freq_table(pos_freq_table)
    pdc_data = _pack_deltas_counts(
        {int(k): int(v) for k, v in pos_delta_counts.items()}
    )

    variant = VARIANT_EMBEDDED if store_model else VARIANT_RECONSTRUCTED

    metadata: Dict[str, Any] = {
        "lexis_variant":               variant,
        "symbol_table":                symbol_table,
        "compressed_bitstream":        compressed_bytes,
        "pos_deltas_bitstream":        pos_deltas_stream,
        "packed_pos_deltas_counts":    pdc_data,
        "pos_deltas_count":            len(char_pos_deltas),
        "packed_sentence_char_counts": scc_data,
        "packed_pos_huffman_bits":     phb_data,
        "packed_pos_n_tags":           pnt_data,
        "packed_pos_tags":             (pt_data, pt_bits),
        "packed_morph_codes":          (mc_data, mc_bits),
        # VLQ key (preferred by decompressor — no length clamping)
        "packed_root_lengths_vlq":     rl_vlq,
        # Legacy 4-bit key kept so old builds can still read new files
        "packed_root_lengths":         (rl_data, rl_bits),
        "num_symbols":                 len(char_classes),
        "num_char_classes":            7,
        "packed_pos_freq_table":       pft_data,
    }

    if store_model:
        # Lexis-E: embed compact context model.
        pv_data, pv_bits = _pack_pos_vocab(context_model.pos_vocab)
        metadata.update({
            "packed_model_weights":  _pack_weights(context_model.weights),
            "packed_char_context":   _pack_context_matrix(
                context_model.char_context, _CHAR_CLASSES, 7
            ),
            "packed_morph_context":  _pack_context_matrix(
                {k: dict(v) for k, v in context_model.morph_context.items()},
                _MORPH_CODES_R, 7
            ),
            "packed_struct_context": zlib.compress(
                _pack_context_matrix(
                    {k: dict(v) for k, v in context_model.struct_context.items()},
                    _POS_VOCAB, 7
                ), level=9
            ),
            "packed_char_vocab":  _pack_int_vocab(context_model.char_vocab),
            "packed_morph_vocab": _pack_int_vocab(context_model.morph_vocab),
            "packed_pos_vocab":   (pv_data, pv_bits),
        })

    packed       = msgpack.packb(metadata, use_bin_type=True)
    packed_bytes = cast(bytes, packed)
    Path(output_path).write_bytes(packed_bytes)

    original_size   = len(text.encode("utf-8"))
    compressed_size = len(compressed_bytes)
    return {
        "lexis_variant":           variant,
        "original_size":           original_size,
        "compressed_size":         compressed_size,
        "compression_ratio":       compressed_size / original_size if original_size else 0.0,
        "bpb":                     (compressed_size * 8) / original_size if original_size else 0.0,
        "payload_size":            len(packed_bytes),
        "full_payload_bpb":        (len(packed_bytes) * 8) / original_size if original_size else 0.0,
        "discourse_symbols":       len(symbol_table),
        "discourse_reduction_pct": round(100 * (orig_len - disc_len) / orig_len, 2)
        if orig_len else 0.0,
    }


# ---------------------------------------------------------------------------
# Decompression
# ---------------------------------------------------------------------------

def decompress(input_path: str) -> str:
    """
    Decompress a .lexis msgpack file.

    Format detection via the 'lexis_variant' key:
      'embedded'      → Lexis-E: load compact serialised context model.
      'reconstructed' → Lexis-R: re-train from structural metadata.
      (absent)        → legacy: treat as Lexis-E if char_context present.
    """
    raw = Path(input_path).read_bytes()
    payload: Dict[str, Any]
    try:
        payload = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception:
        payload = json.loads(raw.decode("utf-8"))

    symbol_table: dict = payload.get("symbol_table", {})

    if isinstance(payload, dict) and "compressed_bitstream" in payload:
        encoded_sentences = _build_encoded_sentences_from_metadata(payload)

        variant = payload.get("lexis_variant", None)
        use_embedded = (
            variant == VARIANT_EMBEDDED
            or (
                variant is None
                and ("char_context" in payload or "packed_char_context" in payload)
            )
        )

        if use_embedded:
            print("[Decompress] Lexis-E: loading embedded context model...")
            context_model = _build_context_model_from_payload(payload)
        else:
            print("[Decompress] Lexis-R: re-training context model from metadata...")
            context_model = _build_context_model_from_sentences(encoded_sentences)

        decoder     = ArithmeticDecoder()
        num_symbols = int(
            payload.get("num_symbols", len(payload.get("char_morph_codes", [])))
        )
        char_classes = decoder.decode(
            payload["compressed_bitstream"],
            context_model,
            encoded_sentences,
            num_symbols,
        )

        # ── pos_deltas ────────────────────────────────────────────────────────
        pos_deltas_stream = payload.get("pos_deltas_bitstream", b"")
        pos_deltas_count  = int(payload.get("pos_deltas_count", 0))

        if "packed_pos_deltas_counts" in payload:
            counts = _unpack_deltas_counts(bytes(payload["packed_pos_deltas_counts"]))
        else:
            raw_counts = payload.get("pos_deltas_counts", {})
            counts = {int(k): int(v) for k, v in raw_counts.items()}

        pos_deltas = ArithmeticDecoder().decode_unigram_counts(
            bytes(pos_deltas_stream), counts, pos_deltas_count
        )

        # ── sentence_char_counts ──────────────────────────────────────────────
        if "packed_sentence_char_counts" in payload:
            sentence_counts = _unpack_u8_list(
                bytes(payload["packed_sentence_char_counts"])
            )
        else:
            sentence_counts = [int(c) for c in payload.get("sentence_char_counts", [])]

        char_stream = _reconstruct_chars(char_classes, pos_deltas, sentence_counts)
        roots       = _split_roots(char_stream)

        # ── morph_codes (for apply_morph) ─────────────────────────────────────
        if "packed_morph_codes" in payload:
            mc_data, mc_bits = payload["packed_morph_codes"]
            morph_codes_nested = _unpack_token_array(bytes(mc_data), mc_bits, 4)
        else:
            morph_codes_nested = payload.get("morph_codes", [])

        morph_codes_flat = _flatten(morph_codes_nested)
        words  = [
            apply_morph(root, morph_codes_flat[idx] if idx < len(morph_codes_flat) else 0)
            for idx, root in enumerate(roots)
        ]
        result = _join_words(words)
        result = result[0].upper() + result[1:] if result else result

        if symbol_table:
            result = decode_symbols(result, symbol_table)

        return autocorrect(result)

    if isinstance(payload, dict) and "morphology" in payload:
        words  = [
            apply_morph(entry["root"], entry["code"])
            for entry in payload.get("morphology", [])
        ]
        result = _join_words(words)
        result = result[0].upper() + result[1:] if result else result
        if symbol_table:
            result = decode_symbols(result, symbol_table)
        return autocorrect(result)

    return ""


# ---------------------------------------------------------------------------
# Analysis mode
# ---------------------------------------------------------------------------

def analyse(text: str, model: str | None = None) -> None:
    """Run pipeline in analysis mode — print stats at each stage."""
    normalized = normalize_text(text)

    print("[Stage 4+5] Running discourse analysis...")
    discourse_compressed, symbol_table = _run_discourse(normalized)
    orig_tokens = len(normalized.split())
    comp_tokens = len(discourse_compressed.split())
    print(f"[Stage 4+5] Symbols: {len(symbol_table)}")
    print(
        f"[Stage 4+5] Token reduction: {orig_tokens} \u2192 {comp_tokens} "
        f"({100 * (orig_tokens - comp_tokens) / orig_tokens:.2f}%)"
    )

    analyser    = MorphologicalAnalyser(use_spacy=True, model_name=model)
    morph_stats = analyser.char_savings(normalized)

    encoder      = CharacterEncoder()
    encode_stats = encoder.stats(normalized)

    encoded_sentences, pos_freq_table = _encode_for_model(normalized, model=model)
    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)
    bpb_value = context_model.bpb(
        normalized, _EncodedPipeline(encoded_sentences)
    )

    print(f"Model: {model or SPACY_MODEL}")
    print(f"Context-mixing bpb: {bpb_value:.4f}")

    print("Stage 2 \u2014 Morphology")
    for key, value in morph_stats.items():
        print(f"  {key}: {value}")

    print("Stage 5 \u2014 Character Encoding")
    for key, value in encode_stats.items():
        print(f"  {key}: {value}")

    pos_huffman_results = [
        {
            "pos_huffman_bits": sentence.get("pos_huffman_bits", 0),
            "tag_count":        sentence.get("pos_encoding", {}).get("tag_count", 0),
            "pos_huffman_codes": sentence.get("pos_huffman_codes", {}),
        }
        for sentence in encoded_sentences
    ]
    pos_huffman_summary = _summarise_pos_huffman(pos_huffman_results, pos_freq_table)

    print("Stage 5 \u2014 POS Huffman Summary")
    for key, value in pos_huffman_summary.items():
        print(f"  {key}: {value}")

    print("Stage 6 \u2014 Probability Model (Context Mixing)")
    print(f"  bpb: {bpb_value}")
    print(f"  char_vocab_size:  {len(context_model.char_vocab)}")
    print(f"  morph_vocab_size: {len(context_model.morph_vocab)}")
    print(f"  pos_vocab_size:   {len(context_model.pos_vocab)}")
    print(f"  weights: {[round(w, 4) for w in context_model.weights]}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical text compression pipeline (Lexis-E / Lexis-R)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="Compress input text")
    compress_parser.add_argument("input",  help="Input text file")
    compress_parser.add_argument("output", help="Output .lexis file")
    compress_parser.add_argument("--model", default=None, help="spaCy model to use")
    compress_parser.add_argument(
        "--variant",
        choices=[VARIANT_EMBEDDED, VARIANT_RECONSTRUCTED],
        default=VARIANT_EMBEDDED,
        help=(
            "'embedded' (default, Lexis-E): store compact context model for fast "
            "decompression. 'reconstructed' (Lexis-R): omit context model; re-derive "
            "at decompress time for a smaller artifact."
        ),
    )

    decompress_parser = subparsers.add_parser("decompress", help="Decompress a .lexis file")
    decompress_parser.add_argument("input", help="Input .lexis file")

    analyse_parser = subparsers.add_parser("analyse", help="Analyse input text")
    analyse_parser.add_argument("input",  help="Input text file")
    analyse_parser.add_argument("--model", default=None)

    args = parser.parse_args()

    if args.command == "compress":
        text        = _read_text(args.input)
        store_model = args.variant == VARIANT_EMBEDDED
        result      = compress_to_file(text, args.output, model=args.model, store_model=store_model)
        variant_label = "Lexis-E" if store_model else "Lexis-R"
        print(f"[{variant_label}] Wrote compressed payload to {args.output}")
        print(f"[{variant_label}] char-stream bpb:    {result['bpb']:.4f}")
        print(f"[{variant_label}] full-payload bpb:   {result['full_payload_bpb']:.4f}")
        print(f"[{variant_label}] payload size:       {result['payload_size']} bytes")
    elif args.command == "decompress":
        reconstructed = decompress(args.input)
        print(reconstructed)
    elif args.command == "analyse":
        text = _read_text(args.input)
        analyse(text, model=args.model)


if __name__ == "__main__":
    main()
