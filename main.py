"""Pipeline entry point and CLI wiring."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, cast

from compression.metadata_codec import decode_metadata, encode_metadata, is_lexi_file

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
    """
    Run Stage 4 (coreference resolution) + Stage 5 (symbol encoding).

    Returns (compressed_text, symbol_table). The compressed_text has repeated
    named-entity mentions replaced with §E{n} symbols. The symbol_table is
    needed at decode time to restore the original text.
    """
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
        sentence_data, desc="Stage 5 — Encoding sentences", unit="sent"
    ):
        encoded_sentences.append(
            char_encoder.encode_sentence_full(
                morphology, syntax, structural_encoder, freq_table
            )
        )

    return encoded_sentences, freq_table


def _summarise_pos_huffman(results: list[dict], freq_table: dict[str, int]) -> dict:
    """Aggregate POS Huffman encoding into a summary."""
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
        segment_classes = class_stream[idx : idx + count]
        segment_deltas = pos_deltas[idx : idx + count]
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


_ATTACH_LEFT  = set(".,;:!?)'-—%-/")
_ATTACH_RIGHT = set("($#/")
_OPEN_QUOTE_AFTER = set("!?(— ")


def _join_words(words: list[str]) -> str:
    """Join tokens intelligently, suppressing spaces around punctuation."""
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
    result = result.replace(" — ", "—")
    return result.strip()


def _build_context_model(payload: Dict[str, Any]) -> ContextMixingModel:
    model = ContextMixingModel()
    char_context   = defaultdict(Counter)
    morph_context  = defaultdict(Counter)
    struct_context = defaultdict(Counter)

    for key, counter in payload.get("char_context", {}).items():
        char_context[int(key)] = Counter(
            {int(k): int(v) for k, v in dict(counter).items()}
        )
    for key, counter in payload.get("morph_context", {}).items():
        morph_context[int(key)] = Counter(
            {int(k): int(v) for k, v in dict(counter).items()}
        )
    for key, counter in payload.get("struct_context", {}).items():
        struct_context[str(key)] = Counter(
            {int(k): int(v) for k, v in dict(counter).items()}
        )

    model.char_context   = char_context
    model.morph_context  = morph_context
    model.struct_context = struct_context
    model.weights   = list(payload.get("model_weights", model.weights))
    model.char_vocab  = [int(v) for v in payload.get("char_vocab",  model.char_vocab)]
    model.morph_vocab = [int(v) for v in payload.get("morph_vocab", model.morph_vocab)]
    model.pos_vocab   = [str(v) for v in payload.get("pos_vocab",   model.pos_vocab)]
    return model


def _build_encoded_sentences_from_metadata(payload: Dict[str, Any]) -> List[Dict]:
    root_lengths = payload.get("root_lengths", [])
    pos_bits     = payload.get("pos_huffman_bits", [])
    pos_n_tags   = payload.get("pos_n_tags", [])
    pos_tags     = payload.get("pos_tags", [])
    morph_codes  = payload.get("morph_codes", [])

    encoded: List[Dict] = []
    for idx, lengths in enumerate(root_lengths):
        sentence_pos   = pos_tags[idx]    if idx < len(pos_tags)    else []
        sentence_morph = morph_codes[idx] if idx < len(morph_codes) else []
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
            }
        )
    return encoded


def compress(text: str, output_path: str, model: str | None = None) -> Dict:
    """Run full available pipeline on text. Return stats."""
    normalized = normalize_text(text)

    print("[Stage 4+5] Running discourse analysis and symbol encoding...")
    discourse_compressed, symbol_table = _run_discourse(normalized)
    print(f"[Stage 4+5] Symbols encoded: {len(symbol_table)}")

    analyser   = MorphologicalAnalyser(use_spacy=True, model_name=model)
    morphology = analyser.analyse_sentence(normalized)

    encoder = CharacterEncoder()
    stats   = encoder.stats(normalized)

    encoded_sentences, pos_freq_table = _encode_for_model(normalized, model=model)
    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)
    bpb_value = context_model.bpb(normalized, _EncodedPipeline(encoded_sentences))

    payload = {
        "symbol_table": symbol_table,
        "morphology": [
            {"original": original, "root": root, "code": code}
            for original, root, code in morphology
        ],
        "stats": stats,
        "stage6": {
            "bpb": bpb_value,
            "encoded_sentences": len(encoded_sentences),
            "char_vocab": context_model.char_vocab,
            "morph_vocab": context_model.morph_vocab,
            "pos_vocab": context_model.pos_vocab,
            "pos_freq_table": pos_freq_table,
            "weights": context_model.weights,
            "char_context_size": CHAR_CONTEXT_SIZE,
            "morph_context_size": MORPH_CONTEXT_SIZE,
            "struct_context_size": STRUCT_CONTEXT_SIZE,
            "global_char_distribution": context_model.global_char_distribution(),
        },
    }

    Path(output_path).write_text(json.dumps(payload, indent=2))
    return payload


def compress_to_file(text: str, output_path: str, model: str | None = None) -> Dict:
    """
    Full compression pipeline with arithmetic coding.

    Writes a LEXI-envelope binary file (Option-B, field-id + length-prefix).
    Stage 4+5 discourse symbol encoding runs on the normalised text to produce
    a symbol_table.  The character/morphology pipeline encodes the *original
    normalised text* (not the §-symbol version) because §E{n} tokens are
    outside the phonetic alphabet.  The symbol_table is stored in the payload
    and applied as the very last step in decompress().
    """
    normalized = normalize_text(text)

    print("[Stage 4+5] Running discourse analysis and symbol encoding...")
    discourse_compressed, symbol_table = _run_discourse(normalized)
    print(f"[Stage 4+5] Symbols encoded: {len(symbol_table)}")
    orig_len = len(normalized)
    disc_len = len(discourse_compressed)
    print(
        f"[Stage 4+5] Text length: {orig_len} → {disc_len} "
        f"({100*(orig_len-disc_len)/orig_len:.2f}% reduction)"
    )

    encoded_sentences, pos_freq_table = _encode_for_model(normalized, model=model)

    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)

    char_classes:          List[int]        = []
    char_pos_deltas:       List[int]        = []
    sentence_char_counts:  List[int]        = []
    pos_huffman_bits:      List[float]      = []
    pos_n_tags:            List[int]        = []
    pos_tags:              List[List[str]]  = []
    morph_codes:           List[List[int]]  = []
    root_lengths:          List[List[int]]  = []

    for sentence in encoded_sentences:
        char_classes.extend(sentence.get("char_classes", []))
        char_pos_deltas.extend(sentence.get("pos_deltas", []))
        sentence_char_counts.append(len(sentence.get("char_classes", [])))
        pos_huffman_bits.append(float(sentence.get("pos_huffman_bits", 0.0)))
        pos_n_tags.append(int(sentence.get("pos_n_tags", 0)))
        pos_tags.append(sentence.get("pos_tags", []))
        morph_codes.append(sentence.get("morph_codes", []))
        root_lengths.append([len(root) for root in sentence.get("roots", [])])

    arith_enc = ArithmeticEncoder()
    compressed_bytes = arith_enc.encode(
        char_classes, context_model, {}, encoded_sentences
    )

    pos_delta_counts  = Counter(char_pos_deltas)
    pos_deltas_stream = arith_enc.encode_unigram_counts(char_pos_deltas, pos_delta_counts)

    metadata = {
        # bitstreams (raw bytes — not mode-encoded)
        "compressed_bitstream":  compressed_bytes,
        "pos_deltas_bitstream":  pos_deltas_stream,
        # Stage 4+5
        "symbol_table":          symbol_table,
        # deltas
        "pos_deltas_counts":     {int(k): int(v) for k, v in pos_delta_counts.items()},
        "pos_deltas_count":      len(char_pos_deltas),
        # per-sentence arrays
        "sentence_char_counts": sentence_char_counts,
        "pos_huffman_bits":      pos_huffman_bits,
        "pos_n_tags":            pos_n_tags,
        "pos_tags":              pos_tags,
        "morph_codes":           morph_codes,
        "root_lengths":          root_lengths,
        # context model
        "model_weights":  context_model.weights,
        "char_context":   {int(k): dict(v) for k, v in context_model.char_context.items()},
        "morph_context":  {int(k): dict(v) for k, v in context_model.morph_context.items()},
        "struct_context": {str(k): dict(v) for k, v in context_model.struct_context.items()},
        "char_vocab":     context_model.char_vocab,
        "morph_vocab":    context_model.morph_vocab,
        "pos_vocab":      context_model.pos_vocab,
        # misc scalars
        "num_symbols":      len(char_classes),
        "num_char_classes": 7,
        "pos_freq_table":   pos_freq_table,
    }

    # --- encode with the new LEXI binary envelope (replaces msgpack) ---
    binary = encode_metadata(metadata)
    Path(output_path).write_bytes(binary)

    original_size   = len(text.encode("utf-8"))
    compressed_size = len(compressed_bytes)
    return {
        "original_size":          original_size,
        "compressed_size":        compressed_size,
        "compression_ratio":      compressed_size / original_size if original_size else 0.0,
        "bpb":                    (compressed_size * 8) / original_size if original_size else 0.0,
        "discourse_symbols":      len(symbol_table),
        "discourse_reduction_pct": round(100 * (orig_len - disc_len) / orig_len, 2)
                                   if orig_len else 0.0,
    }


def decompress(input_path: str) -> str:
    """Decompress a .lexis file.  Auto-detects LEXI envelope, msgpack, or JSON."""
    raw = Path(input_path).read_bytes()
    payload: Dict[str, Any]

    if is_lexi_file(raw):
        # --- new LEXI binary envelope ---
        payload = decode_metadata(raw)
    else:
        # --- legacy msgpack / JSON fallback (read-only, never written) ---
        try:
            import msgpack  # type: ignore
            payload = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        except Exception:
            payload = json.loads(raw.decode("utf-8"))

    symbol_table: dict = payload.get("symbol_table", {})

    if isinstance(payload, dict) and "compressed_bitstream" in payload:
        context_model     = _build_context_model(payload)
        encoded_sentences = _build_encoded_sentences_from_metadata(payload)

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

        pos_deltas_stream  = payload.get("pos_deltas_bitstream", b"")
        pos_deltas_counts  = payload.get("pos_deltas_counts", {})
        pos_deltas_count   = int(payload.get("pos_deltas_count", 0))
        counts = {int(k): int(v) for k, v in pos_deltas_counts.items()}
        pos_deltas = ArithmeticDecoder().decode_unigram_counts(
            bytes(pos_deltas_stream), counts, pos_deltas_count
        )

        sentence_counts = [int(c) for c in payload.get("sentence_char_counts", [])]
        char_stream     = _reconstruct_chars(char_classes, pos_deltas, sentence_counts)
        roots           = _split_roots(char_stream)

        morph_codes_flat = _flatten(payload.get("morph_codes", []))
        words = [
            apply_morph(root, morph_codes_flat[idx] if idx < len(morph_codes_flat) else 0)
            for idx, root in enumerate(roots)
        ]
        result = _join_words(words)
        result = result[0].upper() + result[1:] if result else result

        if symbol_table:
            result = decode_symbols(result, symbol_table)

        return autocorrect(result)

    if isinstance(payload, dict) and "morphology" in payload:
        words = [
            apply_morph(entry["root"], entry["code"])
            for entry in payload.get("morphology", [])
        ]
        result = _join_words(words)
        result = result[0].upper() + result[1:] if result else result
        if symbol_table:
            result = decode_symbols(result, symbol_table)
        return autocorrect(result)

    return ""


def analyse(text: str, model: str | None = None) -> None:
    """Run pipeline in analysis mode — print stats at each stage."""
    normalized = normalize_text(text)

    print("[Stage 4+5] Running discourse analysis...")
    discourse_compressed, symbol_table = _run_discourse(normalized)
    orig_tokens = len(normalized.split())
    comp_tokens = len(discourse_compressed.split())
    print(f"[Stage 4+5] Symbols: {len(symbol_table)}")
    print(
        f"[Stage 4+5] Token reduction: {orig_tokens} → {comp_tokens} "
        f"({100*(orig_tokens-comp_tokens)/orig_tokens:.2f}%)"
    )

    analyser   = MorphologicalAnalyser(use_spacy=True, model_name=model)
    morph_stats = analyser.char_savings(normalized)

    encoder      = CharacterEncoder()
    encode_stats = encoder.stats(normalized)

    encoded_sentences, pos_freq_table = _encode_for_model(normalized, model=model)
    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)
    bpb_value = context_model.bpb(normalized, _EncodedPipeline(encoded_sentences))

    print(f"Model: {model or SPACY_MODEL}")
    print(f"Context-mixing bpb: {bpb_value:.4f}")

    print("Stage 2 — Morphology")
    for key, value in morph_stats.items():
        print(f"  {key}: {value}")

    print("Stage 5 — Character Encoding")
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

    print("Stage 5 — POS Huffman Summary")
    for key, value in pos_huffman_summary.items():
        print(f"  {key}: {value}")

    print("Stage 6 — Probability Model (Context Mixing)")
    print(f"  bpb: {bpb_value}")
    print(f"  char_vocab_size: {len(context_model.char_vocab)}")
    print(f"  morph_vocab_size: {len(context_model.morph_vocab)}")
    print(f"  pos_vocab_size: {len(context_model.pos_vocab)}")
    print(f"  weights: {[round(w, 4) for w in context_model.weights]}")


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical text compression pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="Compress input text")
    compress_parser.add_argument("input",  help="Input text file")
    compress_parser.add_argument("output", help="Output binary file")
    compress_parser.add_argument("--model", default=None, help="spaCy model to use.")

    decompress_parser = subparsers.add_parser("decompress", help="Decompress archive")
    decompress_parser.add_argument("input", help="Input binary file")

    analyse_parser = subparsers.add_parser("analyse", help="Analyse input text")
    analyse_parser.add_argument("input",  help="Input text file")
    analyse_parser.add_argument("--model", default=None)

    args = parser.parse_args()

    if args.command == "compress":
        text = _read_text(args.input)
        compress_to_file(text, args.output, model=args.model)
        print(f"Wrote LEXI archive to {args.output}")
    elif args.command == "decompress":
        reconstructed = decompress(args.input)
        print(reconstructed)
    elif args.command == "analyse":
        text = _read_text(args.input)
        analyse(text, model=args.model)


if __name__ == "__main__":
    main()
