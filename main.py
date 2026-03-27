"""Pipeline entry point and CLI wiring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict

try:
    from tqdm import tqdm  # type: ignore
except Exception:

    def tqdm(iterable, **kwargs):  # type: ignore
        return iterable


from compression.alphabet.morph_codes import apply_morph
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
from compression.pipeline.stage5_encode import CharacterEncoder, StructuralEncoder
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.utils import chunk_text


def _get_nlp(model: str | None = None):
    model_name = model or SPACY_MODEL
    try:
        import spacy

        # Enable GPU if available (robust across spaCy versions)
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
    ):  # ← ADD
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


def compress(text: str, output_path: str, model: str | None = None) -> Dict:
    """Run full available pipeline on text. Return stats."""
    normalized = normalize_text(text)
    analyser = MorphologicalAnalyser(use_spacy=True, model_name=model)
    morphology = analyser.analyse_sentence(normalized)

    encoder = CharacterEncoder()
    stats = encoder.stats(normalized)

    encoded_sentences, pos_freq_table = _encode_for_model(normalized, model=model)
    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)
    bpb_value = context_model.bpb(normalized, _EncodedPipeline(encoded_sentences))

    payload = {
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


def decompress(input_path: str) -> str:
    """Reconstruct text from compressed file."""
    payload = json.loads(Path(input_path).read_text())
    words = [
        apply_morph(entry["root"], entry["code"])
        for entry in payload.get("morphology", [])
    ]
    return " ".join(words)


def analyse(text: str, model: str | None = None) -> None:
    """Run pipeline in analysis mode — print stats at each stage."""
    normalized = normalize_text(text)
    analyser = MorphologicalAnalyser(use_spacy=True, model_name=model)
    morph_stats = analyser.char_savings(normalized)

    encoder = CharacterEncoder()
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
            "tag_count": sentence.get("pos_encoding", {}).get("tag_count", 0),
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
    print(f"  char_context_size: {CHAR_CONTEXT_SIZE}")
    print(f"  morph_context_size: {MORPH_CONTEXT_SIZE}")
    print(f"  struct_context_size: {STRUCT_CONTEXT_SIZE}")
    print(f"  weights: {[round(w, 4) for w in context_model.weights]}")


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hierarchical text compression pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    compress_parser = subparsers.add_parser("compress", help="Compress input text")
    compress_parser.add_argument("input", help="Input text file")
    compress_parser.add_argument("output", help="Output binary file (JSON for now)")
    compress_parser.add_argument(
        "--model",
        default=None,
        help="spaCy model to use. Overrides the default in config.py.",
    )

    decompress_parser = subparsers.add_parser("decompress", help="Decompress archive")
    decompress_parser.add_argument("input", help="Input binary file (JSON for now)")

    analyse_parser = subparsers.add_parser("analyse", help="Analyse input text")
    analyse_parser.add_argument("input", help="Input text file")
    analyse_parser.add_argument(
        "--model",
        default=None,
        help="spaCy model to use (e.g. en_core_web_lg, en_core_web_trf). "
        "Overrides the default in config.py.",
    )

    args = parser.parse_args()

    if args.command == "compress":
        text = _read_text(args.input)
        compress(text, args.output, model=args.model)
        print(f"Wrote compressed payload to {args.output}")
    elif args.command == "decompress":
        reconstructed = decompress(args.input)
        print(reconstructed)
    elif args.command == "analyse":
        text = _read_text(args.input)
        analyse(text, model=args.model)


if __name__ == "__main__":
    main()
