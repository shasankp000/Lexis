"""Pipeline entry point and CLI wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from compression.alphabet.morph_codes import apply_morph
from compression.alphabet.symbol_alphabet import SymbolAlphabet
from compression.config import SPACY_MAX_LENGTH, SPACY_MODEL
from compression.pipeline.stage1_normalize import normalize_text
from compression.pipeline.stage2_morphology import MorphologicalAnalyser
from compression.pipeline.stage3_syntax import analyse_sentence
from compression.pipeline.stage5_encode import CharacterEncoder, StructuralEncoder
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.utils import chunk_text


def _get_nlp():
    try:
        import spacy

        nlp = spacy.load(SPACY_MODEL)
        nlp.max_length = SPACY_MAX_LENGTH
        return nlp
    except Exception as exc:
        raise RuntimeError(f"spaCy model {SPACY_MODEL} not available") from exc


def _encode_for_model(text: str) -> list[dict]:
    nlp = _get_nlp()
    analyser = MorphologicalAnalyser(use_spacy=True)
    symbol_alphabet = SymbolAlphabet()
    structural_encoder = StructuralEncoder(symbol_alphabet)
    char_encoder = CharacterEncoder()

    encoded_sentences: list[dict] = []
    for chunk in chunk_text(text):
        doc = nlp(chunk)
        for sent in doc.sents:
            syntax = analyse_sentence(sent)
            morphology = analyser.analyse_sentence(sent.text)
            encoded_sentences.append(
                char_encoder.encode_sentence_full(
                    morphology, syntax, structural_encoder
                )
            )
    return encoded_sentences


def _summarise_pos_delta(results: list[dict]) -> dict:
    """Aggregate POS delta test results into a summary."""
    total = len(results)
    ratios = [float(r.get("improvement_ratio", 0.0)) for r in results]
    delta_helped = sum(1 for ratio in ratios if ratio > 1.0)
    delta_hurt = sum(1 for ratio in ratios if 0.0 < ratio < 1.0)
    neutral = total - delta_helped - delta_hurt

    bits_saved = sum(
        float(r.get("flat_cost", 0.0)) - float(r.get("delta_cost", 0.0))
        for r in results
        if float(r.get("improvement_ratio", 0.0)) > 1.0
    )
    bits_lost = sum(
        float(r.get("delta_cost", 0.0)) - float(r.get("flat_cost", 0.0))
        for r in results
        if 0.0 < float(r.get("improvement_ratio", 0.0)) < 1.0
    )

    used_delta = sum(1 for r in results if r.get("used_delta") is True)
    used_delta_pct = (used_delta / total * 100.0) if total else 0.0

    return {
        "total_sentences": total,
        "delta_helped": delta_helped,
        "delta_hurt": delta_hurt,
        "neutral": neutral,
        "delta_helped_pct": (delta_helped / total * 100.0) if total else 0.0,
        "used_delta": used_delta,
        "used_delta_pct": used_delta_pct,
        "net_bits_saved": bits_saved - bits_lost,
    }


class _EncodedPipeline:
    def __init__(self, encoded_sentences: list[dict]) -> None:
        self._encoded_sentences = encoded_sentences

    def encode_for_model(self, text: str) -> list[dict]:
        return self._encoded_sentences


def compress(text: str, output_path: str) -> Dict:
    """Run full available pipeline on text. Return stats."""
    normalized = normalize_text(text)
    analyser = MorphologicalAnalyser(use_spacy=True)
    morphology = analyser.analyse_sentence(normalized)

    encoder = CharacterEncoder()
    stats = encoder.stats(normalized)

    encoded_sentences = _encode_for_model(normalized)
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
            "weights": context_model.weights,
            "char_context_size": len(context_model.char_context),
            "morph_context_size": len(context_model.morph_context),
            "struct_context_size": len(context_model.struct_context),
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


def analyse(text: str) -> None:
    """Run pipeline in analysis mode — print stats at each stage."""
    normalized = normalize_text(text)
    analyser = MorphologicalAnalyser(use_spacy=True)
    morph_stats = analyser.char_savings(normalized)

    encoder = CharacterEncoder()
    encode_stats = encoder.stats(normalized)

    encoded_sentences = _encode_for_model(normalized)
    context_model = ContextMixingModel()
    context_model.train(encoded_sentences)
    bpb_value = context_model.bpb(normalized, _EncodedPipeline(encoded_sentences))

    print("Stage 2 — Morphology")
    for key, value in morph_stats.items():
        print(f"  {key}: {value}")

    print("Stage 5 — Character Encoding")
    for key, value in encode_stats.items():
        print(f"  {key}: {value}")

    pos_delta_results = [
        sentence.get("pos_delta_report", {})
        for sentence in encoded_sentences
        if isinstance(sentence.get("pos_delta_report", None), dict)
    ]
    pos_delta_summary = _summarise_pos_delta(pos_delta_results)

    print("Stage 5 — POS Delta Summary")
    for key, value in pos_delta_summary.items():
        print(f"  {key}: {value}")

    print("Stage 6 — Probability Model (Context Mixing)")
    print(f"  bpb: {bpb_value}")
    print(f"  char_vocab_size: {len(context_model.char_vocab)}")
    print(f"  morph_vocab_size: {len(context_model.morph_vocab)}")
    print(f"  pos_vocab_size: {len(context_model.pos_vocab)}")
    print(f"  char_context_size: {len(context_model.char_context)}")
    print(f"  morph_context_size: {len(context_model.morph_context)}")
    print(f"  struct_context_size: {len(context_model.struct_context)}")


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

    decompress_parser = subparsers.add_parser("decompress", help="Decompress archive")
    decompress_parser.add_argument("input", help="Input binary file (JSON for now)")

    analyse_parser = subparsers.add_parser("analyse", help="Analyse input text")
    analyse_parser.add_argument("input", help="Input text file")

    args = parser.parse_args()

    if args.command == "compress":
        text = _read_text(args.input)
        compress(text, args.output)
        print(f"Wrote compressed payload to {args.output}")
    elif args.command == "decompress":
        reconstructed = decompress(args.input)
        print(reconstructed)
    elif args.command == "analyse":
        text = _read_text(args.input)
        analyse(text)


if __name__ == "__main__":
    main()
