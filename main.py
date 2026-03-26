"""Pipeline entry point and CLI wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from compression.alphabet.morph_codes import apply_morph
from compression.alphabet.symbol_alphabet import SymbolAlphabet
from compression.pipeline.stage1_normalize import normalize_text
from compression.pipeline.stage2_morphology import MorphologicalAnalyser
from compression.pipeline.stage3_syntax import analyse_sentence
from compression.pipeline.stage5_encode import CharacterEncoder, StructuralEncoder
from compression.pipeline.stage6_probability import ContextMixingModel


def _get_nlp():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception as exc:
        raise RuntimeError("spaCy model en_core_web_sm not available") from exc


def _encode_for_model(text: str) -> list[dict]:
    nlp = _get_nlp()
    analyser = MorphologicalAnalyser(use_spacy=True)
    symbol_alphabet = SymbolAlphabet()
    structural_encoder = StructuralEncoder(symbol_alphabet)
    char_encoder = CharacterEncoder()

    encoded_sentences: list[dict] = []
    doc = nlp(text)
    for sent in doc.sents:
        syntax = analyse_sentence(sent)
        morphology = analyser.analyse_sentence(sent.text)
        encoded_sentences.append(
            char_encoder.encode_sentence_full(morphology, syntax, structural_encoder)
        )
    return encoded_sentences


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
