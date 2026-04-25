"""Run end-to-end scaling tests from small to large character budgets."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from tempfile import TemporaryDirectory

from compression.metadata_codec import decode_metadata, is_lexi_file
from compression.pipeline.stage7_arithmetic import ArithmeticDecoder
from compression.pipeline.stage1_normalize import normalize_text
import main as lexis_main

_build_context_model = lexis_main._build_context_model
_build_encoded_sentences_from_metadata = lexis_main._build_encoded_sentences_from_metadata
_reconstruct_chars = lexis_main._reconstruct_chars
_split_roots = lexis_main._split_roots
compress_to_file = lexis_main.compress_to_file
decompress = lexis_main.decompress
_maybe_unwrap_zstd = getattr(lexis_main, "_maybe_unwrap_zstd", lambda payload: payload)


def _word_overlap_ratio(expected: str, actual: str) -> float:
    exp_words = expected.split()
    act_words = actual.split()
    denom = max(len(exp_words), len(act_words), 1)
    matches = sum(1 for a, b in zip(exp_words, act_words) if a == b)
    return matches / denom


def _word_bag_overlap_ratio(expected: str, actual: str) -> float:
    """Order-insensitive multiset overlap over tokenized words."""
    exp_counts = Counter(expected.split())
    act_counts = Counter(actual.split())
    overlap = sum(min(exp_counts[w], act_counts[w]) for w in exp_counts.keys() | act_counts.keys())
    denom = max(sum(exp_counts.values()), sum(act_counts.values()), 1)
    return overlap / denom


def _first_diff_index(expected: str, actual: str) -> int:
    for idx, (a, b) in enumerate(zip(expected, actual)):
        if a != b:
            return idx
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return -1


def run_scaling_test(
    input_path: str,
    sizes: list[int],
    model: str | None = None,
) -> list[dict[str, float | int | bool]]:
    source_text = Path(input_path).read_text(encoding="utf-8")
    rows: list[dict[str, float | int | bool]] = []

    with TemporaryDirectory(prefix="lexis_scale_") as tmpdir:
        tmp = Path(tmpdir)
        for size in sizes:
            raw_slice = source_text[:size]
            normalized = normalize_text(raw_slice)
            norm_bytes = len(normalized.encode("utf-8")) or 1

            lexi_path = tmp / f"sample_{size}.lexi"
            stats = compress_to_file(raw_slice, str(lexi_path), model=model)
            decoded = decompress(str(lexi_path))

            raw_payload = _maybe_unwrap_zstd(lexi_path.read_bytes())
            roots_count = -1
            morph_token_count = -1
            root_morph_aligned = False
            if is_lexi_file(raw_payload):
                payload = decode_metadata(raw_payload)
                morph_token_count = sum(len(s) for s in payload.get("morph_codes", []))
                context_model = _build_context_model(payload)
                encoded_sentences = _build_encoded_sentences_from_metadata(payload)
                decoder = ArithmeticDecoder()
                num_symbols = int(payload.get("num_symbols", 0))
                char_classes = decoder.decode(
                    payload["compressed_bitstream"],
                    context_model,
                    encoded_sentences,
                    num_symbols,
                )
                pos_deltas = ArithmeticDecoder().decode_unigram_counts(
                    bytes(payload.get("pos_deltas_bitstream", b"")),
                    {int(k): int(v) for k, v in payload.get("pos_deltas_counts", {}).items()},
                    int(payload.get("pos_deltas_count", 0)),
                )
                sentence_counts = [int(c) for c in payload.get("sentence_char_counts", [])]
                char_stream = _reconstruct_chars(char_classes, pos_deltas, sentence_counts)
                roots_count = len(_split_roots(char_stream))
                root_morph_aligned = roots_count == morph_token_count

            char_stream_bpb = (int(stats["compressed_size"]) * 8) / norm_bytes
            full_payload_bpb = (lexi_path.stat().st_size * 8) / norm_bytes
            overlap = _word_overlap_ratio(normalized, decoded)
            bag_overlap = _word_bag_overlap_ratio(normalized, decoded)
            seq_ratio = SequenceMatcher(None, normalized, decoded).ratio()
            first_diff = _first_diff_index(normalized, decoded)

            rows.append(
                {
                    "chars": size,
                    "normalized_chars": len(normalized),
                    "exact_match": decoded == normalized,
                    "char_stream_bpb": round(char_stream_bpb, 4),
                    "full_payload_bpb": round(full_payload_bpb, 4),
                    "word_overlap_positional": round(overlap, 4),
                    "word_overlap_bag": round(bag_overlap, 4),
                    "char_seq_ratio": round(seq_ratio, 4),
                    "first_diff_idx": first_diff,
                    "roots_count": roots_count,
                    "morph_tokens": morph_token_count,
                    "root_morph_aligned": root_morph_aligned,
                    "char_stream_bytes": int(stats["compressed_size"]),
                    "full_payload_bytes": lexi_path.stat().st_size,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Lexis scaling test runner")
    parser.add_argument("--input", default="moby_dick.txt")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[500, 1000, 2500, 5000, 10_000, 25_000, 50_000, 100_000],
    )
    parser.add_argument("--model", default=None)
    parser.add_argument("--csv", default=None, help="Optional CSV output path")
    args = parser.parse_args()

    rows = run_scaling_test(args.input, args.sizes, model=args.model)

    print(
        "chars | norm_chars | exact | char_stream_bpb | full_payload_bpb | "
        "word_pos | word_bag | char_seq | first_diff | roots/morph | aligned | "
        "char_bytes | payload_bytes"
    )
    for r in rows:
        print(
            f"{r['chars']:>5} | {r['normalized_chars']:>10} | "
            f"{str(r['exact_match']):>5} | {r['char_stream_bpb']:>15} | "
            f"{r['full_payload_bpb']:>16} | {r['word_overlap_positional']:>8} | "
            f"{r['word_overlap_bag']:>8} | {r['char_seq_ratio']:>8} | "
            f"{r['first_diff_idx']:>10} | "
            f"{r['roots_count']:>5}/{r['morph_tokens']:<5} | "
            f"{str(r['root_morph_aligned']):>7} | "
            f"{r['char_stream_bytes']:>10} | {r['full_payload_bytes']:>13}"
        )

    if args.csv:
        fieldnames = list(rows[0].keys()) if rows else []
        with Path(args.csv).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote CSV: {args.csv}")


if __name__ == "__main__":
    main()

