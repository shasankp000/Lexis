"""Full pipeline round-trip test: compress → decompress → compare.

Usage:
    python test_round_trip_pipeline.py [--chars N] [--no-discourse]

Options:
    --chars N        Number of characters to test (default: full file)
    --no-discourse   Skip Stage 4+5 discourse encoding (baseline comparison)
"""

from __future__ import annotations

import argparse
import re
import sys

from main import compress_to_file, decompress
from compression.pipeline.stage1_normalize import normalize_text
from compression.pipeline.stage4_discourse import DiscourseAnalyser
from compression.pipeline.stage5_discourse_symbols import validate_round_trip


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def run_discourse_roundtrip(text: str) -> None:
    """Standalone Stage 4+5 round-trip check before full pipeline."""
    print("\n=== Stage 4+5 Discourse Round-Trip ===")
    analyser = DiscourseAnalyser()
    stage4_result = analyser.analyse_document(text)
    result = validate_round_trip(text, stage4_result)
    print(f"  Round-trip OK:     {result['round_trip_ok']}")
    print(f"  Original tokens:   {result['original_tokens']}")
    print(f"  Compressed tokens: {result['compressed_tokens']}")
    print(f"  Reduction:         {result['reduction_pct']}%")
    print(f"  Symbols used:      {result['symbols_used']}")
    print(f"  Symbol table:      {result['symbol_table']}")
    if not result['round_trip_ok']:
        i = result['first_mismatch_idx']
        print(f"  First mismatch at char {i}")
        sys.exit(1)


def run_full_roundtrip(text: str, label: str = "") -> None:
    """Full compress_to_file → decompress round-trip check."""
    print(f"\n=== Full Pipeline Round-Trip {label}===")
    compress_to_file(text, "moby_dick.bin")
    decompressed = decompress("moby_dick.bin")

    orig = _normalise(text)
    dec = _normalise(decompressed)

    i = next(
        (j for j in range(min(len(orig), len(dec))) if orig[j] != dec[j]),
        min(len(orig), len(dec)),
    )

    match = orig == dec
    print(f"  Match:             {match}")
    print(f"  Orig length:       {len(orig)}")
    print(f"  Dec length:        {len(dec)}")
    if not match:
        print(f"  First mismatch at: {i}")
        print(f"  ORIG: {repr(orig[max(0, i-40):i+80])}")
        print(f"  DEC:  {repr(dec[max(0, i-40):i+80])}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chars", type=int, default=None,
        help="Characters to test (default: full file)"
    )
    parser.add_argument(
        "--no-discourse", action="store_true",
        help="Skip Stage 4+5 standalone check"
    )
    args = parser.parse_args()

    raw = open("moby_dick.txt", encoding="utf-8").read()
    if args.chars:
        raw = raw[:args.chars]

    text = normalize_text(raw)
    print(f"Text length: {len(text)} chars, {len(text.split())} tokens")

    if not args.no_discourse:
        run_discourse_roundtrip(text)

    run_full_roundtrip(text)


if __name__ == "__main__":
    main()
