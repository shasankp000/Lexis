"""Evaluate bits-per-byte (bpb) of the Lexis pipeline on FineWeb samples.

Usage
-----
    python eval_fineweb_bpb.py --samples 10 --chars 5000

Flags
-----
--samples   Number of FineWeb documents to evaluate  (default: 10)
--chars     Max characters to take from each doc     (default: 5000, 0=all)
--split     FineWeb dataset split                    (default: train)
--seed      Random seed for shuffling                (default: 42)
--out       Optional path to write JSON results      (default: none)

What is measured
----------------
For each document the script runs compress_to_file() and measures:

  bpb = (compressed_bitstream_bytes * 8) / original_utf8_bytes

The 'compressed_bitstream_bytes' is the size of the raw arithmetic-coded
char-class bitstream only (stage 7 output), NOT the full msgpack payload
(which also stores the context model, POS tags, etc.).  This is the
fairest apples-to-apples comparison against other arithmetic coders.

A bpb of 8.0 means no compression.  English text typically compresses to
~1.5-2.5 bpb with a strong LM.  This pipeline targets ~3-5 bpb.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Silence spaCy / HuggingFace noise unless user wants verbose output
# ---------------------------------------------------------------------------
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)


def _load_fineweb(n_samples: int, max_chars: int, split: str, seed: int) -> list[str]:
    """Stream n_samples documents from FineWeb, truncated to max_chars."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' package not installed.  Run: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"[FineWeb] Streaming {n_samples} samples from split='{split}'...")
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split=split,
        streaming=True,
        trust_remote_code=True,
    )

    rng = random.Random(seed)
    # Reservoir-sample to get a random subset without loading everything
    reservoir: list[str] = []
    for i, row in enumerate(ds):
        text: str = row.get("text", "")
        if max_chars > 0:
            text = text[:max_chars]
        if len(text) < 100:   # skip very short docs
            continue
        if len(reservoir) < n_samples:
            reservoir.append(text)
        else:
            j = rng.randint(0, i)
            if j < n_samples:
                reservoir[j] = text
        if i >= n_samples * 20:  # look at 20x samples to get a good spread
            break

    rng.shuffle(reservoir)
    print(f"[FineWeb] Loaded {len(reservoir)} documents.")
    return reservoir


def _eval_one(text: str, idx: int) -> dict[str, Any]:
    """Run the full pipeline on one document and return bpb stats."""
    from main import compress_to_file

    original_bytes = len(text.encode("utf-8"))

    with tempfile.NamedTemporaryFile(suffix=".lexis", delete=False) as f:
        tmp_path = f.name

    try:
        t0 = time.time()
        stats = compress_to_file(text, tmp_path)
        elapsed = time.time() - t0

        compressed_size = stats.get("compressed_size", 0)   # arithmetic bitstream bytes
        bpb = (compressed_size * 8) / original_bytes if original_bytes else float("inf")

        return {
            "sample": idx,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_size,
            "bpb": round(bpb, 4),
            "compression_ratio": round(stats.get("compression_ratio", 0.0), 4),
            "discourse_symbols": stats.get("discourse_symbols", 0),
            "discourse_reduction_pct": stats.get("discourse_reduction_pct", 0.0),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as exc:
        return {
            "sample": idx,
            "original_bytes": original_bytes,
            "error": str(exc),
            "bpb": None,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Lexis pipeline bpb on FineWeb samples"
    )
    parser.add_argument("--samples", type=int, default=10,
                        help="Number of FineWeb documents to evaluate (default: 10)")
    parser.add_argument("--chars", type=int, default=5000,
                        help="Max chars per document, 0=unlimited (default: 5000)")
    parser.add_argument("--split", type=str, default="train",
                        help="FineWeb dataset split (default: train)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write JSON results")
    args = parser.parse_args()

    documents = _load_fineweb(args.samples, args.chars, args.split, args.seed)

    results: list[dict] = []
    for idx, doc in enumerate(documents):
        print(f"\n[{idx+1}/{len(documents)}] Evaluating sample ({len(doc)} chars)...")
        result = _eval_one(doc, idx)
        results.append(result)
        if result.get("bpb") is not None:
            print(
                f"  bpb:               {result['bpb']:.4f}"
                f"\n  original_bytes:    {result['original_bytes']}"
                f"\n  compressed_bytes:  {result['compressed_bytes']}"
                f"\n  discourse_symbols: {result['discourse_symbols']}"
                f"\n  elapsed:           {result['elapsed_s']}s"
            )
        else:
            print(f"  ERROR: {result.get('error')}")

    # Aggregate
    valid = [r for r in results if r.get("bpb") is not None]
    if valid:
        avg_bpb = sum(r["bpb"] for r in valid) / len(valid)
        min_bpb = min(r["bpb"] for r in valid)
        max_bpb = max(r["bpb"] for r in valid)
        total_orig = sum(r["original_bytes"] for r in valid)
        total_comp = sum(r["compressed_bytes"] for r in valid)
        overall_bpb = (total_comp * 8) / total_orig if total_orig else float("inf")

        print("\n" + "=" * 50)
        print("AGGREGATE RESULTS")
        print("=" * 50)
        print(f"  Samples evaluated:  {len(valid)}/{len(results)}")
        print(f"  Avg bpb:            {avg_bpb:.4f}")
        print(f"  Overall bpb:        {overall_bpb:.4f}  (total bytes)")
        print(f"  Min bpb:            {min_bpb:.4f}")
        print(f"  Max bpb:            {max_bpb:.4f}")
        print(f"  Total original:     {total_orig:,} bytes")
        print(f"  Total compressed:   {total_comp:,} bytes")
        print("=" * 50)

        summary = {
            "samples_evaluated": len(valid),
            "avg_bpb": round(avg_bpb, 4),
            "overall_bpb": round(overall_bpb, 4),
            "min_bpb": round(min_bpb, 4),
            "max_bpb": round(max_bpb, 4),
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "per_sample": results,
        }
    else:
        print("\nNo valid results obtained.")
        summary = {"error": "no valid results", "per_sample": results}

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nResults written to: {args.out}")


if __name__ == "__main__":
    main()
