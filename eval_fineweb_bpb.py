"""Evaluate bits-per-byte (bpb) of the Lexis pipeline on FineWeb samples.

Usage
-----
    python eval_fineweb_bpb.py --samples 50 --chars 10000 --out results.json

Flags
-----
--samples   Number of FineWeb documents to evaluate  (default: 50)
--chars     Max characters to take from each doc     (default: 5000, 0=all)
--split     FineWeb dataset split                    (default: train)
--seed      Random seed for shuffling                (default: 42)
--out       Optional path to write JSON results      (default: none)
--verbose   Print per-sample progress                (default: False)

What is measured
----------------
For each document the script runs compress_to_file() and measures TWO bpb values:

  bpb (char-stream) = (compressed_bitstream_bytes * 8) / original_utf8_bytes

    Where 'compressed_bitstream_bytes' is the size of the raw arithmetic-coded
    char-class bitstream only (stage 7 output). This isolates core character-level
    compression quality, comparable to other arithmetic coders.

  bpb (full payload) = (lexis_file_bytes * 8) / original_utf8_bytes

    Where 'lexis_file_bytes' is the size of the complete .lexis msgpack file on
    disk, including the char-stream bitstream, POS tag sequences, tree shape
    encodings, online context model state, pos-delta stream, and discourse symbol
    table. This is the honest end-to-end compression ratio.

The char-stream bpb and full-payload bpb are both reported in aggregate output
and per-sample JSON results.

A bpb of 8.0 means no compression.  English text typically compresses to
~1.5-2.5 bpb with a strong LM.  This pipeline targets ~3-5 bpb.

Model initialisation
--------------------
SpaCy (en_core_web_lg) and the HuggingFace coreference model are loaded
ONCE at startup and reused across all samples, avoiding the ~2-3s
cold-start penalty per document.
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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# One-time model initialisation
# ---------------------------------------------------------------------------

def _init_models(spacy_model: str = "en_core_web_lg"):
    """
    Load spaCy and the coreference model once.
    Returns (nlp, discourse_analyser).
    """
    print("[Init] Loading spaCy model...")
    import spacy
    gpu_fn = getattr(spacy, "prefer_gpu", None) or getattr(spacy, "require_gpu", None)
    if callable(gpu_fn):
        gpu_fn()
    nlp = spacy.load(spacy_model)
    from compression.config import SPACY_MAX_LENGTH
    nlp.max_length = SPACY_MAX_LENGTH
    print(f"[Init] spaCy loaded: {spacy_model}")

    print("[Init] Loading coreference model...")
    from compression.pipeline.stage4_discourse import DiscourseAnalyser
    analyser = DiscourseAnalyser(use_spacy=True, device="cpu")
    print("[Init] Coreference model loaded.")

    return nlp, analyser


# ---------------------------------------------------------------------------
# FineWeb loader
# ---------------------------------------------------------------------------

def _load_fineweb(n_samples: int, max_chars: int, split: str, seed: int) -> list[str]:
    """Stream n_samples documents from FineWeb, truncated to max_chars."""
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets", file=sys.stderr)
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
    reservoir: list[str] = []
    for i, row in enumerate(ds):
        text: str = row.get("text", "")
        if max_chars > 0:
            text = text[:max_chars]
        if len(text) < 100:
            continue
        if len(reservoir) < n_samples:
            reservoir.append(text)
        else:
            j = rng.randint(0, i)
            if j < n_samples:
                reservoir[j] = text
        if i >= n_samples * 20:
            break

    rng.shuffle(reservoir)
    print(f"[FineWeb] Loaded {len(reservoir)} documents.")
    return reservoir


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def _eval_one(
    text: str,
    idx: int,
    nlp,
    discourse_analyser,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline on one document and return bpb stats."""
    # Monkey-patch module-level singletons so pipeline reuses loaded models.
    import main as main_module
    import compression.pipeline.stage4_discourse as stage4_module
    import compression.pipeline.stage2_morphology as stage2_module

    # Patch _get_nlp to return the already-loaded nlp
    _orig_get_nlp = main_module._get_nlp
    main_module._get_nlp = lambda model=None: nlp

    # Patch DiscourseAnalyser.__init__ to reuse the already-loaded analyser
    _orig_run_discourse = main_module._run_discourse
    def _fast_run_discourse(text_in: str, device: str = "cpu"):
        from compression.pipeline.stage5_discourse_symbols import encode_symbols
        stage4_result = discourse_analyser.analyse_document(text_in)
        compressed, symbol_table = encode_symbols(text_in, stage4_result)
        return compressed, symbol_table
    main_module._run_discourse = _fast_run_discourse

    original_bytes = len(text.encode("utf-8"))

    with tempfile.NamedTemporaryFile(suffix=".lexis", delete=False) as f:
        tmp_path = f.name

    try:
        t0 = time.time()
        stats = main_module.compress_to_file(text, tmp_path)
        elapsed = time.time() - t0

        compressed_size = stats.get("compressed_size", 0)  # char-stream bitstream only
        full_payload_size = Path(tmp_path).stat().st_size   # complete .lexis file on disk

        bpb = (compressed_size * 8) / original_bytes if original_bytes else float("inf")
        full_payload_bpb = (full_payload_size * 8) / original_bytes if original_bytes else float("inf")

        return {
            "sample": idx,
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_size,
            "bpb": round(bpb, 4),
            "full_payload_bytes": full_payload_size,
            "full_payload_bpb": round(full_payload_bpb, 4),
            "compression_ratio": round(stats.get("compression_ratio", 0.0), 4),
            "discourse_symbols": stats.get("discourse_symbols", 0),
            "discourse_reduction_pct": stats.get("discourse_reduction_pct", 0.0),
            "elapsed_s": round(elapsed, 2),
        }
    except Exception as exc:
        import traceback
        return {
            "sample": idx,
            "original_bytes": original_bytes,
            "error": str(exc),
            "traceback": traceback.format_exc() if verbose else None,
            "bpb": None,
            "full_payload_bpb": None,
        }
    finally:
        Path(tmp_path).unlink(missing_ok=True)
        # Restore originals
        main_module._get_nlp = _orig_get_nlp
        main_module._run_discourse = _orig_run_discourse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Lexis pipeline bpb on FineWeb samples"
    )
    parser.add_argument("--samples", type=int, default=50,
                        help="Number of FineWeb documents to evaluate (default: 50)")
    parser.add_argument("--chars", type=int, default=5000,
                        help="Max chars per document, 0=unlimited (default: 5000)")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None,
                        help="Optional path to write JSON results")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-sample details and tracebacks on error")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_lg",
                        dest="spacy_model")
    args = parser.parse_args()

    # Load models once
    nlp, discourse_analyser = _init_models(args.spacy_model)

    # Load documents
    documents = _load_fineweb(args.samples, args.chars, args.split, args.seed)

    results: list[dict] = []
    t_total = time.time()

    for idx, doc in enumerate(documents):
        if args.verbose:
            print(f"\n[{idx+1}/{len(documents)}] Evaluating sample ({len(doc)} chars)...")
        else:
            print(f"  [{idx+1:3d}/{len(documents)}] ", end="", flush=True)

        result = _eval_one(doc, idx, nlp, discourse_analyser, verbose=args.verbose)
        results.append(result)

        if result.get("bpb") is not None:
            if args.verbose:
                print(
                    f"  bpb (char-stream): {result['bpb']:.4f}  "
                    f"bpb (full-payload): {result['full_payload_bpb']:.4f}  "
                    f"orig={result['original_bytes']}B  "
                    f"comp={result['compressed_bytes']}B  "
                    f"full={result['full_payload_bytes']}B  "
                    f"disc={result['discourse_symbols']}  "
                    f"t={result['elapsed_s']}s"
                )
            else:
                print(
                    f"bpb={result['bpb']:.4f}  "
                    f"full={result['full_payload_bpb']:.4f}  "
                    f"t={result['elapsed_s']}s"
                )
        else:
            print(f"ERROR: {result.get('error', 'unknown')}")

    total_elapsed = time.time() - t_total

    # Aggregate
    valid = [r for r in results if r.get("bpb") is not None]
    if valid:
        bpb_values = [r["bpb"] for r in valid]
        avg_bpb = sum(bpb_values) / len(bpb_values)
        variance = sum((b - avg_bpb) ** 2 for b in bpb_values) / len(bpb_values)
        std_bpb = variance ** 0.5
        min_bpb = min(bpb_values)
        max_bpb = max(bpb_values)

        fp_values = [r["full_payload_bpb"] for r in valid]
        avg_fp_bpb = sum(fp_values) / len(fp_values)
        fp_variance = sum((b - avg_fp_bpb) ** 2 for b in fp_values) / len(fp_values)
        std_fp_bpb = fp_variance ** 0.5
        min_fp_bpb = min(fp_values)
        max_fp_bpb = max(fp_values)

        total_orig = sum(r["original_bytes"] for r in valid)
        total_comp = sum(r["compressed_bytes"] for r in valid)
        total_full = sum(r["full_payload_bytes"] for r in valid)
        overall_bpb = (total_comp * 8) / total_orig if total_orig else float("inf")
        overall_fp_bpb = (total_full * 8) / total_orig if total_orig else float("inf")

        print("\n" + "=" * 65)
        print("AGGREGATE RESULTS")
        print("=" * 65)
        print(f"  Samples evaluated          : {len(valid)}/{len(results)}")
        print(f"  Avg bpb  (char-stream)     : {avg_bpb:.4f}  (±{std_bpb:.4f} std)")
        print(f"  Overall  (char-stream)     : {overall_bpb:.4f}  (pooled bytes)")
        print(f"  Min/Max  (char-stream)     : {min_bpb:.4f} / {max_bpb:.4f}")
        print(f"  Avg bpb  (full payload)    : {avg_fp_bpb:.4f}  (±{std_fp_bpb:.4f} std)")
        print(f"  Overall  (full payload)    : {overall_fp_bpb:.4f}  (pooled bytes)")
        print(f"  Min/Max  (full payload)    : {min_fp_bpb:.4f} / {max_fp_bpb:.4f}")
        print(f"  Total original             : {total_orig:,} bytes")
        print(f"  Total compressed (char)    : {total_comp:,} bytes")
        print(f"  Total compressed (full)    : {total_full:,} bytes")
        print(f"  Total elapsed              : {total_elapsed:.1f}s")
        print("=" * 65)

        summary = {
            "samples_evaluated": len(valid),
            "avg_bpb": round(avg_bpb, 4),
            "std_bpb": round(std_bpb, 4),
            "overall_bpb": round(overall_bpb, 4),
            "min_bpb": round(min_bpb, 4),
            "max_bpb": round(max_bpb, 4),
            "avg_full_payload_bpb": round(avg_fp_bpb, 4),
            "std_full_payload_bpb": round(std_fp_bpb, 4),
            "overall_full_payload_bpb": round(overall_fp_bpb, 4),
            "min_full_payload_bpb": round(min_fp_bpb, 4),
            "max_full_payload_bpb": round(max_fp_bpb, 4),
            "total_original_bytes": total_orig,
            "total_compressed_bytes": total_comp,
            "total_full_payload_bytes": total_full,
            "total_elapsed_s": round(total_elapsed, 1),
            "per_sample": results,
        }
    else:
        print("\nNo valid results obtained.")
        summary = {"error": "no valid results", "per_sample": results}

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"Results written to: {args.out}")


if __name__ == "__main__":
    main()
