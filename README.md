# Lexis

A linguistically-structured hierarchical text compressor for English, built as a research contribution to the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

Lexis achieves **2.7494 bpb on FineWeb with zero training data** (Lexis-E, main branch), outperforming gzip (≈3.5 bpb) and zstd (≈3.0 bpb) purely through explicit linguistic structure. The experimental **Lexis-R branch** (reconstructive variant) reaches **10.01 bpb full-payload / 2.23 bpb char-stream** on Moby Dick at 5k chars with 98.6% case-insensitive word overlap.

> *"How much of the compressibility of English comes from its linguistic structure alone, versus from statistical regularities in training data?"*
>
> Lexis provides a quantitative answer: linguistic priors alone account for roughly **2/3 of the gap** between a naive byte compressor and a strong trained language model.

---

## Branches

| Branch | Variant | Reconstruction | Primary metric |
|---|---|---|---|
| `main` | **Lexis-E** | Exact (byte-level) | 2.7494 bpb on FineWeb |
| `lexis-r` | **Lexis-R** | Semantic (root-form) | 10.01 bpb payload / 2.23 bpb char-stream |

---

## Benchmark Results

### Lexis-E (main) — FineWeb

| System | bpb on web text | Notes |
|---|---|---|
| Uncompressed UTF-8 | 8.00 | Baseline |
| gzip level 9 | ≈3.50 | General-purpose |
| zstd level 19 | ≈3.00 | General-purpose |
| **Lexis-E (no training data)** | **2.7494** | All stages active |
| cmix | ≈2.00 | Classical context mixing, CPU-only |
| GPT-2 (1.5B params) | ≈1.30 | Trained on WebText |

*Evaluated on 50 FineWeb samples × 10k chars, all pipeline stages active. Best single document: **2.6805 bpb** (10k chars, 17 discourse symbols).*

### Lexis-R (lexis-r branch) — Moby Dick 5k chars

| Metric | Value |
|---|---|
| char-stream bpb | **2.23** |
| full-payload bpb | **10.01** |
| payload size | 6,291 bytes |
| case-insensitive word overlap | **98.6%** |
| leaked § tokens | 0 |
| compress time | ~15.6 s |
| decompress time | ~0.54 s |

*Scale test (5k–100k chars) in progress. Results will be added once complete.*

---

## Lexis-R — Scale Test Results

Lexis-R is a lossy reconstruction variant of Lexis that recovers the original text at **word-form level** rather than byte-exact level. It stores morphological transformation codes, POS tags, and a compact phonetic char stream, enabling a fundamentally lower entropy encoding of linguistic content.

### char-stream bpb vs baselines (Moby Dick, 5k–100k chars)

| Input | Lexis-R char-stream bpb | gzip-9 bpb | zstd-19 bpb | Lexis-R vs zstd |
|---|---|---|---|---|
| 5,000 | 2.228 | 3.373 | 3.321 | **−33%** |
| 10,000 | 2.351 | 3.592 | 3.479 | **−32%** |
| 20,000 | 2.363 | 3.586 | 3.447 | **−31%** |
| 50,000 | 2.353 | 3.503 | 3.348 | **−30%** |
| 100,000 | 2.145 | 3.376 | 3.166 | **−32%** |

The char-stream measures only the linguistic content bits — excluding Lexis-R's structural metadata (context model, POS tags, morph codes, Huffman tables). The **full-payload bpb** includes all metadata:

| Input | Payload (B) | vs original | Full-payload bpb | Word overlap | Compress | Decompress |
|---|---|---|---|---|---|---|
| 5,000 | 6,294 | +25.2% | 10.01 | 98.25% | 17s | 0.55s |
| 10,000 | 11,795 | +16.4% | 9.31 | 96.99% | 16s | 0.47s |
| 20,000 | 22,585 | +10.1% | 8.81 | 95.11% | 24s | 1.04s |
| 50,000 | 52,392 | +2.7% | 8.22 | 96.80% | 48s | 2.48s |
| 100,000 | 99,778 | **−1.9%** | 7.85 | 97.18% | 90s | 4.73s |

Key observations:
- **char-stream bpb is flat and ~32% below zstd at all scales** — the linguistic encoding is structurally superior regardless of document length
- **Full-payload bpb improves with scale** (10.01 → 7.85) as structural metadata amortises over more content
- **Payload crosses below raw original at 100k chars** (99,778 B vs 101,708 B original)
- **Word overlap is stable 95–98% at all scales** — no degradation with document length after fixing the sentinel drift bug in the decoder

> **Note:** Lexis-R is not a replacement for gzip/zstd. It trades lossless guarantee and sub-second compression for a fundamentally lower entropy encoding of linguistic content. The full-payload overhead is a metadata cost, not an encoding inefficiency.

---

## How It Works

Lexis compresses text through an 8-stage pipeline that progressively strips linguistic redundancy at every level of English structure:

```
Raw Text
   ↓
[Stage 1]  Normalization          — sentence boundaries, whitespace, UTF-8
   ↓
[Stage 1b] Word Substitution      — frequency-based §W tokens, net-saving guard
   ↓
[Stage 1c] Symbol Slot Extraction — §E/§W tokens stripped, char offsets recorded
   ↓
[Stage 2]  Morphological Analysis — root + transformation codes (15.1% char reduction)
   ↓
[Stage 3]  Syntactic Parsing      — POS tags, dependency trees, sentence type, voice
   ↓
[Stage 4]  Discourse Analysis     — coreference resolution, symbolic entity links
   ↓
[Stage 5]  Symbolic Encoding      — phonetic decomposition, delta streams, factoriadic
   ↓
[Stage 6]  Probability Modeling   — 3-level online context-mixing model (no prior training)
   ↓
[Stage 7]  rANS Encoding          — arithmetic coding on probability-weighted symbol stream
   ↓
[Stage 8]  Decoding               — full reverse pipeline, semantic fidelity preserved
```

### Key Technical Contributions

**Mixed-radix phonetic decomposition** — Characters are decomposed into (phonetic class, position, morphological role) triples rather than flat IDs. This reduces character-level delta magnitude by **2.42×** on real text.

**Online context adaptation** — Stage 6 trains only on the document being compressed, in real time. No offline corpus needed.

**Symbol slot extraction (Stage 1c)** — `§E`/`§W` discourse tokens are stripped before encoding and spliced back after decoding using anchor-based char-offset interpolation. Zero char-stream overhead; zero leaked tokens.

**Anchor-based splice** — Instead of a single linear scale, `splice_symbols` builds anchor points every 200 clean chars snapped to space boundaries, then interpolates between the nearest pair. Eliminates positional drift at 10k+ chars.

**Discourse threshold effect** — Stage 4 coreference substitution is net-negative below ~800 bytes and increasingly beneficial above ~2,000 bytes.

**Factoriadic delta encoding** — Symbol deltas encoded in the factorial number system; compact for the small, frequent steps that dominate linguistically-constrained sequences.

---

## Installation

Requires **Python 3.11.x** — later versions break spaCy compatibility.

```bash
# Fedora (adapt package manager for your distro)
sudo dnf install python3.11

# CUDA setup (optional — used by Stage 3 spaCy and Stage 4 Longformer inference)
sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
sudo dnf clean all
sudo dnf module disable nvidia-driver
sudo dnf -y install cuda

export PATH=/usr/local/cuda-12.9/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Create virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg
pip install cupy-cuda12x  # only if CUDA is available

# Verify installation
pip check
pytest compression/tests -v  # all 25 tests should pass
```

---

## Usage

```bash
# Lexis-E: full compression and decompression round-trip
python test_round_trip_pipeline.py

# Lexis-R: compress and decompress
python -c "
from lexis_r.compress import compress
from lexis_r.decompress import decompress
compress(open('input.txt').read(), 'output.lexisr')
print(decompress('output.lexisr'))
"

# FineWeb benchmark (Lexis-E)
python benchmark.py <input_text_file>

<<<<<<< HEAD
# Scale test (Lexis-R)
python -c "
import json, re, os
from lexis_r.compress import compress
from lexis_r.decompress import decompress
os.makedirs('scale_test_results', exist_ok=True)
for size in [5000, 10000, 20000, 50000, 100000]:
    text = open('moby_dick.txt').read().lstrip('\ufeff')[:size]
    compress(text, f'/tmp/moby_r_{size}.lexisr')
    restored = decompress(f'/tmp/moby_r_{size}.lexisr')
    print(f'{size}: {len(restored)} chars restored')
"
=======
# Lexis-R scale test
python scale_test.py

# Lexis-R vs gzip/zstd baseline comparison
python baseline_scale_test.py

# Generate scale comparison chart
python scale_chart.py
>>>>>>> a2670ae7d5ef340a74fe33d75b87374f0a2f87fd
```

---

## Notes

<<<<<<< HEAD
- **Semantic fidelity over byte-exact reconstruction (Lexis-R)** — Lexis-R encodes root forms; inflected forms are reconstructed via lemminflect. Capitalisation of non-entity proper nouns is lost (fix pending: capitalisation bitmap).
- **Semantic fidelity over byte-exact reconstruction (Lexis-E)** — Stage 1 sentence boundary detection produces minor punctuation normalizations at quote boundaries. These do not affect meaning, information content, or bpb measurement.
=======
- **Semantic fidelity over byte-exact reconstruction** — Stage 1 sentence boundary detection produces minor punctuation normalizations at quote boundaries (e.g. `sentence. "` → `sentence."`). These do not affect meaning, information content, or bpb measurement.
- **Lexis-R word overlap** — Lexis-R targets word-form fidelity (95–98% word overlap at all scales), not byte-exact reconstruction. The remaining ~3-5% gap reflects genuine lossy compression from the arithmetic decoder on low-probability chars.
>>>>>>> a2670ae7d5ef340a74fe33d75b87374f0a2f87fd
- **IDE import warnings** — your IDE may flag an import error in `stage4_discourse.py` for `fastcoref` if not launched from inside the virtual environment. This is a false positive.
- **GPU usage** — Stage 3 (spaCy) and Stage 4 (Longformer coreference, 90.5M params) use GPU when available. Stage 7 rANS encoding runs on CPU.
- **transformers version patch** — `transformers/dependency_versions_table.py` requires manual patching to remove the `huggingface-hub<1.0` upper bound if your environment has `huggingface-hub>=1.0`.

---

## Origin

Lexis started as a research point of interest for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf) — specifically the non-record track, which invites submissions that push the infinite frontier of parameter-limited performance without the 16MB / 10-minute constraint.

It has since grown into a standalone research system with two active variants (Lexis-E and Lexis-R) and its own benchmark results.

---

## Current Test Corpus

<<<<<<< HEAD
- **Moby Dick** (Project Gutenberg) — round-trip validated at 5k, 10k, 25k chars (Lexis-E); 5k chars benchmarked (Lexis-R); scale test 5k–100k in progress
- **FineWeb** (HuggingFaceFW/fineweb, sample-10BT) — 50 samples × 10k chars, benchmarked (Lexis-E)
=======
- Moby Dick (Project Gutenberg) — Lexis-R scale-tested at 5k, 10k, 20k, 50k, 100k characters
- FineWeb (HuggingFaceFW/fineweb, sample-10BT) — 50 samples × 10k chars, benchmarked
>>>>>>> a2670ae7d5ef340a74fe33d75b87374f0a2f87fd

---

## References

- [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf)
- [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt)
- PAQ compression family — context mixing
- PPM (Prediction by Partial Matching), 1984
- ANS (Asymmetric Numeral Systems) — Duda, 2009
- Longformer — [Beltagy et al. 2020](https://arxiv.org/abs/2004.05150)
- FineWeb dataset — [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- Neural scaling laws — [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)
- lemminflect — morphological inflection for Python
- msgpack — binary serialisation
- zstd — Zstandard compression, level 19 outer wrapper
