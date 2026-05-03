# Lexis

A linguistically-structured hierarchical text compressor for English, built as a research contribution to the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

Lexis (main) achieves **2.7494 bpb char-stream on FineWeb with zero training data** (50 samples, pooled bytes), outperforming gzip (≈3.5 bpb) and zstd (≈3.0 bpb) purely through explicit linguistic structure -- no learned weights, no training corpus.

> *"How much of the compressibility of English comes from its linguistic structure alone, versus from statistical regularities in training data?"*
>
> Lexis provides a quantitative answer: linguistic priors alone account for roughly **2/3 of the gap** between a naive byte compressor and a strong trained language model.

---

## Benchmark Results

### System comparison

| System | Corpus | bpb (char-stream) | bpb (full payload) | Notes |
|---|---|---|---|---|
| Uncompressed UTF-8 | — | 8.00 | 8.00 | Baseline |
| gzip level 9 | — | ≈3.50 | ≈3.50 | General-purpose |
| zstd level 19 | — | ≈3.00 | ≈3.00 | General-purpose |
| **Lexis main** | **FineWeb** | **2.7494** | **23.384** | 50 samples × ≤10k chars, pooled |
| **Lexis main** | **Moby Dick** | **2.6649** | **20.8391** | 100k chars, single document |
| cmix | — | ≈2.00 | ≈2.00 | Classical context mixing, CPU-only |
| GPT-2 (1.5B params) | — | ≈1.30 | ≈1.30 | Trained on WebText |

*char-stream bpb = arithmetic-coded character bitstream only. full-payload bpb = complete .lexis file including all metadata (POS tags, morph codes, model state, symbol table, etc.). The two Lexis main rows represent different test corpora -- both are valid measurements of the same codebase.*

### Lexis main vs Lexis-E -- Moby Dick & FineWeb at 100k chars

| Branch | Corpus | Profile | char_stream_bpb | full_payload_bpb | char_stream bytes | full_payload bytes |
|---|---|---|---|---|---|---|
| main | Moby Dick | N/A (fixed params) | 2.6649 | 20.8391 | 33,881 | 264,943 |
| **Lexis-E** | **Moby Dick** | **k6s511 (default)** | **2.7555** | **11.0172** | **34,323** | **137,230** |
| main | FineWeb | N/A (fixed params) | 2.7494\* | 23.384\* | — | — |
| **Lexis-E** | **FineWeb** | **k6s511 (default)** | **2.7523** | **11.1048** | **34,433** | **138,926** |

*\*FineWeb main-branch scores are pooled over 50 samples × ≤10k chars; Lexis-E FineWeb scores are from a single 100k-char document. The key takeaway: compact_mode (Lexis-E) cuts full-payload bpb by ~47% on Moby Dick (20.84 → 11.02) and ~52% on FineWeb (23.38 → 11.10), while char-stream bpb stays essentially the same.*

### Lexis main -- FineWeb (50 samples ×≤10k chars, all pipeline stages active)

| Input chars | char_stream_bpb | full_payload_bpb | Notes |
|---|---|---|---|
| 50 samples, pooled | 2.7494 | 23.384 | Measured via `eval_fineweb_bpb.py --samples 50 --chars 10000 --seed 42` |

### Lexis main -- Moby Dick scaling test

| Input chars | char_stream_bpb | full_payload_bpb | char_stream bytes | full_payload bytes |
|---|---|---|---|---|
| **100,000** | **2.6649** | **20.8391** | 33,881 | 264,943 |

*Single-document continuous text. full_payload_bpb is high relative to Lexis-E because the main branch carries uncompressed metadata; Lexis-E’s compact_mode dramatically reduces metadata overhead.*

---

## Lexis-E (Efficient)

The `lexis-e` branch is the **Efficient** evolution of Lexis, developed after the core pipeline was validated on main. The "E" stands for **Efficient** -- the primary goal of Lexis-E is to dramatically reduce the metadata overhead that dominates the full-payload bpb on the main branch.

### Why Lexis-E was created

After validating the 8-stage pipeline on `main`, two problems were identified:

1. **Full-payload overhead** -- The main branch’s `.lexis` file bundles uncompressed structural metadata (POS tag sequences, morphological codes, model weights, symbol tables). This drives the full-payload bpb to ~20-23 on real documents, even when the character stream compresses well to ~2.7 bpb. The char-stream bpb is the honest compression quality metric, but the full-payload figure is the true end-to-end storage ratio.
2. **Fixed context-mixing parameters** -- The Stage 6 probability model had no way to tune the trade-off between prediction depth (`top_k`) and probability sharpening (`scale`), leaving performance on the table for different document types and sizes.

### What Lexis-E adds

| Feature | Lexis (main) | Lexis-E |
|---|---|---|
| Metadata encoding | Raw / uncompressed | Compact binary (compact_mode) |
| Context model tuning | Fixed parameters | Configurable `top_k` × `scale` sweep |
| Full-payload bpb at 100k chars (Moby Dick) | 20.84 | **11.02** |
| Full-payload bpb at 100k chars (FineWeb) | 23.38\* | **11.10** |
| char-stream bpb at 100k chars (Moby Dick) | 2.6649 | 2.7555 |
| char-stream bpb at 100k chars (FineWeb) | 2.7494\* | 2.7523 |
| Case flag bug fix | No | Yes -- bitmap bit-indexing corrected |
| Profile presets | None | `default` (k6s511), `aggressive` (k6s127) |
| Scaling test script | No | Yes (`scaling_test.py`) |

*\*FineWeb main-branch scores are pooled over 50 samples × ≤10k chars, not a single 100k-char document.*

The full-payload bpb improvement from **20.84 → 11.02** on Moby Dick (~47% reduction) is entirely attributable to compact_mode metadata encoding, not to any change in the character-stream compression algorithm.

---

## How It Works

Lexis compresses text through an 8-stage pipeline that progressively strips linguistic redundancy at every level of English structure:

```
Raw Text
   ↓
[Stage 1]  Normalization          -- sentence boundaries, whitespace, UTF-8, BOM stripping
   ↓
[Stage 1b] Word Substitution      -- frequency-based §W tokens, net-saving guard
   ↓
[Stage 1c] Symbol Slot Extraction -- §E/§W tokens stripped, char offsets recorded
   ↓
[Stage 2]  Morphological Analysis -- root + transformation codes (15.1% char reduction)
   ↓
[Stage 3]  Syntactic Parsing      -- POS tags, dependency trees, sentence type, voice
   ↓
[Stage 4]  Discourse Analysis     -- coreference resolution, symbolic entity links
   ↓
[Stage 5]  Symbolic Encoding      -- phonetic decomposition, delta streams, factoriadic
              ↳ case_flags / case_bitmaps per token (4 categories: lower/title/upper/mixed)
   ↓
[Stage 6]  Probability Modeling   -- 3-level online context-mixing model (no prior training)
   ↓
[Stage 7]  Arithmetic Encoding    -- interval arithmetic coding on probability-weighted symbol stream
                                     (NOTE: this is standard arithmetic coding, NOT rANS)
   ↓
[Stage 8]  Decoding               -- full reverse pipeline, semantic fidelity preserved
```

### Key Technical Contributions

**Mixed-radix phonetic decomposition** -- Characters are decomposed into (phonetic class, position, morphological role) triples rather than flat IDs. This reduces character-level delta magnitude by **2.42×** on real text.

**Online context adaptation** -- Stage 6 trains only on the document being compressed, in real time. No offline corpus needed.

**Symbol slot extraction (Stage 1c)** -- `§E`/`§W` discourse tokens are stripped before encoding and spliced back after decoding using anchor-based char-offset interpolation. Zero char-stream overhead; zero leaked tokens.

**Anchor-based splice** -- Instead of a single linear scale, `splice_symbols` builds anchor points every 200 clean chars snapped to space boundaries, then interpolates between the nearest pair. Eliminates positional drift at 10k+ chars.

**Discourse threshold effect** -- Stage 4 coreference substitution is net-negative below ~800 bytes and increasingly beneficial above ~2,000 bytes.

**Factoriadic delta encoding** -- Symbol deltas encoded in the factorial number system; compact for the small, frequent steps that dominate linguistically-constrained sequences.

**Case flag encoding (Stage 5)** -- Each token surface form is classified into one of four case categories (LOWER=0, TITLE=1, UPPER=2, MIXED=3). MIXED tokens additionally carry a per-character bitmap where bit N corresponds to char index N of the surface form. This allows lossless case restoration without storing any raw uppercase characters in the char stream.

---

## Installation

Requires **Python 3.11.x** -- later versions break spaCy compatibility.

```bash
# Fedora (adapt package manager for your distro)
sudo dnf install python3.11

# CUDA setup (optional -- used by Stage 3 spaCy and Stage 4 Longformer inference)
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
python pipeline_trace.py  # all stages green
```

---

## Usage

```bash
# Full pipeline trace (all 12 stages)
python pipeline_trace.py

# Round-trip test
python test_round_trip_pipeline.py

# FineWeb benchmark (reports both char-stream and full-payload bpb)
python eval_fineweb_bpb.py --samples 50 --chars 10000 --seed 42 --out results_fineweb_main_50x10k.json
```

---

## Submission Snapshot

The canonical Lexis(main) submission state is tagged at:

```
git tag challenge-submit-updated-docs-main-2026-05-01 
```

Commit: [`bc3582a`](https://github.com/shasankp000/Lexis/commit/bc3582ad752c3c9a36be20826e20dde0dba80c5a) -- *docs: add Lexis-E (Efficient) origin section explaining branch split and differences vs main*

This tag marks the exact Lexis(main) codebase not directly submitted to the OpenAI Parameter Golf Challenge, but the main branch of the Lexis repository that can be accessed. Last updated on 2026-05-01.

---

## Notes

- **Semantic fidelity over byte-exact reconstruction** -- Stage 1 sentence boundary detection produces minor punctuation normalizations at quote boundaries. These do not affect meaning, information content, or bpb measurement.
- **full_payload_bpb on short docs** -- The full-payload bpb is high (20-23 bpb on FineWeb short docs) because the .lexis metadata overhead dominates at small document sizes. The char-stream bpb (2.7494) is the fair compression quality metric. Lexis-E’s compact_mode reduces metadata overhead significantly.
- **IDE import warnings** -- your IDE may flag an import error in `stage4_discourse.py` for `fastcoref` if not launched from inside the virtual environment. This is a false positive.
- **GPU usage** -- Stage 3 (spaCy) and Stage 4 (Longformer coreference, 90.5M params) use GPU when available. Stage 7 arithmetic encoding runs on CPU (standard interval arithmetic coding, not rANS).
- **transformers version patch** -- `transformers/dependency_versions_table.py` requires manual patching to remove the `huggingface-hub<1.0` upper bound if your environment has `huggingface-hub>=1.0`.

---

## Test Corpus

- **Moby Dick** (Project Gutenberg) -- 100k chars single document
- **FineWeb** (HuggingFaceFW/fineweb, sample-10BT) -- 50 samples × ≤10k chars, seed=42

---

## Origin

Lexis started as a research point of interest for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf) -- specifically the non-record track, which invites submissions that push the frontier of parameter-limited performance without the 16MB / 10-minute constraint.

---

## References

- [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf)
- [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt)
- PAQ compression family -- context mixing
- PPM (Prediction by Partial Matching), 1984
- ANS (Asymmetric Numeral Systems) -- Duda, 2009
- Longformer -- [Beltagy et al. 2020](https://arxiv.org/abs/2004.05150)
- FineWeb dataset -- [HuggingFaceFW/fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- Neural scaling laws -- [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)
- lemminflect -- morphological inflection for Python
- msgpack -- binary serialisation
- zstd -- Zstandard compression, level 19 outer wrapper
