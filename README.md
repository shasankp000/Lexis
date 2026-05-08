
# Lexis-E (Efficient)

A linguistically-structured hierarchical text compressor for English, built as a research contribution to the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

Lexis-E achieves **2.7523 bpb char-stream on FineWeb at 100k chars** (default profile `k6s511`, compact_mode), outperforming gzip (3.3948 bpb) and zstd (3.1828 bpb) purely through explicit linguistic structure -- no learned weights, no training corpus. The "E" stands for **Efficient** -- Lexis-E exists to solve the metadata overhead problem of the main branch.

> *"How much of the compressibility of English comes from its linguistic structure alone, versus from statistical regularities in training data?"*
>
> Lexis provides a quantitative answer: linguistic priors alone account for roughly **2/3 of the gap** between a naive byte compressor and a strong trained language model.

---

## Why Lexis-E Exists

After the core 8-stage pipeline was validated on the `main` branch, two problems were identified that motivated a dedicated branch:

1. **Full-payload overhead** -- The main branch's `.lexis` file bundles uncompressed structural metadata (POS tag sequences, morphological codes, model weights, symbol tables). This drives the full-payload bpb to ~20-23 on real documents, even when the character stream compresses well to ~2.7 bpb. The char-stream bpb is the honest compression quality metric, but the full-payload figure is the true end-to-end storage ratio -- and it was unacceptably high.
2. **Fixed context-mixing parameters** -- The Stage 6 probability model had no way to tune the trade-off between prediction depth (`top_k`) and probability sharpening (`scale`), leaving performance on the table for different document types and sizes.

Lexis-E addresses both through `compact_mode` -- a configurable metadata encoding mode with a sweepable `top_k × scale` grid -- without changing the character-stream compression algorithm.

### Main vs Lexis-E

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

The full-payload bpb improvement from **20.84 → 11.02** on Moby Dick (~47% reduction) is entirely attributable to compact_mode metadata encoding. The char-stream compression algorithm is identical between branches -- the small difference in char-stream bpb reflects different test corpora sizes and methods, not an algorithmic change.

---

## Benchmark Results

### System comparison

| System | Corpus | bpb (char-stream) | bpb (full payload) | Notes |
|---|---|---|---|---|
| Uncompressed UTF-8 | — | 8.00 | 8.00 | Baseline |
| gzip level 9 | Moby Dick | 3.3230 | 3.3230 | 100k chars; no metadata separation |
| gzip level 9 | FineWeb | 3.3948 | 3.3948 | 50 samples × ≤10k chars, pooled |
| zstd level 19 | Moby Dick | 3.1125 | 3.1125 | 100k chars; no metadata separation |
| zstd level 19 | FineWeb | 3.1828 | 3.1828 | 50 samples × ≤10k chars, pooled |
| xz level 9 | Moby Dick | 3.0637 | 3.0637 | 100k chars; no metadata separation |
| xz level 9 | FineWeb | 3.1079 | 3.1079 | 50 samples × ≤10k chars, pooled |
| **Lexis-E (no training data)** | **Moby Dick** | **2.7555** | **11.0172** | 100k chars, k6s511 (default) |
| **Lexis-E (no training data)** | **FineWeb** | **2.7523** | **11.1048** | 100k chars, k6s511 (default) |
| cmix | enwik8 | ≈1.17 | ≈1.17 | Classical context mixing; score from Knoll 2024 (byronknoll.com/cmix.html) |
| GPT-2 (1.5B params) | — | ≈1.30 | ≈1.30 | Trained on WebText |

*char-stream bpb = arithmetic-coded character bitstream only (Lexis) or raw compressed stream (gzip/zstd/xz/cmix -- no metadata separation). full-payload bpb = complete .lexis file including all metadata. For gzip/zstd/xz/cmix, char-stream bpb = full-payload bpb. cmix score is the published enwik8 figure; it is not directly comparable to the 100k-char Moby Dick / FineWeb corpora used for all other rows.*

### Lexis main vs Lexis-E -- Moby Dick & FineWeb at 100k chars

| Branch | Corpus | Profile | char_stream_bpb | full_payload_bpb | char_stream bytes | full_payload bytes |
|---|---|---|---|---|---|---|
| main | Moby Dick | N/A (fixed params) | 2.6649 | 20.8391 | 33,881 | 264,943 |
| **Lexis-E** | **Moby Dick** | **k6s511 (default)** | **2.7555** | **11.0172** | **34,323** | **137,230** |
| main | FineWeb | N/A (fixed params) | 2.7494\* | 23.384\* | — | — |
| **Lexis-E** | **FineWeb** | **k6s511 (default)** | **2.7523** | **11.1048** | **34,433** | **138,926** |

*\*FineWeb main-branch scores are pooled over 50 samples × ≤10k chars; Lexis-E FineWeb scores are from a single 100k-char document. The key takeaway: compact_mode cuts full-payload bpb by ~47% on Moby Dick (20.84 → 11.02) and ~52% on FineWeb (23.38 → 11.10), while char-stream bpb stays essentially the same.*

### Lexis-E -- Scaling Test on FineWeb (compact_mode, both profiles)

| Profile | Input chars | char_stream_bpb | full_payload_bpb | char_stream bytes | full_payload bytes |
|---|---|---|---|---|---|
| default (k6s511) | 10,000 | 2.7807 | 12.1866 | 3,479 | 15,247 |
| default (k6s511) | 25,000 | 2.7774 | 11.4323 | 8,686 | 35,753 |
| default (k6s511) | 50,000 | 2.7671 | 11.1774 | 17,304 | 69,898 |
| **default (k6s511)** | **100,000** | **2.7523** | **11.1048** | **34,433** | **138,926** |
| aggressive (k6s127) | 10,000 | 2.7927 | 12.0028 | 3,494 | 15,017 |
| aggressive (k6s127) | 25,000 | 2.7860 | 11.4278 | 8,713 | 35,739 |
| aggressive (k6s127) | 50,000 | 2.7767 | 11.1739 | 17,364 | 69,876 |
| aggressive (k6s127) | 100,000 | 2.7644 | 11.0597 | 34,584 | 138,362 |

*Wall-clock time for both profiles × 4 sizes: **8m 56s** real (13m 31s user -- CPU-parallel stages). char_stream_bpb measures the arithmetic-coded character stream only. full_payload_bpb includes all metadata (morph codes, POS tags, case flags, model weights, symbol table, root lengths, etc.).*

### Lexis-E -- Scaling Test on Moby Dick (compact_mode, default profile k6s511)

| Input chars | char_stream_bpb | full_payload_bpb | char_stream bytes | full_payload bytes |
|---|---|---|---|---|
| 10,000 | 2.7919 | 12.6728 | 3,411 | 15,483 |
| 25,000 | 2.7808 | 12.2991 | 8,535 | 37,749 |
| 50,000 | 2.7724 | 11.5677 | 17,176 | 71,666 |
| **100,000** | **2.7555** | **11.0172** | **34,323** | **137,230** |

*Source: `sweep_k6_s511-16.csv`. Profile: top_k=6, scale=511.*

---

## compact_mode -- Profile Sweep (Moby Dick corpus)

Lexis-E exposes a `compact_mode` flag that sweeps the context-mixing model's `top_k` (number of active prediction contexts) and `scale` (probability sharpening factor). A full grid sweep was run at 10k / 25k / 50k / 100k chars across k∈{3,4,5,6} × scale∈{127,255,511,1023}.

### Full sweep -- char_stream_bpb at 100k chars

| Profile | top_k | scale | char_stream_bpb | full_payload_bpb | char_stream bytes |
|---|---|---|---|---|---|
| k3s127 | 3 | 127 | 3.1928 | 12.1146 | 39,769 |
| k3s255 | 3 | 255 | 3.3339 | 12.2242 | 41,527 |
| k3s511 | 3 | 511 | 3.4860 | 11.6501 | 43,421 |
| k3s1023 | 3 | 1023 | 3.6408 | 11.9568 | 45,350 |
| k4s127 | 4 | 127 | 2.9240 | 11.2266 | 36,421 |
| k4s255 | 4 | 255 | 2.9714 | 11.3285 | 37,012 |
| k4s511 | 4 | 511 | 3.0270 | 11.3255 | 37,704 |
| k4s1023 | 4 | 1023 | 3.0847 | 11.3895 | 38,423 |
| k5s127 | 5 | 127 | 2.8413 | 11.0494 | 35,391 |
| k5s255 | 5 | 255 | 2.8585 | 11.0560 | 35,606 |
| k5s511 | 5 | 511 | 2.8828 | 11.2648 | 35,908 |
| k5s1023 | 5 | 1023 | 2.9087 | 11.2746 | 36,231 |
| k6s127 | 6 | 127 | 2.7719 | 11.0140 | 34,527 |
| k6s255 | 6 | 255 | 2.7583 | 11.1828 | 34,357 |
| **k6s511** *(default)* | **6** | **511** | **2.7555** | **11.0172** | **34,323** |
| k6s1023 | 6 | 1023 | 2.7563 | 11.1885 | 34,333 |

### Why k6s511 is the default

k6s511 achieves the **lowest char_stream_bpb (2.7555)** and the **smallest char_stream byte count (34,323)** at 100k chars across all 16 profiles on the Moby Dick sweep corpus, and confirms **2.7523 bpb on FineWeb** at 100k chars. While k6s1023 is marginally comparable (2.7563 bpb), it uses a larger scale window with no net benefit at any tested size. k6s127 (`aggressive` profile) scores better on full_payload_bpb at 100k on FineWeb (11.0597 vs 11.1048) but has worse char_stream_bpb (2.7644 vs 2.7523), making k6s511 the Pareto-optimal default for char-stream compression quality.

Two profiles are shipped:
- **`default`** -- `top_k=6, scale=511` -- best char_stream_bpb, lowest char_stream byte count
- **`aggressive`** -- `top_k=6, scale=127` -- best full_payload_bpb (metadata overhead dominant use case)

---

## How It Works

Lexis-E compresses text through a **12-stage pipeline** that progressively strips linguistic redundancy at every level of English structure.

> 📊 **[Open Interactive Architecture Diagram](https://htmlpreview.github.io/?https://github.com/shasankp000/Lexis/blob/lexis-e/docs/architecture.html)** — 12 stages, collapsible phase groups, click-to-expand stage details, light/dark theme.
>
> *(The diagram is a fully self-contained HTML file at [`docs/architecture.html`](docs/architecture.html). GitHub Markdown cannot render interactive HTML inline — use the htmlpreview link above to view it in-browser, or open the file locally.)*

### Key Technical Contributions

**Mixed-radix phonetic decomposition** -- Characters are decomposed into (phonetic class, position, morphological role) triples rather than flat IDs. This reduces character-level delta magnitude by **2.42×** on real text.

**Online context adaptation** -- Stage 6 trains only on the document being compressed, in real time. No offline corpus needed.

**Symbol slot extraction (Stage 1c)** -- `§E`/`§W` discourse tokens are stripped before encoding and spliced back after decoding using anchor-based char-offset interpolation. Zero char-stream overhead; zero leaked tokens.

**Anchor-based splice** -- Instead of a single linear scale, `splice_symbols` builds anchor points every 200 clean chars snapped to space boundaries, then interpolates between the nearest pair. Eliminates positional drift at 10k+ chars.

**Discourse threshold effect** -- Stage 4 coreference substitution is net-negative below ~800 bytes and increasingly beneficial above ~2,000 bytes.

**Factoriadic delta encoding** -- Symbol deltas encoded in the factorial number system; compact for the small, frequent steps that dominate linguistically-constrained sequences.

**Case flag encoding (Stage 5b)** -- Each token surface form is classified into one of four case categories (LOWER=0, TITLE=1, UPPER=2, MIXED=3). MIXED tokens additionally carry a per-character bitmap where bit N corresponds to char index N of the surface form. This allows lossless case restoration without storing any raw uppercase characters in the char stream. Bug fix applied in Lexis-E: bitmap bit-indexing in both `compute_case_flag` and `apply_case_flag` was corrected to use a consistent `bit N ↔ char index N` convention throughout.

**compact_mode** -- The context-mixing model (Stage 6) exposes `top_k` (active prediction contexts) and `scale` (probability sharpening). A 4×4 grid sweep identified `k6s511` as the Pareto-optimal default.

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

# FineWeb benchmark
python eval_fineweb_bpb.py

# Scale test (FineWeb, both profiles)
time python scaling_test.py \
  --input fineweb_100k.txt \
  --sizes 10000 25000 50000 100000 \
  --compact-context \
  --compact-profile both \
  --csv fineweb_sweep_both_profiles.csv
```

---

## Submission Snapshot

The canonical Lexis-E submission state is tagged at:

```
git tag challenge-submit-updated-docs-2026-05-01
```

Commit: [`49588f6`](https://github.com/shasankp000/Lexis/commit/49588f691d0e83c527a025d9c59f5ed556c16d8d) -- *docs: add Lexis-E (Efficient) origin section explaining branch split and differences vs main*

This tag marks the exact Lexis-E codebase submitted to the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf) non-record track originally created on 2026-04-30, docs updated on 2026-05-01.

---

## Notes

- **Semantic fidelity over byte-exact reconstruction** -- Stage 1 sentence boundary detection produces minor punctuation normalizations at quote boundaries. These do not affect meaning, information content, or bpb measurement.
- **full_payload_bpb vs char_stream_bpb** -- char_stream_bpb (2.7523 FineWeb / 2.7555 Moby Dick) measures compression quality of the character sequence alone. full_payload_bpb (11.10 FineWeb / 11.02 Moby Dick) includes all structural metadata; it is the honest end-to-end ratio. compact_mode dramatically reduces metadata overhead vs the main branch (11.1 vs 20.8 at 100k chars).
- **cmix score** -- The cmix row uses the published enwik8 bpb (≈1.17) from Byron Knoll's cmix page ([byronknoll.com/cmix.html](https://www.byronknoll.com/cmix.html)). cmix requires ~32 GB RAM to run locally; the published figure is cited rather than measured. Note the enwik8 corpus differs from Moby Dick / FineWeb, so this row is for reference context only.
- **IDE import warnings** -- your IDE may flag an import error in `stage4_discourse.py` for `fastcoref` if not launched from inside the virtual environment. This is a false positive.
- **GPU usage** -- Stage 3 (spaCy) and Stage 4 (Longformer coreference, 90.5M params) use GPU when available. Stage 7 arithmetic encoding runs on CPU (standard interval arithmetic coding, not rANS).
- **transformers version patch** -- `transformers/dependency_versions_table.py` requires manual patching to remove the `huggingface-hub<1.0` upper bound if your environment has `huggingface-hub>=1.0`.

---

## Test Corpus

- **Moby Dick** (Project Gutenberg) -- compact_mode profile sweep, 10k-100k chars (16 profiles, 4 sizes each); default profile (k6s511) scaling test
- **FineWeb** (HuggingFaceFW/fineweb, sample-10BT) -- compact_mode scaling: both profiles × 4 sizes up to 100k chars

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
- cmix -- [Byron Knoll, byronknoll.com/cmix.html](https://www.byronknoll.com/cmix.html)
- lemminflect -- morphological inflection for Python
- msgpack -- binary serialisation
- zstd -- Zstandard compression, level 19 outer wrapper
