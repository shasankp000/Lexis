# Lexis

A linguistically-structured hierarchical text compressor for English, built as a research contribution to the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf).

Lexis achieves **2.7494 bpb on FineWeb with zero training data** (Lexis-E), outperforming gzip (≈3.5 bpb) and zstd (≈3.0 bpb) purely through explicit linguistic structure.

> *"How much of the compressibility of English comes from its linguistic structure alone, versus from statistical regularities in training data?"*
>
> Lexis provides a quantitative answer: linguistic priors alone account for roughly **2/3 of the gap** between a naive byte compressor and a strong trained language model.

---

## Benchmark Results

### Lexis-E — FineWeb

| System | bpb on web text | Notes |
|---|---|---|
| Uncompressed UTF-8 | 8.00 | Baseline |
| gzip level 9 | ≈3.50 | General-purpose |
| zstd level 19 | ≈3.00 | General-purpose |
| **Lexis-E (no training data)** | **2.7494** | All stages active |
| cmix | ≈2.00 | Classical context mixing, CPU-only |
| GPT-2 (1.5B params) | ≈1.30 | Trained on WebText |

*Evaluated on 50 FineWeb samples × 10k chars, all pipeline stages active. Best single document: **2.6805 bpb** (10k chars, 17 discourse symbols).*

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
# Full compression and decompression round-trip
python test_round_trip_pipeline.py

# FineWeb benchmark
python benchmark.py <input_text_file>
```

---

## Notes

- **Semantic fidelity over byte-exact reconstruction** — Stage 1 sentence boundary detection produces minor punctuation normalizations at quote boundaries. These do not affect meaning, information content, or bpb measurement.
- **IDE import warnings** — your IDE may flag an import error in `stage4_discourse.py` for `fastcoref` if not launched from inside the virtual environment. This is a false positive.
- **GPU usage** — Stage 3 (spaCy) and Stage 4 (Longformer coreference, 90.5M params) use GPU when available. Stage 7 rANS encoding runs on CPU.
- **transformers version patch** — `transformers/dependency_versions_table.py` requires manual patching to remove the `huggingface-hub<1.0` upper bound if your environment has `huggingface-hub>=1.0`.

---

## Origin

Lexis started as a research point of interest for the [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf) — specifically the non-record track, which invites submissions that push the infinite frontier of parameter-limited performance without the 16MB / 10-minute constraint.

---

## Current Test Corpus

- **Moby Dick** (Project Gutenberg) — round-trip validated at 5k, 10k, 25k chars (Lexis-E)
- **FineWeb** (HuggingFaceFW/fineweb, sample-10BT) — 50 samples × 10k chars, benchmarked (Lexis-E)

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
