# GPU-Accelerated Hierarchical Text Compression
## Research Discussion Document
**Status:** Work in Progress  
**Started:** March 2026  
**Last Updated:** March 28, 2026

---

## 1. Origin — Parameter Golf Challenge

The discussion started from OpenAI's [Parameter Golf Challenge](https://github.com/openai/parameter-golf):

- Train the best language model that fits in **16MB**
- Must run in under **10 minutes on 8xH100s**
- Scored by **bits per byte (bpb)** on FineWeb validation text — lower is better
- Current leaderboard best: ~**1.12 bpb**
- Challenge runs March 18 – April 30, 2026

### What the challenge actually asks

The model must predict what comes next in text as accurately as possible. Good prediction = good compression. The score (bpb) literally measures compression quality. The model *is* a compressor.

### Key insight

The rules say nothing about the model needing to be a neural network. This opened the door to a classical compression approach.

---

## 2. The Core Research Direction

Rather than competing on the neural leaderboard (which is dominated by transformer hyperparameter tuning), the decision was to pursue a **genuinely novel approach**:

> A GPU-accelerated hierarchical text compressor that explicitly models morphological, syntactic, and discourse structure as a cascade of probability models, using symbolic tree representations to exploit literary and grammatical redundancy that flat n-gram models cannot see.

### Why this is interesting

- Almost nobody on the leaderboard is exploring this direction
- Plays to theoretical/analytical strengths rather than low-level ML tuning
- Even a non-record submission with a well-explained novel approach stands out
- Has potential as a legitimate research paper contribution

---

## 3. The Pipeline

### Overview

```
Raw Text
   ↓
[Stage 1]  Normalization
   ↓
[Stage 2]  Morphology          → root + transform codes
   ↓
[Stage 3]  Syntax              → tree structures + POS
   ↓
[Stage 4]  Discourse           → coreference links + device templates
   ↓
[Stage 5]  Symbolic Encoding   → compact symbol stream
   ↓
[Stage 6]  Probability Model   → distributions per symbol (CPU)
   ↓
[Stage 7]  ANS Encoding        → compressed binary (GPU)
   ↓
   16MB artifact
   ↓
[Stage 8]  Reverse             → original text
```

### Stage 1 — Text Ingestion & Normalization
- Sentence boundary detection
- Whitespace and encoding normalization
- Basic tokenization

### Stage 2 — Morphological Analysis ✅ Implemented
- Lemmatization — reduce words to root forms ("running" → "run")
- Morpheme decomposition — split into root + affixes
- Inflection tagging — record tense, number, degree separately from root
- Output: every word becomes a **root + transformation code** pair
- See Section 9 for prototype results

### Stage 3 — POS Tagging & Syntactic Parsing
- Assign POS tags to every token
- Build **syntactic trees** per sentence (the keystone structure)
- Identify phrase boundaries (NP, VP, PP etc.)
- Tag clause types (main, subordinate, relative)
- Identify sentence type (declarative, interrogative, imperative, exclamatory)
- Tag voice (active/passive) and tense/aspect

### Stage 4 — Discourse & Coreference Analysis ✅ Implemented
- Coreference resolution — link pronouns and definite references back to antecedents
- Discourse connective tagging — cause/effect, contrast, sequence, elaboration
- Symbolic link building — every reference to something earlier becomes a pointer
- Literary device detection — parallelism, anaphora, antithesis, simile patterns
- Uses `longformer-large-4096` fine-tuned on OntoNotes for coreference

### Stage 5 — Symbolic Encoding ✅ Implemented
- Tree shapes encoded as structure symbols
- POS sequences encoded relative to tree position
- Words encoded as root + morphological transformation code
- Coreference pointers replace repeated noun phrases
- Literary device templates replace their instances
- Discourse relation types get single-symbol codes

### Stage 6 — Probability Modeling (CPU)
- Compute probability distributions over symbols at each position
- Distributions informed by everything from stages 1–5
- Context mixing — combining evidence from multiple hierarchy levels

### Stage 7 — GPU Accelerated Encoding
- Parallel probability table lookups across thousands of contexts
- **ANS (Asymmetric Numeral Systems)** encoding — GPU-friendly alternative to arithmetic coding
- Novel contribution: GPU parallelism designed around the symbolic structure, not bolted on afterward

### Stage 8 — Decoding (Reverse Pipeline)
- GPU-accelerated ANS decoding
- Reconstruct probability models
- Reconstruct sentences from trees + word forms
- Resolve morphological codes back to surface forms
- Follow coreference pointers to reconstruct full noun phrases

---

## 4. The Symbol Alphabet

### What it is

The complete set of distinct symbols the encoder needs to represent — analogous to assembly language opcodes. Every possible English sentence gets broken down into symbols from this fixed set.

### The assembly language analogy

Just as assembly has opcodes (MOV, ADD, JMP) and operands (register names, values), the symbol alphabet has:
- **Structural symbols** — grammar patterns (NP_START, PAST_TENSE, CAUSE_RELATION...)
- **Lexical symbols** — actual words/roots filling those patterns

### Alphabet layers

**Morphological codes (~100 symbols)**
```
PLURAL, PAST_TENSE, PRESENT_PART, PAST_PART,
COMPARATIVE, SUPERLATIVE, NEGATION, AGENT,
NOMINALIZATION, ADVERBIAL, IRREGULAR
```

**Tree shape codes (~2,000 symbols)**
Valid syntactic tree shapes in English

**Phrase and clause labels (~50 symbols)**
```
NP, VP, PP, ADJP, ADVP, SBAR, S...
```

**Sentence type codes (~10 symbols)**
```
DECLARATIVE, INTERROGATIVE, IMPERATIVE, EXCLAMATORY
```

**Discourse relation codes (~30 symbols)**
```
CAUSE, CONTRAST, SEQUENCE, ELABORATION,
CONDITION, ADDITION, SUMMARY
```

**Literary device codes (~40 symbols)**
```
PARALLELISM, ANAPHORA, SIMILE, ANTITHESIS,
METAPHOR, HYPERBOLE, PERSONIFICATION
```

**Lexical layer**
- Chosen approach: **character-by-character encoding**
- Small 26-letter alphabet + special symbols
- Handles unknown words, proper nouns, jargon naturally
- Tradeoff: slightly worse bpb score vs simpler architecture
- Compensated by highly accurate character transition predictions

### Special characters
```
'^' = start of word
'$' = end of word
'_' = whitespace
'.' = punctuation (normalized)
```

---

## 5. Key Technical Ideas

### 5a. Character Transition Graph

Build a graph where each node is a character and edges represent transition probabilities — what character tends to follow this one.

**Procedural generation** — given a starting character, follow the most probable path through the graph to reconstruct the word. Common words become nearly free because the graph predicts them with high confidence.

**Bit cost** — cost of encoding a character = `-log2(probability)`. Higher probability = fewer bits. Common words genuinely cost less.

Example from toy data:
```
'the'      → 4.20 total bits  (1.40 bits/char)
'silently' → 20.45 total bits (2.56 bits/char)
```

### 5b. ABC → 123 Mapping (Bidirectional)

Assign numeric IDs to symbols based on frequency — most common symbols get lowest IDs. The foundation everything sits on. Fully reversible.

### 5c. Delta Encoding

Instead of storing symbol values absolutely, store the **difference between consecutive values**.

- Works well when consecutive symbols are numerically close
- **Does NOT work at the character level with flat alphabet** — character IDs jump unpredictably
- **Does work with phonetic decomposition** — class and position deltas stay in −5 to +5 range
- **Does work at the structural symbol level** — where POS sequences and tree shapes follow predictable patterns
- Finding: delta encoding at the character level requires the mixed-radix phonetic coordinate system to be effective

### 5d. Factoriadic Encoding

Represent numbers in the factorial number system — each digit position represents a factorial rather than a power of 10.

```
0   → [0]
1   → [1, 0]
5   → [2, 1, 0]
100 → [4, 0, 2, 0, 0]
```

- Small numbers get compact representations
- All values round-trip perfectly (verified in tests)
- Directly complements decomposed deltas — most deltas are 0, ±1, ±2 which cost almost nothing

### 5e. Procedural Decompression

Instead of storing each character explicitly, store **transition rules** and reconstruct words by following the highest probability path.

- Common words encode as start signal only — the graph generates them
- Uncommon words encode as start signal + deviation points
- Unknown words encode as start signal + full character sequence
- Deviation encoding: bit flags where 0 = follow most probable path, 1 = explicit character

### 5f. Mixed-Radix Phonetic Decomposition ✅ Tested

Replace the flat alphabet (a=0, b=1...z=25) with a 3-component coordinate system:

```
character → (phonetic_class, position_in_class, morphological_role)
```

**Phonetic classes (class component):**
| Class ID | Family | Members |
|---|---|---|
| 0 | Vowels | a, e, i, o, u |
| 1 | Stops | b, d, g, k, p, t |
| 2 | Fricatives | f, h, s, v, x, z |
| 3 | Nasals | m, n |
| 4 | Liquids | l, r |
| 5 | Other | c, j, q, w, y |
| 6 | Special | ^, $, _, . |

**Position component:** index of the letter within its class family (fixed lookup, 0-based)

**Morphological role component:** 0=word-start, 1=middle, 2=word-end, 3=standalone

**Key property:** Class and position are a fixed lookup table defined once. Role is the only component computed dynamically per character based on its position in its word.

**Relationship to complex numbers:** The (class, position) pair maps directly to a 2D complex plane coordinate. The complex magnitude between two characters equals the geometric distance in this space — small magnitude = phonetically similar = cheap transition. Mixed-radix gives the encoding mechanism; complex numbers give the geometric intuition. They are the same idea from two directions.

### 5g. GPU Acceleration Strategy

The reason classical algorithms like PAQ are slow is sequential probability updates on CPU. The novel design principle here:

**Design the probability model for GPU parallelism from the ground up, not as an afterthought.**

- Build all context tables and pattern lookups in parallel on GPU
- Feed results into lightweight sequential ANS coder
- GPU handles expensive "what's the probability given all contexts" lookups
- CPU handles fast final encoding arithmetic
- Uses all 8xH100s that would otherwise sit idle in a CPU-only approach

---

## 6. The Compiler Analogy

The pipeline closely mirrors a compiler:

```
Compiler                    Our Pipeline
─────────────────────────────────────────────────
Source code                 Raw text
Lexer / tokenizer           Morphological analysis
Parser                      Syntactic parser
Abstract Syntax Tree        Syntactic tree
Intermediate representation Symbolic encoding
Machine code output         Compressed binary output
```

Key difference: a compiler's output is designed to be **executed**. Our output is designed to be **reversed** back into the original text perfectly.

---

## 7. End-to-End Pipeline Walk-Through

Using the example sentence **"the old man"**:

**Step 1 — Raw input:** `the old man`

**Step 2 — Add markers:**
```
^ t h e $  _  ^ o l d $  _  ^ m a n $
```

**Step 3 — Assign 3-part coordinates per character:**
| Char | Class | Pos | Role |
|---|---|---|---|
| t | 1 (stop) | 5 | 0 (word-start) |
| h | 2 (fricative) | 1 | 1 (middle) |
| e | 0 (vowel) | 1 | 2 (word-end) |
| o | 0 (vowel) | 3 | 0 (word-start) |
| l | 4 (liquid) | 0 | 1 (middle) |
| d | 1 (stop) | 1 | 2 (word-end) |

**Step 4 — Compute deltas (subtract consecutive coordinates):**
| Transition | Flat delta | Class Δ | Pos Δ | Role Δ |
|---|---|---|---|---|
| t → h | −12 | +1 | −4 | +1 |
| h → e | −3 | −2 | 0 | +1 |
| e → o | +10 | 0 | +2 | −2 |

**Step 5 — Factoriadic encode each small delta → compressed binary output**

**Step 6 — Decode in reverse → recover "the old man" perfectly**

---

## 8. Test Results

### Test Suite 1 — Original compression_test.py (toy data, CPU)

#### Test 1 — Character Transition Graph ✅
- Graph built from toy text
- Procedural generation works — "the" generated correctly from just 't'
- Bit costs confirmed — common words genuinely cheaper than rare words

#### Test 2 — Delta Encoding with Flat Alphabet ⚠️
- Round-trip recovery works perfectly
- Delta range **larger** than absolute range at character level
- Confirmed finding: flat alphabet delta encoding is counterproductive at character level

#### Test 3 — Factoriadic Encoding ✅
- All test numbers round-trip perfectly
- Small numbers get compact representations as expected

---

### Test Suite 2 — Track A: Mixed-Radix Phonetic Decomposition

#### Test A1 — Toy corpus (75 pairs)
- Flat alphabet mean |Δ| = **9.41**, span = 46
- Decomposed class mean |Δ| = **2.55**, span = 10
- Decomposed position mean |Δ| = **1.92**, span = 10
- 2D combined magnitude = **3.61** → **2.61× improvement** over flat

#### Test A2 — Real text corpus (2,173 pairs from ~500-word Wikipedia-style passage)
- Flat alphabet mean |Δ| = **7.95**, span = 49
- Decomposed class mean |Δ| = **2.43**, span = 10
- Decomposed position mean |Δ| = **1.66**, span = 10
- Morphological role mean |Δ| = **0.38**, span = 2 — nearly free to encode separately
- 2D combined magnitude = **3.28** → **2.42× improvement** over flat
- 3D combined magnitude = **3.36** (role barely adds to geometric distance)
- 6.1% of transitions occur within the same phonetic class (class delta = 0)

#### Key findings from Track A
- Improvement is consistent between toy and real text — not a corpus artefact
- Morphological role belongs in a **separate 2-bit stream**, not mixed into the main delta
- Special symbols (^, $, _, .) successfully assigned to class 6 — handled uniformly
- Delta encoding is now beneficial at the character level **because** of phonetic decomposition

---

### Test Suite 3 — Track B: Morphological Analysis (Stage 2 Prototype)

#### Prototype details
- Pure Python, no external dependencies
- 13 transformation codes implemented: BASE, PLURAL, PAST_TENSE, PRESENT_PART, PAST_PART, THIRD_SING, COMPARATIVE, SUPERLATIVE, ADVERBIAL, NEGATION, AGENT, NOMINALIZE, IRREGULAR
- Irregular forms dictionary: ~60 common English irregulars hardcoded
- Suffix stripping with doubled-consonant fix and silent-e restoration

#### Results on demo sentence
*"the old man walked slowly home and the tired dog ran quickly away"*
- walked → walk + PAST_TENSE ✅
- slowly → slow + ADVERBIAL ✅
- ran → run + IRREGULAR ✅
- quickly → quick + ADVERBIAL ✅
- **15.1% of characters eliminated** — replaced by compact morph codes

#### Known limitation
- Silent-e restoration is imperfect without a vocabulary lookup (e.g. `connected → connecte`)
- This is resolved by using spaCy's lemmatizer in the real implementation
- All logic and structure is correct — only the final root surface form needs a dictionary

---

### Test Suite 4 — Session 3: Full Pipeline Round-Trip (Moby Dick, 10,000 chars)

#### Stage 4+5 Discourse Round-Trip ✅
- Model: `longformer-large-4096` fine-tuned on OntoNotes (90.5M params)
- Input: 9,742 chars / 1,526 tokens from Moby Dick
- **Round-trip OK: True**
- Original tokens: 1,526 → Compressed tokens: 1,515
- Token reduction: **0.72%**
- Symbols used: 15–16 (entity symbols like `§E1`=`the United States`, `§E2`=`Moby Dick` etc.)
- Coreference + entity replacement working end-to-end

#### Full Pipeline Round-Trip — bugs found and fixed

**Bug 1: UTF-8 BOM corruption (fixed)**
- `moby_dick.txt` has a UTF-8 BOM (`\ufeff`) as first character
- Test script was reading with `encoding="utf-8"` which left the BOM in the string
- Encoder and decoder both saw different starting points → `Match: False` at position 0
- Fix: open with `encoding="utf-8-sig"` which auto-strips BOM
- File patched: `test_round_trip_pipeline.py`

**Bug 2: `most → mosted` morphology corruption (fixed)**
- Root cause: `most`, `best`, `least`, `more`, `better` etc. were in `_IRREGULAR_SURFACE_TO_ROOT` with identity mappings (`"most": "most"`)
- `_analyse_token_spacy` was returning `("most", IRREGULAR)` for these
- `apply_morph("most", IRREGULAR)` calls `lemminflect(root, "VBD")` → produced `"mosted"` (past tense of a fake verb)
- Fix in `stage2_morphology.py`: added `_IDENTITY_WORDS` frozenset (all entries where surface == root) and return `(word, BASE)` for them before the IRREGULAR branch
- Fix in `morph_codes.py`: added the same words to `_DECODE_OVERRIDES` as decoder-side safety net for any old payloads
- Files patched: `compression/pipeline/stage2_morphology.py`, `compression/alphabet/morph_codes.py`

---

## 9. Open Research Questions

### Resolved since last session
- ✅ Complex numbers vs mixed-radix: confirmed they are the same idea from two directions
- ✅ Delta encoding placement: beneficial at character level only with phonetic decomposition
- ✅ POS delta encoding tested in Track B with structural encoder metrics
- ✅ Stage 4+5 discourse round-trip: confirmed working end-to-end
- ✅ UTF-8 BOM corruption: fixed in test script
- ✅ `most → mosted` morphology bug: fixed via `_IDENTITY_WORDS` guard and `_DECODE_OVERRIDES`

### Still open
- Full pipeline round-trip `Match: True` still pending — needs verification after morphology fix
- How to order structural symbol IDs so common sequential patterns produce small deltas
- Full formal design of the complete symbol alphabet across all layers
- Whether the 2.42× character-level improvement survives when combined with the full pipeline
- Evaluate POS delta improvement on larger, more diverse corpora

### Complex number / quaternion extension
- The 3rd component (morphological role) doesn't map onto the 2D complex plane
- Options: treat as a second independent stream (recommended), or extend to a 3D space (quaternion)
- Recommended: encode role as a flat 2-bit code — it has only 4 values and near-zero entropy

---

## 10. Current Status by Stage

| Stage | Status | Notes |
|---|---|---|
| Stage 1 — Normalization | ✅ Implemented | `normalize_text` handles whitespace, line endings, UTF-8 BOM |
| Stage 2 — Morphology | ✅ Implemented + Patched | spaCy-backed lemmatization + rule-based fallback; `_IDENTITY_WORDS` lossless fix applied |
| Stage 3 — Syntax/POS | ✅ Implemented | spaCy POS/dep, tree-shape serialization, sentence type/voice, POS delta |
| Stage 4 — Discourse | ✅ Implemented | Longformer coreference + entity symbol encoding; round-trip verified |
| Stage 5 — Symbolic Encoding | ✅ Implemented | Character + structural + discourse streams wired; round-trip verified |
| Stage 6 — Probability Model | ✅ Implemented (context-mixing) | 3-level context model + bpb + serialisation |
| Stage 7 — ANS/GPU Encoding | ✅ Implemented (CPU rANS) | Correct rANS encode/decode + tests; GPU pending |
| Stage 8 — Decoding | ✅ Implemented | Full decode path + ANS stream reconstruction |

---

## 11. Next Steps

### Immediate (start of next session)
1. **Verify** `Match: True` for full pipeline round-trip after morphology patch — run `python test_round_trip_pipeline.py --chars 10000`
2. **If more mismatches appear**, repeat the same diagnostic: find the first mismatch character, trace back to which pipeline stage introduces it
3. **Scale up** round-trip test to 50k, 100k chars once 10k passes cleanly

### Track B — Structural layer
1. Evaluate structural streams on larger corpora
2. Decide structural symbol ID ordering (fixed vs frequency-based)

### System integration
1. GPU ANS implementation or CUDA acceleration strategy
2. Finalize interleaved stream layout for binary payload format
3. End-to-end compression benchmarking on FineWeb

### Hardware note
Local environment (Nobara Linux, RTX 3060 12GB, Ryzen 5 4600G, 16GB DDR4) is ready for:
- spaCy model download and Stage 3 prototype
- GPU ANS encoding experiments (Stage 7)

---

## 12. References & Inspirations

- [OpenAI Parameter Golf Challenge](https://github.com/openai/parameter-golf)
- [NanoGPT Speedrunning](https://github.com/KellerJordan/modded-nanogpt) — the challenge this is inspired by
- PAQ compression family — context mixing approach (KGB Archiver is a frontend for PAQ6)
- PPM (Prediction by Partial Matching) — classical text compression, 1984
- ANS (Asymmetric Numeral Systems) — modern alternative to arithmetic coding, used in zstd
- nvCOMP — NVIDIA's GPU-accelerated compression library
- Neural scaling laws paper — [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361)
- Factoriadic number system — factorial base representation
- Mixed-radix number systems — generalization of positional notation
- Longformer — [Beltagy et al. 2020](https://arxiv.org/abs/2004.05150) — used for coreference resolution in Stage 4

---

*Last updated: March 28, 2026. Session 3 complete.*
