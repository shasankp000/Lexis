# Lexis-E Payload Metadata Fields

All fields are written by `compress_to_file()` in `main.py` and packed into a
single msgpack binary. They are read back verbatim by `decompress()` via
`_build_context_model()` and `_build_encoded_sentences_from_metadata()`.

---

## Core Bitstreams

| Field | Python type | Example value | Notes |
|---|---|---|---|
| `compressed_bitstream` | `bytes` | `b'\x8f\x3a\x11\xc4...'` | Arithmetic-coded char-class stream. Size ≈ bpb × char_count / 8. |
| `pos_deltas_bitstream` | `bytes` | `b'\x02\xf1\xa3\x00...'` | Arithmetic-coded phonetic position-delta stream. |
| `pos_deltas_count` | `int` | `2847` | Total number of deltas encoded; needed by decoder to know when to stop. |

---

## Per-Sentence Arrays

Each list has one entry per sentence in the document. Lists-of-lists have one
inner list per sentence, with one element per token inside that sentence.

| Field | Python type | Example value | Notes |
|---|---|---|---|
| `sentence_char_counts` | `List[int]` | `[23, 15, 31, 18, ...]` | Total encoded characters (incl. `^`/`$` boundary markers) for each sentence. Used by decoder to slice `char_classes` back into per-sentence segments. |
| `pos_tags` | `List[List[str]]` | `[["NOUN","VERB","DET","NOUN"], ["PRON","AUX","ADJ"], ...]` | spaCy universal POS tag per token, per sentence. Used to rebuild `char_pos_tags` stream in `_build_encoded_sentences_from_metadata`. |
| `morph_codes` | `List[List[int]]` | `[[0, 2, 0, 1], [0, 0, 6], ...]` | Integer morphological transformation code per token, per sentence. `0` = BASE (no change), `1` = PLURAL, `2` = PAST_TENSE, etc. Used to rebuild `char_morph_codes` stream. |
| `root_lengths` | `List[List[int]]` | `[[4, 3, 3, 3], [2, 2, 4], ...]` | Character length of each lemma root per token, per sentence. Used by `_build_encoded_sentences_from_metadata` to know how many chars to assign each POS/morph tag to. |
| `pos_huffman_bits` | `List[float]` | `[14.2, 9.0, 22.7, 11.5, ...]` | Huffman-coded bit cost of the POS tag sequence for each sentence. Used only for `struct_prob` weighting inside the context stream; **not** needed for arithmetic decode correctness. |
| `pos_n_tags` | `List[int]` | `[4, 3, 7, 5, ...]` | Number of POS tags in each sentence. Redundant with `len(pos_tags[i])`; stored for fast access during context-stream construction. |

---

## Position Delta Counts

| Field | Python type | Example value | Notes |
|---|---|---|---|
| `pos_deltas_counts` | `Dict[int, int]` | `{-3: 12, -1: 89, 0: 341, 1: 94, 2: 8}` | Frequency table of every delta value that appeared in the position stream. Used to reconstruct the unigram distribution for `decode_unigram_counts`. Keys are signed integers (delta values); values are occurrence counts. |

---

## Context Model — Transition Counts

These three dicts are the trained online context model. They are the largest
fields in the payload. Each maps a context key to a `Counter` of observed
next-symbol frequencies.

| Field | Python type | Example value | Notes |
|---|---|---|---|
| `char_context` | `Dict[int, Dict[int, int]]` | `{0: {0: 41, 2: 7, 5: 3}, 3: {1: 12, 4: 8}, ...}` | Char-class bigram counts. Outer key = previous char class (0–6); inner key = next char class; value = count. Window size = `CHAR_CONTEXT_SIZE`. |
| `morph_context` | `Dict[int, Dict[int, int]]` | `{0: {3: 102, 5: 14}, 2: {1: 30}, ...}` | Morph-code bigram counts. Outer key = current morph code; inner key = next char class; value = count. Window size = `MORPH_CONTEXT_SIZE`. |
| `struct_context` | `Dict[str, Dict[int, int]]` | `{"NOUN": {0: 88, 3: 22}, "VERB": {2: 41}, ...}` | POS-tag → char-class transition counts. Outer key = current POS tag string; inner key = next char class; value = count. Window size = `STRUCT_CONTEXT_SIZE`. |

---

## Context Model — Weights and Vocab

| Field | Python type | Example value | Notes |
|---|---|---|---|
| `model_weights` | `List[float]` | `[0.9, 0.05, 0.05]` | Interpolation weights for the three context sub-models: `[char_weight, morph_weight, struct_weight]`. Must sum to 1.0. |
| `char_vocab` | `List[int]` | `[0, 1, 2, 3, 4, 5, 6]` | All char-class IDs seen during training. Always exactly 7 entries (one per phonetic class). |
| `morph_vocab` | `List[int]` | `[0, 1, 2, 3, 5, 6, 8, 9, 12]` | All morph-code IDs seen during training. Variable length; only codes that actually appeared. |
| `pos_vocab` | `List[str]` | `["NOUN","VERB","PUNCT","DET","ADP","ADJ","ADV","PRON","AUX","PROPN","NUM","PART","CCONJ","SCONJ","X","SYM","INTJ"]` | All POS tag strings seen during training. Variable length; only tags that actually appeared. |

---

## Scalars and Tables

| Field | Python type | Example value | Notes |
|---|---|---|---|
| `num_symbols` | `int` | `2847` | Total number of char-class symbols in the compressed stream. Tells the arithmetic decoder exactly when to stop decoding. |
| `num_char_classes` | `int` | `7` | Number of distinct phonetic char classes. Always `7`; stored for forward-compatibility. |
| `pos_freq_table` | `Dict[str, int]` | `{"NOUN": 412, "VERB": 298, "PUNCT": 187, "DET": 134, ...}` | Corpus-level POS tag frequency counts. Used to build the Huffman tree for POS encoding/decoding. |
| `huffman_codes` | `Dict[str, str]` | `{"NOUN": "00", "VERB": "01", "PUNCT": "10", "SYM": "11000", ...}` | Precomputed Huffman bit-string per POS tag. **Fully redundant** — reconstructible from `pos_freq_table` at decode time. Stored only as a decode-time shortcut. |

---

## Stage 4+5 — Discourse Symbol Table

| Field | Python type | Example value | Notes |
|---|---|---|---|
| `symbol_table` | `Dict[str, str]` | `{"§E0": "John Smith", "§E1": "New York", "§E2": "the company"}` | Maps each `§E{n}` discourse placeholder back to its original surface form. Applied as the very last step in `decompress()` via `decode_symbols()`. Empty dict `{}` if no coreference chains were found. |

---

## Field Redundancy Summary

| Field | Required for decode? | Notes |
|---|---|---|
| `compressed_bitstream` | ✅ Yes | Core payload |
| `pos_deltas_bitstream` | ✅ Yes | Core payload |
| `pos_deltas_count` | ✅ Yes | Decoder stop condition |
| `sentence_char_counts` | ✅ Yes | Char-stream slicing |
| `pos_tags` | ✅ Yes | Context stream rebuild |
| `morph_codes` | ✅ Yes | Context stream rebuild |
| `root_lengths` | ✅ Yes | Context stream rebuild |
| `char_context` | ✅ Yes | Probability model |
| `morph_context` | ✅ Yes | Probability model |
| `struct_context` | ✅ Yes | Probability model |
| `model_weights` | ✅ Yes | Probability model |
| `char_vocab` | ✅ Yes | Distribution domain |
| `morph_vocab` | ✅ Yes | Distribution domain |
| `pos_vocab` | ✅ Yes | Distribution domain |
| `num_symbols` | ✅ Yes | Decoder stop condition |
| `pos_freq_table` | ✅ Yes | Huffman tree rebuild |
| `symbol_table` | ✅ Yes (if non-empty) | Stage 4+5 decode |
| `pos_deltas_counts` | ✅ Yes | Delta decoder distribution |
| `pos_huffman_bits` | ⚠️ Soft | Only used for `struct_prob`; derivable from `pos_tags` + `pos_freq_table` |
| `pos_n_tags` | ⚠️ Soft | Redundant with `len(pos_tags[i])` |
| `huffman_codes` | ❌ No | Fully reconstructible from `pos_freq_table` |
| `num_char_classes` | ❌ No | Always `7`; forward-compat stub only |
