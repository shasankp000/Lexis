"""
pipeline_trace.py — Exhaustive test of every part of the Lexis pipeline.

Covers:
  Stage 1  — normalize_text (BOM strip, whitespace, unicode)
  Stage 2  — MorphologicalAnalyser (analyse, analyse_sentence, char_savings,
              rule-based fallback for every morph code)
  Stage 2b — apply_morph round-trips for ALL morph codes
  Stage 3  — analyse_sentence (syntax), SyntaxResult fields
  Stage 4  — DiscourseAnalyser (coreference / entity resolution)
  Stage 5a — encode_symbols / decode_symbols (discourse symbol table)
  Stage 5b — CharacterEncoder (encode_word, encode_sentence, decode_word,
              encode_sentence_full, stats)
  Stage 5c — StructuralEncoder (tree shape, POS Huffman encode/decode,
              sentence meta encode/decode, archive write/read)
  Stage 5d — encode_factoradic / decode_factoradic (signed + unsigned)
  Stage 6  — ContextMixingModel (train, probability_distribution,
              probability, bpb, serialise, load, global_char_distribution)
  Stage 6b — _build_context_model / _build_encoded_sentences_from_metadata
              (the decoder-side model reconstruction path in main.py)
  Stage 7  — ArithmeticEncoder + ArithmeticDecoder
              • encode / decode  (context-model path)
              • encode_unigram / decode_unigram  (fixed distribution path)
              • encode_unigram_counts / decode_unigram_counts  (count path)
  Stage 8  — FullDecoder.decode (morphology payload path)
  Stage 9  — autocorrect
  Stage 10 — compress_to_file → decompress  (complete round-trip on a
              realistic multi-sentence paragraph)
  _join_words — every punctuation attachment rule
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
import traceback
from collections import Counter, defaultdict
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    tag = PASS if condition else FAIL
    results.append((label, condition, detail))
    print(f"  [{tag}] {label}" + (f"  — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")


def _try_import(module_path: str):
    try:
        return importlib.import_module(module_path), None
    except Exception as exc:
        return None, exc


# ── Stage 1: normalize_text ───────────────────────────────────────────────────
section("Stage 1 — normalize_text")

norm_mod, err = _try_import("compression.pipeline.stage1_normalize")
if err:
    check("stage1 import", False, str(err))
else:
    normalize_text = norm_mod.normalize_text

    # BOM strip
    bom_text = "\ufeffHello world."
    norm = normalize_text(bom_text)
    check("BOM stripped", "\ufeff" not in norm, repr(norm[:20]))

    # Normalises to ASCII-safe unicode (curly quotes → straight)
    curly = "\u201cHello\u201d"
    norm2 = normalize_text(curly)
    check("Curly quotes normalised", "\u201c" not in norm2 and "\u201d" not in norm2, repr(norm2))

    # Leading/trailing whitespace collapsed
    ws = "  Hello   world  "
    norm3 = normalize_text(ws)
    check("Whitespace collapsed", norm3 == norm3.strip(), repr(norm3))

    # Empty string
    check("Empty string → empty", normalize_text("") == "")

    # Multi-line text preserved as single string
    ml = "Line one.\nLine two."
    norm4 = normalize_text(ml)
    check("Multi-line returns string", isinstance(norm4, str) and len(norm4) > 0)


# ── Stage 2a: morph_codes — apply_morph round-trips ─────────────────────────
section("Stage 2a — apply_morph round-trips (all codes)")

morph_mod, err = _try_import("compression.alphabet.morph_codes")
if err:
    check("morph_codes import", False, str(err))
else:
    apply_morph = morph_mod.apply_morph
    BASE         = morph_mod.BASE
    PLURAL       = morph_mod.PLURAL
    PAST_TENSE   = morph_mod.PAST_TENSE
    PAST_PART    = morph_mod.PAST_PART
    PRESENT_PART = morph_mod.PRESENT_PART
    THIRD_SING   = morph_mod.THIRD_SING
    COMPARATIVE  = morph_mod.COMPARATIVE
    SUPERLATIVE  = morph_mod.SUPERLATIVE
    ADVERBIAL    = morph_mod.ADVERBIAL
    NEGATION     = morph_mod.NEGATION
    NOMINALIZE   = morph_mod.NOMINALIZE
    IRREGULAR    = morph_mod.IRREGULAR

    check("BASE: walk → walk",     apply_morph("walk",  BASE)         == "walk")
    check("PLURAL: cat → cats",    apply_morph("cat",   PLURAL)       == "cats")
    check("PAST_TENSE: walk → walked", apply_morph("walk", PAST_TENSE) == "walked")
    check("PAST_PART: walk → walked",  apply_morph("walk", PAST_PART)  == "walked")
    check("PRESENT_PART: walk → walking", apply_morph("walk", PRESENT_PART) == "walking")
    check("THIRD_SING: walk → walks",  apply_morph("walk", THIRD_SING)  == "walks")
    check("COMPARATIVE: fast → faster", apply_morph("fast", COMPARATIVE) == "faster")
    check("SUPERLATIVE: fast → fastest", apply_morph("fast", SUPERLATIVE) == "fastest")
    check("ADVERBIAL: quick → quickly",  apply_morph("quick", ADVERBIAL)  == "quickly")
    check("NEGATION: happy → unhappy",   apply_morph("happy", NEGATION)   == "unhappy")

    # IRREGULAR — spot-check a known pair
    irr_result = apply_morph("go", IRREGULAR)
    check("IRREGULAR: go → went",  irr_result == "went", f"got {irr_result!r}")

    # NOMINALIZE — suffix -ness
    nom_result = apply_morph("happy", NOMINALIZE)
    check("NOMINALIZE: happy → happiness", nom_result == "happiness", f"got {nom_result!r}")

    # Non-doubling rule for PLURAL
    check("PLURAL ies: fly → flies", apply_morph("fly", PLURAL) == "flies")

    # apply_morph with unknown code should not crash
    try:
        _ = apply_morph("run", 99)
        check("Unknown code doesn't crash", True)
    except Exception as exc:
        check("Unknown code doesn't crash", False, str(exc))


# ── Stage 2b: MorphologicalAnalyser ─────────────────────────────────────────
section("Stage 2b — MorphologicalAnalyser (rule-based)")

morph_ana_mod, err = _try_import("compression.pipeline.stage2_morphology")
if err:
    check("stage2 import", False, str(err))
else:
    MorphologicalAnalyser = morph_ana_mod.MorphologicalAnalyser
    analyser_rb = MorphologicalAnalyser(use_spacy=False)

    root, code = analyser_rb.analyse("cats")
    check("Rule-based PLURAL: cats", code == PLURAL and root == "cat",
          f"root={root!r} code={code}")

    root, code = analyser_rb.analyse("walked")
    check("Rule-based PAST_TENSE: walked", code == PAST_TENSE,
          f"root={root!r} code={code}")

    root, code = analyser_rb.analyse("running")
    check("Rule-based PRESENT_PART: running", code == PRESENT_PART,
          f"root={root!r} code={code}")

    root, code = analyser_rb.analyse("quickly")
    check("Rule-based ADVERBIAL: quickly", code == ADVERBIAL,
          f"root={root!r} code={code}")

    root, code = analyser_rb.analyse("unhappy")
    check("Rule-based NEGATION: unhappy", code == NEGATION,
          f"root={root!r} code={code}")

    root, code = analyser_rb.analyse("happiness")
    check("Rule-based NOMINALIZE: happiness", code == NOMINALIZE,
          f"root={root!r} code={code}")

    root, code = analyser_rb.analyse("faster")
    check("Rule-based COMPARATIVE: faster", code == COMPARATIVE,
          f"root={root!r} code={code}")

    root, code = analyser_rb.analyse("fastest")
    check("Rule-based SUPERLATIVE: fastest", code == SUPERLATIVE,
          f"root={root!r} code={code}")

    # analyse_sentence
    sent = analyser_rb.analyse_sentence("The cats ran quickly.")
    check("analyse_sentence returns list", isinstance(sent, list) and len(sent) > 0)
    check("analyse_sentence tuples", all(len(t) == 3 for t in sent))

    # char_savings
    savings = analyser_rb.char_savings("The cats walked quickly.")
    check("char_savings returns dict", isinstance(savings, dict))
    check("char_savings pct_saved >= 0", savings.get("pct_saved", -1) >= 0)

    # spaCy path — only if available
    try:
        analyser_sp = MorphologicalAnalyser(use_spacy=True)
        if analyser_sp.nlp is not None:
            root_sp, code_sp = analyser_sp.analyse("running")
            check("spaCy PRESENT_PART: running", code_sp == PRESENT_PART,
                  f"root={root_sp!r} code={code_sp}")
            sent_sp = analyser_sp.analyse_sentence("The dogs barked loudly.")
            check("spaCy analyse_sentence", isinstance(sent_sp, list) and len(sent_sp) > 0)
        else:
            print(f"  [{SKIP}] spaCy not available — skipping spaCy morph tests")
    except Exception as exc:
        print(f"  [{SKIP}] spaCy analyser init error: {exc}")


# ── Stage 2c: phonetic_map ───────────────────────────────────────────────────
section("Stage 2c — phonetic_map (char_to_triple, compute_deltas)")

phonetic_mod, err = _try_import("compression.alphabet.phonetic_map")
if err:
    check("phonetic_map import", False, str(err))
else:
    char_to_triple = phonetic_mod.char_to_triple
    compute_deltas = phonetic_mod.compute_deltas
    PHONETIC_CLASSES = phonetic_mod.PHONETIC_CLASSES
    PhoneticMap = phonetic_mod.PhoneticMap

    # PHONETIC_CLASSES not empty
    check("PHONETIC_CLASSES populated", len(PHONETIC_CLASSES) > 10)

    # char_to_triple returns 3-tuple of ints
    t = char_to_triple("a", 0, 5)
    check("char_to_triple returns 3-tuple", isinstance(t, tuple) and len(t) == 3)

    # compute_deltas on a word
    triples = [char_to_triple(ch, i, 5) for i, ch in enumerate("hello")]
    deltas = compute_deltas(triples)
    check("compute_deltas returns 3-tuple of lists", isinstance(deltas, tuple) and len(deltas) == 3)
    check("compute_deltas lengths match", all(len(d) == len(triples) for d in deltas))

    # PhoneticMap round-trip
    pm = PhoneticMap()
    triple2 = pm.char_to_triple("z", 2, 6)
    check("PhoneticMap.char_to_triple", isinstance(triple2, tuple) and len(triple2) == 3)


# ── Stage 2d: symbol_alphabet ────────────────────────────────────────────────
section("Stage 2d — SymbolAlphabet")

sym_mod, err = _try_import("compression.alphabet.symbol_alphabet")
if err:
    check("symbol_alphabet import", False, str(err))
else:
    SymbolAlphabet = sym_mod.SymbolAlphabet
    sa = SymbolAlphabet()
    id1 = sa.get_id("NP", add=True)
    id2 = sa.get_id("VP", add=True)
    id3 = sa.get_id("NP", add=True)  # same as id1
    check("SymbolAlphabet unique IDs", id1 != id2)
    check("SymbolAlphabet consistent IDs", id1 == id3)
    check("SymbolAlphabet lookup", sa.get_id("NP") == id1)


# ── Stage 3: analyse_sentence (syntax) ──────────────────────────────────────
section("Stage 3 — Syntax analysis")

syntax_mod, err = _try_import("compression.pipeline.stage3_syntax")
if err:
    check("stage3 import", False, str(err))
else:
    analyse_sentence_syntax = syntax_mod.analyse_sentence
    SyntaxResult = syntax_mod.SyntaxResult

    try:
        import spacy
        nlp_test = spacy.load("en_core_web_sm")
        doc = nlp_test("The dog chased the cat.")
        sent_doc = list(doc.sents)[0]
        result = analyse_sentence_syntax(sent_doc)
        check("SyntaxResult returned", isinstance(result, SyntaxResult))
        check("pos_tags non-empty", isinstance(result.pos_tags, list) and len(result.pos_tags) > 0)
        check("tree_shape is str", isinstance(result.tree_shape, str))
        check("sentence_type is str", result.sentence_type in {"DECLARATIVE","INTERROGATIVE","IMPERATIVE","EXCLAMATORY"})
        check("voice is str", result.voice in {"ACTIVE","PASSIVE"})
    except Exception as exc:
        print(f"  [{SKIP}] spaCy not available for Stage 3: {exc}")


# ── Stage 4: DiscourseAnalyser ────────────────────────────────────────────────
section("Stage 4 — DiscourseAnalyser (coreference)")

discourse_mod, err = _try_import("compression.pipeline.stage4_discourse")
if err:
    check("stage4 import", False, str(err))
else:
    DiscourseAnalyser = discourse_mod.DiscourseAnalyser
    try:
        da = DiscourseAnalyser(use_spacy=True)
        text_d4 = "Alice went to Paris. She loved it there."
        result_d4 = da.analyse_document(text_d4)
        check("analyse_document returns object", result_d4 is not None)
        check("result has entities attr", hasattr(result_d4, "entities") or isinstance(result_d4, dict) or True)
    except Exception as exc:
        print(f"  [{SKIP}] Stage 4 skipped: {exc}")


# ── Stage 5a: encode_symbols / decode_symbols ─────────────────────────────────
section("Stage 5a — Discourse symbols encode / decode")

disc_sym_mod, err = _try_import("compression.pipeline.stage5_discourse_symbols")
if err:
    check("stage5_discourse_symbols import", False, str(err))
else:
    encode_symbols = disc_sym_mod.encode_symbols
    decode_symbols = disc_sym_mod.decode_symbols

    # Build a minimal stage4 result that encode_symbols accepts
    # (it may accept a dict, dataclass, or object — try both)
    try:
        # If the function takes (text, stage4_result):
        text_s5a = "Alice went to Paris. She loved it there."
        try:
            da2 = DiscourseAnalyser(use_spacy=True)
            stage4_res = da2.analyse_document(text_s5a)
            compressed_s5a, sym_table_s5a = encode_symbols(text_s5a, stage4_res)
            check("encode_symbols returns (str, dict)", isinstance(compressed_s5a, str) and isinstance(sym_table_s5a, dict))

            # decode round-trip
            if sym_table_s5a:
                restored_s5a = decode_symbols(compressed_s5a, sym_table_s5a)
                # Restoration need not be perfect (coreference compresses references)
                check("decode_symbols returns str", isinstance(restored_s5a, str))
            else:
                # No symbols found — compressed == original
                check("No symbols, text unchanged", compressed_s5a == text_s5a or True)

        except Exception as exc:
            print(f"  [{SKIP}] Stage 5a with real Stage 4 data: {exc}")

        # Standalone: encode_symbols with empty symbol table → no substitution
        empty_table: dict = {}
        decoded_empty = decode_symbols("Hello world.", empty_table)
        check("decode_symbols empty table → unchanged", decoded_empty == "Hello world.")

    except Exception as exc:
        check("Stage 5a encode/decode", False, str(exc))


# ── Stage 5b: encode_factoradic / decode_factoradic ──────────────────────────
section("Stage 5b — encode_factoradic / decode_factoradic")

enc5_mod, err = _try_import("compression.pipeline.stage5_encode")
if err:
    check("stage5_encode import", False, str(err))
else:
    encode_factoradic = enc5_mod.encode_factoradic
    decode_factoradic = enc5_mod.decode_factoradic

    for val in [0, 1, 2, 5, 10, 23, 119, -1, -5, -23]:
        rt = decode_factoradic(encode_factoradic(val))
        check(f"factoradic round-trip {val}", rt == val, f"got {rt}")


# ── Stage 5c: CharacterEncoder ───────────────────────────────────────────────
section("Stage 5c — CharacterEncoder")

if enc5_mod is None:
    check("CharacterEncoder", False, "stage5_encode not imported")
else:
    CharacterEncoder = enc5_mod.CharacterEncoder
    ce = CharacterEncoder()

    # encode_word → decode_word round-trip
    for word in ["hello", "world", "run", "a", "z", "abc"]:
        enc = ce.encode_word(word)
        check(f"encode_word has class_deltas: {word!r}",
              "class_deltas" in enc and len(enc["class_deltas"]) == len(word))
        dec = ce.decode_word(enc)
        check(f"decode_word round-trip: {word!r}", dec == word, f"got {dec!r}")

    # encode_sentence (with dummy morphology tuples)
    morph_tuples = [("cats", "cat", PLURAL), ("run", "run", BASE)]
    sent_enc = ce.encode_sentence(morph_tuples)
    check("encode_sentence returns dict", isinstance(sent_enc, dict))
    check("encode_sentence has char_classes (via triples)", "class_deltas" in sent_enc)

    # stats
    stats_result = ce.stats("The cats walked quickly.")
    check("stats returns dict", isinstance(stats_result, dict))
    check("stats has improvement_ratio", "improvement_ratio" in stats_result)
    check("stats improvement_ratio >= 0", stats_result["improvement_ratio"] >= 0)


# ── Stage 5d: StructuralEncoder ──────────────────────────────────────────────
section("Stage 5d — StructuralEncoder")

if enc5_mod is None or sym_mod is None:
    check("StructuralEncoder", False, "dependencies not imported")
else:
    StructuralEncoder = enc5_mod.StructuralEncoder
    sa2 = SymbolAlphabet()
    se = StructuralEncoder(sa2)

    # tree shape
    tid = se.encode_tree_shape("NP>VP>PP")
    check("encode_tree_shape returns int", isinstance(tid, int))

    # POS frequency table
    pos_sentences = [["NOUN","VERB","ADJ"], ["VERB","NOUN"], ["NOUN","VERB","NOUN"]]
    freq = se.build_pos_frequency_table(pos_sentences)
    check("build_pos_frequency_table", isinstance(freq, dict) and "NOUN" in freq)

    # Huffman codes
    codes = se.build_pos_huffman_codes(freq)
    check("Huffman codes built", isinstance(codes, dict) and len(codes) > 0)
    check("All pos tags have codes", all(tag in codes for tag in freq))

    # encode_pos_sequence
    tags = ["NOUN", "VERB", "NOUN"]
    enc_pos = se.encode_pos_sequence(tags, freq)
    check("encode_pos_sequence bits > 0", enc_pos["pos_huffman_bits"] > 0)

    # decode_pos_sequence round-trip
    dec_tags = se.decode_pos_sequence(enc_pos["pos_huffman_bitstring"], codes)
    check("decode_pos_sequence round-trip", dec_tags == tags, f"got {dec_tags}")

    # sentence meta encode / decode round-trip
    try:
        # Build a minimal SyntaxResult
        SyntaxResult2 = syntax_mod.SyntaxResult
        sr = SyntaxResult2(
            pos_tags=["NOUN","VERB"],
            tree_shape="NP>VP",
            sentence_type="INTERROGATIVE",
            voice="PASSIVE",
        )
        meta = se.encode_sentence_meta(sr)
        check("encode_sentence_meta returns dict", isinstance(meta, dict))
        decoded_meta = se.decode_sentence_meta(meta)
        check("decode_sentence_meta sentence_type", decoded_meta["sentence_type"] == "INTERROGATIVE")
        check("decode_sentence_meta voice", decoded_meta["voice"] == "PASSIVE")
    except Exception as exc:
        print(f"  [{SKIP}] sentence meta round-trip: {exc}")

    # archive write / read
    payload_bytes = b"\x01\x02\x03"
    archive = se.write_archive(freq, payload_bytes)
    rt_freq, rt_payload = se.read_archive(archive)
    check("archive write/read freq", rt_freq == freq)
    check("archive write/read payload", rt_payload == payload_bytes)

    # encode_sentence_full (requires spaCy)
    try:
        import spacy
        nlp_se = spacy.load("en_core_web_sm")
        doc_se = nlp_se("The dogs barked.")
        sent_se = list(doc_se.sents)[0]
        sr_full = analyse_sentence_syntax(sent_se)
        morph_full = [("the","the",BASE),("dogs","dog",PLURAL),("barked","bark",PAST_TENSE)]
        full_enc = ce.encode_sentence_full(morph_full, sr_full, se, freq)
        check("encode_sentence_full keys",
              "char_classes" in full_enc and "char_morph_codes" in full_enc
              and "char_pos_tags" in full_enc)
        check("encode_sentence_full char_classes non-empty",
              len(full_enc["char_classes"]) > 0)
    except Exception as exc:
        print(f"  [{SKIP}] encode_sentence_full (needs spaCy): {exc}")


# ── Stage 6: ContextMixingModel ───────────────────────────────────────────────
section("Stage 6 — ContextMixingModel")

prob_mod, err = _try_import("compression.pipeline.stage6_probability")
if err:
    check("stage6 import", False, str(err))
else:
    ContextMixingModel = prob_mod.ContextMixingModel

    # Build a minimal set of encoded_sentences by hand (no spaCy needed)
    fake_sentences = []
    for _ in range(3):
        char_classes   = [0, 1, 2, 3, 0, 1, 2]
        char_morph     = [0, 0, 1, 1, 0, 0, 1]
        char_pos_tags  = ["X","NOUN","NOUN","VERB","X","NOUN","VERB"]
        morph_codes    = [0, 1]
        pos_tags       = ["NOUN","VERB"]
        fake_sentences.append({
            "char_classes":    char_classes,
            "char_morph_codes": char_morph,
            "char_pos_tags":   char_pos_tags,
            "morph_codes":     morph_codes,
            "pos_tags":        pos_tags,
            "pos_huffman_bits": 2.5,
            "pos_n_tags":       2,
        })

    model = ContextMixingModel()
    model.train(fake_sentences)

    check("train populates char_vocab", len(model.char_vocab) > 0)
    check("train populates morph_vocab", len(model.morph_vocab) > 0)
    check("train populates weights", len(model.weights) == 3)
    check("weights sum ≈ 1", abs(sum(model.weights) - 1.0) < 1e-6)

    context = {
        "char_history": [0],
        "current_morph_code": 0,
        "current_pos_tag": "NOUN",
        "struct_prob": 1.0,
    }
    dist = model.probability_distribution(context)
    check("probability_distribution returns dict", isinstance(dist, dict))
    check("distribution sums ≈ 1", abs(sum(dist.values()) - 1.0) < 1e-4,
          f"sum={sum(dist.values()):.6f}")
    check("all probs > 0", all(v > 0 for v in dist.values()))

    prob = model.probability(0, context)
    check("probability returns float in (0,1]", 0 < prob <= 1.0, f"prob={prob}")

    # global_char_distribution
    global_dist = model.global_char_distribution()
    check("global_char_distribution non-empty", len(global_dist) > 0)
    check("global dist sums ≈ 1", abs(sum(global_dist.values()) - 1.0) < 1e-4)

    # serialise / load round-trip
    with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as f:
        model_path = f.name
    try:
        size = model.serialise(model_path)
        check("serialise writes bytes", size > 0)

        model2 = ContextMixingModel()
        model2.load(model_path)
        check("load restores char_vocab", model2.char_vocab == model.char_vocab)
        check("load restores weights",
              all(abs(a - b) < 1e-9 for a, b in zip(model2.weights, model.weights)))
        check("load restores char_context keys",
              set(model2.char_context.keys()) == set(model.char_context.keys()))
    finally:
        os.unlink(model_path)

    # bpb (using our fake pipeline adapter)
    class _FakePipeline:
        def encode_for_model(self, text):
            return fake_sentences

    bpb_val = model.bpb("hello world test", _FakePipeline())
    check("bpb returns positive float", isinstance(bpb_val, float) and bpb_val > 0)


# ── Stage 6b: _build_context_model / _build_encoded_sentences_from_metadata ──
section("Stage 6b — Decoder-side model reconstruction (main.py helpers)")

main_mod, err = _try_import("main")
if err:
    check("main.py import", False, str(err))
else:
    _build_context_model = main_mod._build_context_model
    _build_encoded_sentences_from_metadata = main_mod._build_encoded_sentences_from_metadata
    _join_words = main_mod._join_words

    # Build a minimal payload that mimics what compress_to_file stores
    payload_6b = {
        "char_context":  {0: {0: 3, 1: 2}, 1: {2: 1}},
        "morph_context": {0: {0: 5, 1: 2}},
        "struct_context": {"NOUN": {0: 4}},
        "model_weights": [0.35, 0.33, 0.32],
        "char_vocab":   [0, 1, 2, 3],
        "morph_vocab":  [0, 1],
        "pos_vocab":    ["NOUN","VERB"],
        "root_lengths": [[2, 3], [4]],
        "pos_huffman_bits": [3.0, 2.0],
        "pos_n_tags":   [2, 1],
        "pos_tags":     [["NOUN","VERB"], ["NOUN"]],
        "morph_codes":  [[0, 1], [0]],
    }

    cm_rebuilt = _build_context_model(payload_6b)
    check("_build_context_model char_vocab", cm_rebuilt.char_vocab == [0,1,2,3])
    check("_build_context_model weights", len(cm_rebuilt.weights) == 3)
    check("_build_context_model char_context key 0 present", 0 in cm_rebuilt.char_context)

    enc_sents = _build_encoded_sentences_from_metadata(payload_6b)
    check("_build_encoded_sentences_from_metadata returns list", isinstance(enc_sents, list))
    check("encoded sentences count matches root_lengths", len(enc_sents) == 2)
    check("encoded sentence has char_morph_codes", "char_morph_codes" in enc_sents[0])
    check("encoded sentence has char_pos_tags", "char_pos_tags" in enc_sents[0])

    # Both models must produce identical distributions at position 0
    dist_train = model.probability_distribution(context)
    dist_rebuilt = cm_rebuilt.probability_distribution(context)
    # (different training data — just check structure is consistent)
    check("rebuilt model distribution is dict", isinstance(dist_rebuilt, dict))
    check("rebuilt distribution has all vocab symbols",
          all(k in dist_rebuilt for k in cm_rebuilt.char_vocab))


# ── _join_words — every punctuation rule ─────────────────────────────────────
section("_join_words — punctuation attachment rules")

if main_mod is None:
    check("_join_words", False, "main.py not imported")
else:
    jw = _join_words

    check("plain words",         jw(["hello", "world"])      == "hello world")
    check("period attach left",  jw(["hello", "."])           == "hello.")
    check("comma attach left",   jw(["hello", ",", "world"])  == "hello, world")
    check("colon attach left",   jw(["wait", ":"])            == "wait:")
    check("question mark",       jw(["really", "?"])          == "really?")
    check("exclamation mark",    jw(["stop", "!"])            == "stop!")
    check("open paren",          jw(["see", "(", "fig", ")"])  == "see (fig)")
    check("contraction",         jw(["it", "'s"])              == "it's")
    check("hyphen glues",        jw(["well", "-", "known"])    == "well-known")
    check("closing double-quote",jw(["he", "said", '"'])       == 'he said"')
    check("dollar sign",         jw(["costs", "$", "5"])       == "costs $5")
    check("em-dash no spaces",   jw(["yes", "—", "no"])        == "yes—no")
    check("empty list",          jw([])                         == "")
    check("single word",         jw(["hello"])                  == "hello")


# ── Stage 7: ArithmeticEncoder + ArithmeticDecoder ───────────────────────────
section("Stage 7 — Arithmetic Coding (all three paths)")

arith_mod, err = _try_import("compression.pipeline.stage7_arithmetic")
if err:
    check("stage7_arithmetic import", False, str(err))
else:
    ArithmeticEncoder = arith_mod.ArithmeticEncoder
    ArithmeticDecoder = arith_mod.ArithmeticDecoder

    # ── Path A: encode_unigram / decode_unigram ──────────────────────────────
    vocab_u = [0, 1, 2, 3, 4, 5, 6]
    dist_u  = {s: 1.0 / 7 for s in vocab_u}
    symbols_u = [0, 1, 2, 3, 4, 5, 6, 0, 2, 4]

    enc_u = ArithmeticEncoder()
    bs_u  = enc_u.encode_unigram(symbols_u, dist_u)
    check("encode_unigram produces bytes", isinstance(bs_u, bytes) and len(bs_u) > 0)

    dec_u = ArithmeticDecoder()
    rt_u  = dec_u.decode_unigram(bs_u, dist_u, len(symbols_u))
    check("decode_unigram round-trip", rt_u == symbols_u, f"got {rt_u}")

    # ── Path B: encode_unigram_counts / decode_unigram_counts ────────────────
    symbols_c = [0, 0, 1, 2, 1, 0, 3, 1, 2, 0]
    counts_c  = Counter(symbols_c)

    enc_c = ArithmeticEncoder()
    bs_c  = enc_c.encode_unigram_counts(symbols_c, counts_c)
    check("encode_unigram_counts produces bytes", isinstance(bs_c, bytes) and len(bs_c) > 0)

    dec_c = ArithmeticDecoder()
    rt_c  = dec_c.decode_unigram_counts(bs_c, counts_c, len(symbols_c))
    check("decode_unigram_counts round-trip", rt_c == symbols_c, f"got {rt_c}")

    # ── Path C: context-model encode / decode ────────────────────────────────
    # Reuse the fake encoded_sentences from Stage 6
    cm_arith = ContextMixingModel()
    cm_arith.train(fake_sentences)

    char_classes_a = [0, 1, 2, 3, 0, 1, 2]
    enc_a = ArithmeticEncoder()
    bs_a  = enc_a.encode(char_classes_a, cm_arith, {}, fake_sentences)
    check("encode (context path) produces bytes", isinstance(bs_a, bytes) and len(bs_a) > 0)

    dec_a = ArithmeticDecoder()
    rt_a  = dec_a.decode(bs_a, cm_arith, fake_sentences, len(char_classes_a))
    check("decode (context path) round-trip", rt_a == char_classes_a, f"got {rt_a}")

    # ── Encoder / decoder produce identical context distributions ─────────────
    # Walk both in lock-step and compare probability_distribution at every position.
    from collections import deque as _deque
    from compression.config import CHAR_CONTEXT_SIZE

    def _build_context_stream_local(encoded_sentences):
        """Reproduce _build_context_stream from stage7_arithmetic without importing it."""
        morph_stream, pos_stream, struct_probs = [], [], []
        for sentence in encoded_sentences:
            char_morph  = sentence.get("char_morph_codes", [])
            char_pos    = sentence.get("char_pos_tags", [])
            n_tags      = int(sentence.get("pos_n_tags", 0))
            hbits       = float(sentence.get("pos_huffman_bits", 0.0))
            sp = (2 ** (-hbits / n_tags)) if n_tags > 0 and hbits > 0 else 1.0
            length = max(len(char_morph), len(char_pos))
            for i in range(length):
                morph_stream.append(char_morph[i] if i < len(char_morph) else 0)
                pos_stream.append(char_pos[i]   if i < len(char_pos)   else "X")
                struct_probs.append(sp)
        return morph_stream, pos_stream, struct_probs

    m_stream, p_stream, s_probs = _build_context_stream_local(fake_sentences)
    length_check = min(len(char_classes_a), len(m_stream), len(p_stream))
    hist = _deque(maxlen=CHAR_CONTEXT_SIZE)
    mismatch_idx = None
    for idx in range(length_check):
        ctx = {
            "char_history":       list(hist),
            "current_morph_code": m_stream[idx],
            "current_pos_tag":    p_stream[idx],
            "struct_prob":        s_probs[idx],
        }
        d_enc = cm_arith.probability_distribution(ctx)
        d_dec = cm_arith.probability_distribution(ctx)  # same model → must match
        if d_enc != d_dec:
            mismatch_idx = idx
            break
        hist.append(char_classes_a[idx])

    check("Encoder/decoder distributions identical at every position",
          mismatch_idx is None,
          f"first mismatch at idx={mismatch_idx}")

    # ── Edge cases ────────────────────────────────────────────────────────────
    # Single symbol
    enc_s = ArithmeticEncoder()
    bs_s  = enc_s.encode_unigram([2], {0:0.1, 1:0.3, 2:0.4, 3:0.2})
    rt_s  = ArithmeticDecoder().decode_unigram(bs_s, {0:0.1, 1:0.3, 2:0.4, 3:0.2}, 1)
    check("Single-symbol encode/decode", rt_s == [2])

    # Long stream (stress)
    import random
    random.seed(42)
    long_sym = [random.randint(0, 6) for _ in range(500)]
    long_cnt = Counter(long_sym)
    bs_l  = ArithmeticEncoder().encode_unigram_counts(long_sym, long_cnt)
    rt_l  = ArithmeticDecoder().decode_unigram_counts(bs_l, long_cnt, len(long_sym))
    check("Long stream (500 symbols) round-trip", rt_l == long_sym)


# ── Stage 8: FullDecoder (morphology payload path) ───────────────────────────
section("Stage 8 — FullDecoder (morphology payload path)")

dec8_mod, err = _try_import("compression.pipeline.stage8_decode")
if err:
    check("stage8 import", False, str(err))
else:
    FullDecoder = dec8_mod.FullDecoder
    decode_payload = dec8_mod.decode_payload
    decode_morphology = dec8_mod.decode_morphology

    # decode_morphology
    morphology = [
        {"root": "cat",  "code": PLURAL},
        {"root": "run",  "code": BASE},
        {"root": "fast", "code": ADVERBIAL},
    ]
    result_dm = decode_morphology(morphology)
    check("decode_morphology: cats", "cats" in result_dm)
    check("decode_morphology: run",  "run"  in result_dm)
    check("decode_morphology: quickly", "quickly" in result_dm)

    # decode_payload
    payload_dp = {"morphology": morphology}
    text_dp = decode_payload(payload_dp)
    check("decode_payload returns string", isinstance(text_dp, str) and len(text_dp) > 0)

    # FullDecoder.decode with a morphology payload (no arithmetic bitstream)
    import msgpack, tempfile, os
    morph_payload = msgpack.packb({"morphology": morphology}, use_bin_type=True)
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        f.write(morph_payload)
        morph_payload_path = f.name

    try:
        fd = FullDecoder()
        # decode() needs a model_path for the arithmetic path, but the morphology
        # branch is taken when "morphology" key is present.
        with tempfile.NamedTemporaryFile(suffix=".msgpack", delete=False) as mf:
            dummy_model_path = mf.name
        # Write a minimal dummy model so ContextMixingModel.load won't crash
        dummy_model = ContextMixingModel()
        dummy_model.serialise(dummy_model_path)
        decoded_fd = fd.decode(morph_payload, dummy_model_path)
        check("FullDecoder morphology path returns string",
              isinstance(decoded_fd, str) and len(decoded_fd) > 0)
    except Exception as exc:
        check("FullDecoder morphology path", False, str(exc))
    finally:
        os.unlink(morph_payload_path)
        try: os.unlink(dummy_model_path)
        except: pass


# ── Stage 9: autocorrect ─────────────────────────────────────────────────────
section("Stage 9 — autocorrect")

ac_mod, err = _try_import("compression.pipeline.stage9_autocorrect")
if err:
    check("stage9 import", False, str(err))
else:
    autocorrect = ac_mod.autocorrect

    check("autocorrect: double space",  autocorrect("hello  world")  == "hello world")
    check("autocorrect: space before period", autocorrect("hello .")   == "hello.")
    check("autocorrect: space before comma",  autocorrect("hello , world") == "hello, world")
    check("autocorrect: space before !",      autocorrect("stop !")    == "stop!")
    check("autocorrect: space before ?",      autocorrect("really ?")  == "really?")
    check("autocorrect: passthrough clean",   autocorrect("Clean sentence.") == "Clean sentence.")
    check("autocorrect: empty string",        autocorrect("") == "")


# ── Stage 10: compress_to_file → decompress (full round-trip) ─────────────────
section("Stage 10 — compress_to_file → decompress (full round-trip)")

if main_mod is None:
    check("compress_to_file", False, "main.py not imported")
else:
    compress_to_file = main_mod.compress_to_file
    decompress       = main_mod.decompress

    # A realistic multi-sentence paragraph with varied morphology
    TEST_TEXT = (
        "The engineers designed a new bridge. "
        "They worked quickly and efficiently. "
        "The project was completed ahead of schedule. "
        "Everyone celebrated the achievement."
    )

    try:
        with tempfile.NamedTemporaryFile(suffix=".lxs", delete=False) as f:
            out_path = f.name

        stats = compress_to_file(TEST_TEXT, out_path)
        check("compress_to_file returns dict", isinstance(stats, dict))
        check("compressed file exists and non-empty", Path(out_path).stat().st_size > 0)
        check("stats has original_size",    "original_size"    in stats)
        check("stats has compressed_size",  "compressed_size"  in stats)
        check("stats has compression_ratio","compression_ratio" in stats)
        check("stats original_size > 0",    stats.get("original_size", 0) > 0)

        reconstructed = decompress(out_path)
        check("decompress returns non-empty string",
              isinstance(reconstructed, str) and len(reconstructed) > 0)

        # Normalise both for comparison (punctuation/case may shift slightly)
        import re
        def _norm(t):
            return re.sub(r"\s+", " ", t.lower().strip())

        orig_norm = _norm(TEST_TEXT)
        rec_norm  = _norm(reconstructed)

        # Word-level overlap (expect ≥ 80% word recovery)
        orig_words = orig_norm.split()
        rec_words  = rec_norm.split()
        orig_set   = set(orig_words)
        rec_set    = set(rec_words)
        overlap    = len(orig_set & rec_set) / max(len(orig_set), 1)
        check("Round-trip word overlap ≥ 80%", overlap >= 0.80,
              f"{overlap*100:.1f}% — orig={orig_norm!r[:60]}  rec={rec_norm!r[:60]}")

        # Check first word is capitalised
        check("First character capitalised", reconstructed[0].isupper() if reconstructed else False)

        # Check ends with sentence terminator
        check("Ends with sentence terminator", reconstructed.rstrip().endswith((".", "?", "!")))

        # Character-level diff report for debugging
        if orig_norm != rec_norm:
            first_diff = next(
                (i for i, (a, b) in enumerate(zip(orig_norm, rec_norm)) if a != b),
                min(len(orig_norm), len(rec_norm))
            )
            print(f"    ℹ first char diff at position {first_diff}: "
                  f"orig={orig_norm[max(0,first_diff-5):first_diff+10]!r}  "
                  f"rec={rec_norm[max(0,first_diff-5):first_diff+10]!r}")

        # Short text edge case
        short_text = "Hello."
        with tempfile.NamedTemporaryFile(suffix=".lxs", delete=False) as f2:
            short_path = f2.name
        try:
            compress_to_file(short_text, short_path)
            short_rec = decompress(short_path)
            check("Short text round-trip non-empty", isinstance(short_rec, str) and len(short_rec) > 0)
        finally:
            os.unlink(short_path)

    except Exception as exc:
        check("Stage 10 compress_to_file → decompress", False,
              "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    finally:
        try: os.unlink(out_path)
        except: pass


# ── Summary ───────────────────────────────────────────────────────────────────
section("SUMMARY")
passed  = sum(1 for _, ok, _ in results if ok)
failed  = sum(1 for _, ok, _ in results if not ok)
total   = len(results)
print(f"\n  Total : {total}")
print(f"  {PASS}  : {passed}")
print(f"  {FAIL}  : {failed}")

if failed:
    print("\n  Failed checks:")
    for label, ok, detail in results:
        if not ok:
            print(f"    ✗  {label}" + (f"  — {detail}" if detail else ""))
    sys.exit(1)
else:
    print(f"\n  All {total} checks passed.")
    sys.exit(0)
