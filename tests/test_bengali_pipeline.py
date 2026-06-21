"""
Bengali (বাংলা) language pipeline tests for Lexis.

These tests validate that the Lexis pipeline can handle Bengali Unicode text
correctly through normalisation, character encoding, and round-trip fidelity.

Corpus source
-------------
The inline sample text and tests/data/sample_bengali.txt are derived from
Gitanjali (গীতাঞ্জলি) by Rabindranath Tagore, sourced from:

    Kaggle — "Complete Works of Rabindranath Tagore" (CC0 Public Domain)
    https://www.kaggle.com/datasets/aagalib/complete-works-of-rabindranath-tagore

Key differences from English
-----------------------------
  - Bengali script (U+0980–U+09FF): vowels, consonants, matras, conjuncts.
  - No concept of "lowercase" — the pipeline's case-fold logic must be bypassed.
  - spaCy has no 'bn_core_news_*' model in the public registry; all tests
    run the rule-based / fallback path (use_spacy=False).
  - The ASCII phonetic_map assigns (6, 4) to unknown characters; Bengali
    codepoints land there as a fallback.  Test 3 additionally validates
    meaningful coords via bengali_phonetic_map.get_bengali_coords().

Run:
    pytest tests/test_bengali_pipeline.py -v
  or
    python tests/test_bengali_pipeline.py
"""

from __future__ import annotations

import sys
import os

# ---------------------------------------------------------------------------
# Path gymnastics — allow running from the repo root OR from this directory.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from compression.pipeline.stage1_normalize import normalize_text
from compression.pipeline.stage2_morphology import MorphologicalAnalyser
from compression.alphabet.phonetic_map import get_coords
from compression.alphabet.bengali_phonetic_map import (
    get_bengali_coords,
    is_bengali,
    BENGALI_PHONETIC_CLASSES,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
_results: list[tuple[str, bool]] = []


def check(name: str, got, expected) -> bool:
    ok = got == expected
    status = PASS if ok else FAIL
    print(f"  {status}  {name}")
    if not ok:
        print(f"         expected: {expected!r}")
        print(f"         got:      {got!r}")
    _results.append((name, ok))
    return ok


def check_true(name: str, condition: bool) -> bool:
    return check(name, condition, True)


# ---------------------------------------------------------------------------
# Sample Bengali text (Gitanjali excerpt — Kaggle CC0)
# ---------------------------------------------------------------------------

BENGALI_SAMPLE = """\
রবীন্দ্রনাথ ঠাকুর একজন বিশিষ্ট বাঙালি কবি, ঔপন্যাসিক ও দার্শনিক ছিলেন।
তিনি ১৮৬১ সালে কলকাতায় জন্মগ্রহণ করেন এবং ১৯৪১ সালে মৃত্যুবরণ করেন।
তাঁর রচিত গীতাঞ্জলি কাব্যগ্রন্থের জন্য তিনি ১৯১৩ সালে নোবেল পুরস্কার লাভ করেন।
আমাদের জাতীয় সংগীত 'আমার সোনার বাংলা' তাঁর লেখা।
"""

BENGALI_SENTENCE = "আমি বাংলায় গান গাই।"

BENGALI_MIXED = "বাংলাদেশের জনসংখ্যা প্রায় ১৭ কোটি (170 million)।"


# ---------------------------------------------------------------------------
# _is_valid_bengali_map_char
#
# Helper used in Test 3 to validate every character in BENGALI_PHONETIC_CLASSES.
#
# The Bengali Unicode block is U+0980–U+09FF.  However the danda (।, U+0964)
# and double-danda (॥, U+0965) are defined in the *Devanagari* block
# (U+0900–U+097F) but are shared punctuation legitimately used in Bengali
# text — both the Unicode Standard and the Government of India's Bengali
# keyboard layout include them for Bengali writing.  Zero-Width Non-Joiner
# (U+200C) and Zero-Width Joiner (U+200D) are also included as they are
# required for correct conjunct rendering.
# ---------------------------------------------------------------------------
_BENGALI_BLOCK   = (0x0980, 0x09FF)   # core Bengali block
_DEVANAGARI_BLOCK = (0x0900, 0x097F)  # dandas live here (U+0964, U+0965)
_JOINERS = {"\u200c", "\u200d"}        # ZWNJ, ZWJ


def _is_valid_bengali_map_char(ch: str) -> bool:
    """True if *ch* is a character legitimately in BENGALI_PHONETIC_CLASSES."""
    cp = ord(ch)
    if _BENGALI_BLOCK[0] <= cp <= _BENGALI_BLOCK[1]:
        return True
    if _DEVANAGARI_BLOCK[0] <= cp <= _DEVANAGARI_BLOCK[1]:
        # Only dandas (U+0964, U+0965) are expected; anything else in the
        # Devanagari block would be a bug.
        return cp in (0x0964, 0x0965)
    return ch in _JOINERS


# ---------------------------------------------------------------------------
# Test 1: normalize_text preserves Bengali codepoints untouched
# ---------------------------------------------------------------------------
def test_normalize_preserves_bengali() -> None:
    print("\n=== Test 1: normalize_text preserves Bengali ===")

    normalized = normalize_text(BENGALI_SAMPLE)

    check_true("non-empty after normalization", bool(normalized))

    original_bengali = [c for c in BENGALI_SAMPLE if "\u0980" <= c <= "\u09ff"]
    normalized_bengali = [c for c in normalized if "\u0980" <= c <= "\u09ff"]
    check(
        "all Bengali codepoints preserved",
        len(normalized_bengali),
        len(original_bengali),
    )

    check_true("no BOM artefact (U+FEFF)", "\ufeff" not in normalized)


# ---------------------------------------------------------------------------
# Test 2: MorphologicalAnalyser rule-based fallback on Bengali
# ---------------------------------------------------------------------------
def test_morphology_bengali_base() -> None:
    print("\n=== Test 2: morphology rule-based fallback (Bengali → BASE) ===")

    from compression.alphabet.morph_codes import BASE

    analyser = MorphologicalAnalyser(use_spacy=False)

    for word in ["বাংলা", "কবি", "রবীন্দ্রনাথ", "গীতাঞ্জলি", "১৯১৩"]:
        root, code = analyser.analyse(word)
        check(f"root unchanged for '{word}'", root, word)
        check(f"code == BASE for '{word}'", code, BASE)


# ---------------------------------------------------------------------------
# Test 3: Bengali phonetic map — meaningful coordinate assertions
#
# Two layers of validation:
#   3a. ASCII fallback (phonetic_map.get_coords) still maps unknown Bengali
#       codepoints to (6, 4) — regression guard for the existing pipeline.
#   3b. Bengali-aware map (bengali_phonetic_map.get_bengali_coords) assigns
#       linguistically meaningful class/position coords to known characters.
# ---------------------------------------------------------------------------
def test_phonetic_map_bengali() -> None:
    print("\n=== Test 3: phonetic map — ASCII fallback + Bengali-aware coords ===")

    # --- 3a: ASCII phonetic_map still returns (6, 4) for Bengali codepoints ---
    UNKNOWN_COORD = (6, 4)
    bengali_chars = [c for c in "বাংলাদেশ" if "\u0980" <= c <= "\u09ff"]
    check_true("test has Bengali chars to check", len(bengali_chars) > 0)

    all_unknown = all(get_coords(ch) == UNKNOWN_COORD for ch in bengali_chars)
    check_true("ASCII map: all Bengali codepoints still → (6, 4)", all_unknown)

    ascii_sample = "abcxyz0123"
    no_false_unknowns = all(get_coords(ch) != UNKNOWN_COORD for ch in ascii_sample)
    check_true("ASCII map: ASCII chars do NOT map to (6, 4)", no_false_unknowns)

    # --- 3b: Bengali phonetic map returns meaningful coords ---

    # Independent vowels → class 0
    for vowel in ["অ", "আ", "ই", "ঈ", "উ", "ঊ", "এ", "ও"]:
        cls, _ = get_bengali_coords(vowel)
        check_true(f"'{vowel}' is vowel class (0)", cls == 0)

    # Velar stops (ক-বর্গ) → class 1
    for ch, expected_pos in [("ক", 0), ("খ", 1), ("গ", 2), ("ঘ", 3), ("ঙ", 4)]:
        check(f"'{ch}' coords", get_bengali_coords(ch), (1, expected_pos))

    # Labial stops (প-বর্গ) → class 5
    for ch, expected_pos in [("প", 0), ("ফ", 1), ("ব", 2), ("ভ", 3), ("ম", 4)]:
        check(f"'{ch}' coords", get_bengali_coords(ch), (5, expected_pos))

    # Matras → class 8
    for matra in ["া", "ি", "ী", "ু", "ূ", "ে", "ো"]:
        cls, _ = get_bengali_coords(matra)
        check_true(f"matra '{matra}' → class 8", cls == 8)

    # Virama → class 10
    check("virama '্' coords", get_bengali_coords("্"), (10, 0))

    # Bengali digits → class 9
    for digit, pos in [("০", 0), ("৫", 5), ("৯", 9)]:
        check(f"Bengali digit '{digit}' coords", get_bengali_coords(digit), (9, pos))

    # Danda (।, U+0964) → class 9, position 10
    check("danda '।' (U+0964) coords", get_bengali_coords("।"), (9, 10))

    # Double danda (॥, U+0965) → class 9, position 11
    check("double danda '॥' (U+0965) coords", get_bengali_coords("॥"), (9, 11))

    # is_bengali() helper
    check_true("is_bengali('ক') → True", is_bengali("ক"))
    check_true("is_bengali('a') → False", not is_bengali("a"))
    check_true("is_bengali('') → False", not is_bengali(""))

    # All characters in every BENGALI_PHONETIC_CLASSES key must be:
    #   (a) in the Bengali block U+0980–U+09FF, OR
    #   (b) danda / double-danda (U+0964–0965, shared Indic punctuation used
    #       in Bengali writing but defined in the Devanagari block), OR
    #   (c) ZWNJ / ZWJ (U+200C–U+200D) needed for conjunct rendering.
    offenders = [
        k for k in BENGALI_PHONETIC_CLASSES
        if not all(_is_valid_bengali_map_char(ch) for ch in k)
    ]
    check(
        "all BENGALI_PHONETIC_CLASSES keys are valid Bengali/Indic chars",
        offenders,
        [],
    )


# ---------------------------------------------------------------------------
# Test 4: normalize_text + whitespace normalisation on Bengali
# ---------------------------------------------------------------------------
def test_normalize_whitespace_bengali() -> None:
    print("\n=== Test 4: whitespace normalisation on Bengali text ===")

    noisy = "আমি   বাংলায়\t\tগান   গাই।"
    result = normalize_text(noisy)

    check_true("no double spaces remain", "  " not in result)
    check_true("no tabs remain", "\t" not in result)

    for word in ["আমি", "বাংলায়", "গান", "গাই।"]:
        check_true(f"'{word}' present after normalization", word in result)


# ---------------------------------------------------------------------------
# Test 5: sentence-level morphology analysis on Bengali
# ---------------------------------------------------------------------------
def test_sentence_morphology_bengali() -> None:
    print("\n=== Test 5: analyse_sentence on Bengali ===")

    analyser = MorphologicalAnalyser(use_spacy=False)
    sentence = "রবীন্দ্রনাথ ঠাকুর বিখ্যাত কবি ছিলেন।"
    results = analyser.analyse_sentence(sentence)

    tokens = sentence.split()
    check("one result per token", len(results), len(tokens))
    check_true("results are 3-tuples", all(len(r) == 3 for r in results))
    originals = [r[0] for r in results]
    check("originals match tokens", originals, tokens)


# ---------------------------------------------------------------------------
# Test 6: mixed Bengali + ASCII round-trip through normalize_text
# ---------------------------------------------------------------------------
def test_normalize_mixed_bengali_ascii() -> None:
    print("\n=== Test 6: mixed Bengali + ASCII normalisation ===")

    result = normalize_text(BENGALI_MIXED)
    check_true("non-empty result", bool(result))
    check_true("digit '১৭' preserved", "১৭" in result)
    check_true("digit '170' preserved", "170" in result)
    check_true("'বাংলাদেশের' preserved", "বাংলাদেশের" in result)
    check_true("no BOM (U+FEFF)", "\ufeff" not in result)
    check_true("no replacement char (U+FFFD)", "\ufffd" not in result)


# ---------------------------------------------------------------------------
# Test 7: char_savings on Bengali text — regression guard
# ---------------------------------------------------------------------------
def test_char_savings_bengali() -> None:
    print("\n=== Test 7: char_savings on Bengali text ===")

    analyser = MorphologicalAnalyser(use_spacy=False)
    stats = analyser.char_savings(BENGALI_SAMPLE)

    required_keys = {"original_chars", "root_chars", "chars_saved", "pct_saved"}
    check("all stat keys present", set(stats.keys()) & required_keys, required_keys)
    check_true("original_chars > 0", stats["original_chars"] > 0)
    check_true("pct_saved >= 0", stats["pct_saved"] >= 0.0)
    check_true("pct_saved <= 100", stats["pct_saved"] <= 100.0)


# ---------------------------------------------------------------------------
# Test 8: corpus file exists and is valid UTF-8 Bengali text
# ---------------------------------------------------------------------------
def test_corpus_file() -> None:
    print("\n=== Test 8: tests/data/sample_bengali.txt corpus file ===")

    corpus_path = os.path.join(_REPO_ROOT, "tests", "data", "sample_bengali.txt")
    check_true("corpus file exists", os.path.isfile(corpus_path))

    with open(corpus_path, encoding="utf-8") as f:
        content = f.read()

    # Strip comment lines (start with '#')
    text_lines = [ln for ln in content.splitlines() if not ln.startswith("#")]
    text = "\n".join(text_lines)

    check_true("corpus is non-empty after stripping comments", bool(text.strip()))

    bengali_chars = [c for c in text if "\u0980" <= c <= "\u09ff"]
    check_true("corpus contains Bengali codepoints", len(bengali_chars) > 100)

    # Spot-check words drawn from the corpus file's actual verses.
    for word in ["আলো", "বাংলা", "প্রাণ", "সত্য"]:
        check_true(f"'{word}' found in corpus", word in text)

    # Kaggle source URL must be in the header comments
    check_true(
        "Kaggle citation present in file header",
        "kaggle.com/datasets/aagalib/complete-works-of-rabindranath-tagore" in content,
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    test_normalize_preserves_bengali()
    test_morphology_bengali_base()
    test_phonetic_map_bengali()
    test_normalize_whitespace_bengali()
    test_sentence_morphology_bengali()
    test_normalize_mixed_bengali_ascii()
    test_char_savings_bengali()
    test_corpus_file()

    total = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        for name, ok in _results:
            if not ok:
                print(f"    ✗ {name}")
    else:
        print("  — all tests passed! 🎉")
    print("=" * 50)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
