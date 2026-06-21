
"""
Bengali (বাংলা) language pipeline tests for Lexis.

These tests validate that the Lexis pipeline can handle Bengali Unicode text
correctly through normalisation, character encoding, and round-trip fidelity.

Key differences from English:
  - Bengali script (U+0980–U+09FF): vowels, consonants, matras, conjuncts.
  - No concept of "lowercase" — the pipeline's case-fold logic must be bypassed.
  - spaCy has no 'bn_core_news_*' model yet in the public registry; all tests
    run the rule-based / fallback path (use_spacy=False).
  - The phonetic_map assigns (6, 4) to any character outside its ASCII table.
    Bengali codepoints therefore all land in class 6, which is fine — the
    encoder / decoder round-trip must still be lossless.

Run:
    pytest tests/test_bengali_pipeline.py -v
  or
    python tests/test_bengali_pipeline.py
"""

from __future__ import annotations

import re
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
from compression.alphabet.phonetic_map import get_coords, PHONETIC_CLASSES

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
# Sample Bengali sentences
# ---------------------------------------------------------------------------

# A short paragraph about Rabindranath Tagore, written in natural Bengali.
BENGALI_SAMPLE = """\
রবীন্দ্রনাথ ঠাকুর একজন বিশিষ্ট বাঙালি কবি, ঔপন্যাসিক ও দার্শনিক ছিলেন।
তিনি ১৮৬১ সালে কলকাতায় জন্মগ্রহণ করেন এবং ১৯৪১ সালে মৃত্যুবরণ করেন।
তাঁর রচিত গীতাঞ্জলি কাব্যগ্রন্থের জন্য তিনি ১৯১৩ সালে নোবেল পুরস্কার লাভ করেন।
আমাদের জাতীয় সংগীত 'আমার সোনার বাংলা' তাঁর লেখা।
"""

# A minimal single-sentence test — useful for quick smoke tests.
BENGALI_SENTENCE = "আমি বাংলায় গান গাই।"

# A mixed text: Bengali script + ASCII digits + punctuation.
BENGALI_MIXED = "বাংলাদেশের জনসংখ্যা প্রায় ১৭ কোটি (170 million)।"


# ---------------------------------------------------------------------------
# Test 1: normalize_text preserves Bengali codepoints untouched
# ---------------------------------------------------------------------------
def test_normalize_preserves_bengali() -> None:
    print("\n=== Test 1: normalize_text preserves Bengali ===")

    normalized = normalize_text(BENGALI_SAMPLE)

    # Must not be empty
    check_true("non-empty after normalization", bool(normalized))

    # All Bengali codepoints must survive (U+0980–U+09FF)
    original_bengali = [c for c in BENGALI_SAMPLE if "\u0980" <= c <= "\u09ff"]
    normalized_bengali = [c for c in normalized if "\u0980" <= c <= "\u09ff"]
    check(
        "all Bengali codepoints preserved",
        len(normalized_bengali),
        len(original_bengali),
    )

    # ASCII punctuation normalisation must not touch Bengali content
    check_true(
        "no unexpected replacement artefacts",
        "\ufeff" not in normalized,  # BOM stripped
    )


# ---------------------------------------------------------------------------
# Test 2: MorphologicalAnalyser rule-based fallback on Bengali
#         All Bengali tokens should round-trip as BASE (code 0) because
#         none of the ASCII-specific suffix rules fire.
# ---------------------------------------------------------------------------
def test_morphology_bengali_base() -> None:
    print("\n=== Test 2: morphology rule-based fallback (Bengali → BASE) ===")

    from compression.alphabet.morph_codes import BASE

    analyser = MorphologicalAnalyser(use_spacy=False)

    for word in ["বাংলা", "কবি", "রবীন্দ্রনাথ", "গীতাঞ্জলি", "১৯১৩"]:
        root, code = analyser.analyse(word)
        # Root should be the word unchanged (no ASCII suffix stripping applies).
        check(f"root unchanged for '{word}'", root, word)
        check(f"code == BASE for '{word}'", code, BASE)


# ---------------------------------------------------------------------------
# Test 3: get_coords falls back gracefully for Bengali codepoints
#         Every Bengali character must map to (6, 4) — the unknown-char slot.
# ---------------------------------------------------------------------------
def test_phonetic_map_bengali_fallback() -> None:
    print("\n=== Test 3: phonetic_map Bengali codepoints → (6, 4) ===")

    UNKNOWN_COORD = (6, 4)

    bengali_chars = [c for c in "বাংলাদেশ" if "\u0980" <= c <= "\u09ff"]
    check_true("test has Bengali chars to check", len(bengali_chars) > 0)

    all_unknown = all(get_coords(ch) == UNKNOWN_COORD for ch in bengali_chars)
    check_true("all Bengali codepoints map to (6, 4)", all_unknown)

    # ASCII characters must NOT map to (6, 4) — regression guard.
    ascii_sample = "abcxyz0123"
    no_false_unknowns = all(
        get_coords(ch) != UNKNOWN_COORD for ch in ascii_sample
    )
    check_true("ASCII chars do NOT map to (6, 4)", no_false_unknowns)


# ---------------------------------------------------------------------------
# Test 4: normalize_text + whitespace normalisation on Bengali
# ---------------------------------------------------------------------------
def test_normalize_whitespace_bengali() -> None:
    print("\n=== Test 4: whitespace normalisation on Bengali text ===")

    noisy = "আমি   বাংলায়\t\tগান   গাই।"
    result = normalize_text(noisy)

    # Multiple spaces / tabs must be collapsed to single space.
    check_true("no double spaces remain", "  " not in result)
    check_true("no tabs remain", "\t" not in result)

    # Content must be preserved.
    for word in ["আমি", "বাংলায়", "গান", "গাই।"]:
        check_true(f"'{word}' present after normalization", word in result)


# ---------------------------------------------------------------------------
# Test 5: sentence-level morphology analysis on Bengali
#         Validates that analyse_sentence() does not crash on non-ASCII input
#         and returns one result per whitespace-delimited token.
# ---------------------------------------------------------------------------
def test_sentence_morphology_bengali() -> None:
    print("\n=== Test 5: analyse_sentence on Bengali ===")

    analyser = MorphologicalAnalyser(use_spacy=False)
    sentence = "রবীন্দ্রনাথ ঠাকুর বিখ্যাত কবি ছিলেন।"
    results = analyser.analyse_sentence(sentence)

    # Should return one entry per whitespace-delimited token.
    tokens = sentence.split()
    check(
        "one result per token",
        len(results),
        len(tokens),
    )

    # Each result is a 3-tuple (original, root, code).
    check_true(
        "results are 3-tuples",
        all(len(r) == 3 for r in results),
    )

    # All originals should match the split tokens exactly.
    originals = [r[0] for r in results]
    check("originals match tokens", originals, tokens)


# ---------------------------------------------------------------------------
# Test 6: mixed Bengali + ASCII round-trip through normalize_text
# ---------------------------------------------------------------------------
def test_normalize_mixed_bengali_ascii() -> None:
    print("\n=== Test 6: mixed Bengali + ASCII normalisation ===")

    result = normalize_text(BENGALI_MIXED)
    check_true("non-empty result", bool(result))

    # Digits must survive.
    check_true("digit '১৭' preserved", "১৭" in result)
    check_true("digit '170' preserved", "170" in result)

    # Bengali words must survive.
    check_true("'বাংলাদেশের' preserved", "বাংলাদেশের" in result)

    # No BOM or stray Unicode replacement characters.
    check_true("no BOM (U+FEFF)", "\ufeff" not in result)
    check_true("no replacement char (U+FFFD)", "\ufffd" not in result)


# ---------------------------------------------------------------------------
# Test 7: char_savings on Bengali text — regression guard
#         The metric should not crash. Savings may be 0 or positive.
# ---------------------------------------------------------------------------
def test_char_savings_bengali() -> None:
    print("\n=== Test 7: char_savings on Bengali text ===")

    analyser = MorphologicalAnalyser(use_spacy=False)
    stats = analyser.char_savings(BENGALI_SAMPLE)

    required_keys = {
        "original_chars",
        "root_chars",
        "chars_saved",
        "pct_saved",
    }
    check("all stat keys present", set(stats.keys()) & required_keys, required_keys)
    check_true("original_chars > 0", stats["original_chars"] > 0)
    check_true("pct_saved >= 0", stats["pct_saved"] >= 0.0)
    check_true("pct_saved <= 100", stats["pct_saved"] <= 100.0)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    test_normalize_preserves_bengali()
    test_morphology_bengali_base()
    test_phonetic_map_bengali_fallback()
    test_normalize_whitespace_bengali()
    test_sentence_morphology_bengali()
    test_normalize_mixed_bengali_ascii()
    test_char_savings_bengali()

    total = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed", end="")
    if failed:
        print(f"  ({failed} FAILED)")
        failed_names = [name for name, ok in _results if not ok]
        for name in failed_names:
            print(f"    ✗ {name}")
    else:
        print("  — all tests passed! 🎉")
    print("=" * 50)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
