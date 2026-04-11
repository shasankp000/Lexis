"""
Unit tests for metadata_codec round-trip correctness.
No spaCy, no pipeline — pure encode/decode verification.

Run:  python codec_unit_test.py
"""
from __future__ import annotations

from compression.metadata_codec import (
    _enc_sparse_dict, _dec_sparse_dict,
    _enc_model_weights, _dec_model_weights,
    _enc_flat_dict, _dec_flat_dict,
    _enc_symbol_table, _dec_symbol_table,
    encode_metadata, decode_metadata,
)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, got, expected):
    if got == expected:
        print(f"  {PASS}  {name}")
    else:
        print(f"  {FAIL}  {name}")
        print(f"         expected: {expected}")
        print(f"         got:      {got}")


# ---------------------------------------------------------------------------
# 1. sparse_dict round-trip with exact char_context from the failing run
# ---------------------------------------------------------------------------
def test_sparse_dict():
    print("\n=== sparse_dict round-trip ===")

    # Exact values from the debug output
    char_ctx = {
        5: {6: 749, 0: 44, 1: 19, 2: 99, 5: 84, 4: 6, 3: 4},
        6: {6: 312, 1: 116, 5: 847, 0: 96, 3: 27, 2: 59, 4: 19},
    }
    blob = _enc_sparse_dict(char_ctx)
    got  = _dec_sparse_dict(blob)
    check("class 5 exact",  got.get(5), {0:44, 1:19, 2:99, 3:4, 4:6, 5:84, 6:749})
    check("class 6 exact",  got.get(6), {0:96, 1:116, 2:59, 3:27, 4:19, 5:847, 6:312})

    # morph_context from failing run
    morph_ctx = {
        0: {6: 1623, 1: 491, 2: 330, 0: 715, 5: 996, 4: 233, 3: 119},
    }
    blob2 = _enc_sparse_dict(morph_ctx)
    got2  = _dec_sparse_dict(blob2)
    check("morph 0 exact", got2.get(0), {0:715, 1:491, 2:330, 3:119, 4:233, 5:996, 6:1623})

    # Larger synthetic test
    big = {
        i: {j: (i * 100 + j * 7 + 1) for j in range(7)}
        for i in range(7)
    }
    got_big = _dec_sparse_dict(_enc_sparse_dict(big))
    all_ok = all(got_big.get(i) == big[i] for i in range(7))
    check("7x7 synthetic", all_ok, True)


# ---------------------------------------------------------------------------
# 2. model_weights round-trip
# ---------------------------------------------------------------------------
def test_model_weights():
    print("\n=== model_weights round-trip ===")
    live = [0.579983, 0.233333, 0.186683]
    got  = _dec_model_weights(_enc_model_weights(live))
    # float32 has ~7 sig digits; check within 1e-6
    ok = all(abs(a - b) < 1e-6 for a, b in zip(live, got))
    check("float32 precision", ok, True)
    if not ok:
        print(f"         live={live}")
        print(f"         got ={got}")


# ---------------------------------------------------------------------------
# 3. symbol_table round-trip
# ---------------------------------------------------------------------------
def test_symbol_table():
    print("\n=== symbol_table round-trip ===")
    st = {"§E0": "Ishmael", "§E1": "Ahab", "§E2": "Moby Dick", "§R0": "pursues"}
    got = _dec_symbol_table(_enc_symbol_table(st))
    check("entity keys", got, st)


# ---------------------------------------------------------------------------
# 4. flat_dict round-trip with negative keys (pos_deltas_counts can be negative)
# ---------------------------------------------------------------------------
def test_flat_dict():
    print("\n=== flat_dict round-trip ===")
    d = {-3: 12, 0: 500, 1: 200, 2: 100, 5: 44}
    got = _dec_flat_dict(_enc_flat_dict(d))
    check("with negative key", got, d)


# ---------------------------------------------------------------------------
# 5. Full encode_metadata / decode_metadata round-trip on a minimal payload
# ---------------------------------------------------------------------------
def test_full_roundtrip():
    print("\n=== full encode_metadata / decode_metadata ===")
    meta = {
        "compressed_bitstream": bytes([0xAB, 0xCD, 0xEF]),
        "pos_deltas_bitstream":  bytes([0x01, 0x02]),
        "symbol_table":          {"§E0": "whale"},
        "pos_deltas_counts":     {-1: 5, 0: 100, 1: 80},
        "pos_deltas_count":      185,
        "num_symbols":           42,
        "num_char_classes":      7,
        "sentence_char_counts": [10, 20, 15],
        "pos_n_tags":            [3, 5],
        "char_vocab":            [0, 1, 2, 3, 4, 5, 6],
        "morph_vocab":           [0, 1, 2],
        "pos_huffman_bits":      [2.5, 3.1],
        "pos_tags":              [["NOUN", "VERB"], ["DET", "NOUN", "ADJ"]],
        "morph_codes":           [[0, 1], [2, 0, 1]],
        "root_lengths":          [[3, 4], [2, 5, 3]],
        "model_weights":         [0.579983, 0.233333, 0.186683],
        "char_context":  {5: {0:44, 1:19, 2:99, 3:4, 4:6, 5:84, 6:749},
                          6: {0:96, 1:116, 2:59, 3:27, 4:19, 5:847, 6:312}},
        "morph_context": {0: {0:715, 1:491, 2:330, 3:119, 4:233, 5:996, 6:1623}},
        "struct_context":{"NOUN": {0: 50, 1: 30}, "VERB": {1: 20, 2: 10}},
        "pos_vocab":             ["NOUN", "VERB", "ADJ"],
        "pos_freq_table":        {"NOUN": 120, "VERB": 80, "ADJ": 40},
    }

    binary = encode_metadata(meta)
    got    = decode_metadata(binary)

    check("compressed_bitstream", bytes(got["compressed_bitstream"]), meta["compressed_bitstream"])
    check("symbol_table",         got["symbol_table"],         meta["symbol_table"])
    check("pos_deltas_counts",    got["pos_deltas_counts"],    {int(k):v for k,v in meta["pos_deltas_counts"].items()})
    check("num_symbols",          got["num_symbols"],          meta["num_symbols"])
    check("sentence_char_counts", got["sentence_char_counts"], meta["sentence_char_counts"])
    check("pos_tags",             got["pos_tags"],             meta["pos_tags"])
    check("morph_codes",          got["morph_codes"],          meta["morph_codes"])
    check("root_lengths",         got["root_lengths"],         meta["root_lengths"])
    check("char_context[5]",      got["char_context"].get(5),  meta["char_context"][5])
    check("char_context[6]",      got["char_context"].get(6),  meta["char_context"][6])
    check("morph_context[0]",     got["morph_context"].get(0), meta["morph_context"][0])
    check("struct_context[NOUN]", got["struct_context"].get("NOUN"), meta["struct_context"]["NOUN"])
    wt_ok = all(abs(a-b) < 1e-6 for a,b in zip(got["model_weights"], meta["model_weights"]))
    check("model_weights precision", wt_ok, True)
    check("pos_vocab",  set(got["pos_vocab"]),  set(meta["pos_vocab"]))
    check("pos_freq_table", got["pos_freq_table"], meta["pos_freq_table"])


if __name__ == "__main__":
    test_sparse_dict()
    test_model_weights()
    test_symbol_table()
    test_flat_dict()
    test_full_roundtrip()
    print("\nDone.")
