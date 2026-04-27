from compression.pipeline.stage5_encode import (
    CASE_LOWER,
    CASE_MIXED,
    compute_case_flag,
    apply_case_flag,
)


def test_compute_case_flag_non_alpha_is_lower():
    assert compute_case_flag("123") == (CASE_LOWER, 0)
    assert compute_case_flag("...") == (CASE_LOWER, 0)


def test_compute_case_flag_mixed_bitmap_uses_char_index_bits():
    # e(0) B(1) o(2) o(3) k(4)  -> only bit 1 set
    assert compute_case_flag("eBook") == (CASE_MIXED, 0b10)
    # i(0) P(1) h(2) o(3) n(4) e(5) -> only bit 1 set
    assert compute_case_flag("iPhone") == (CASE_MIXED, 0b10)


def test_apply_case_flag_mixed_bitmap_uses_char_index_bits():
    assert apply_case_flag("ebook", CASE_MIXED, 0b10) == "eBook"
    assert apply_case_flag("iphone", CASE_MIXED, 0b10) == "iPhone"
    assert apply_case_flag("abc", CASE_MIXED, 0b111) == "ABC"
    assert apply_case_flag("ab", CASE_MIXED, 0b01) == "Ab"
    assert apply_case_flag("ab", CASE_MIXED, 0b10) == "aB"
    assert apply_case_flag("abcdef", CASE_MIXED, 0b101010) == "aBcDeF"
    assert apply_case_flag("abcdef", CASE_MIXED, 0b010101) == "AbCdEf"
