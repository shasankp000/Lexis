"""Bengali-aware character encoder for the Lexis pipeline.

BengaliCharacterEncoder subclasses CharacterEncoder and overrides
coordinate lookup to use BENGALI_PHONETIC_CLASSES for Bengali codepoints
while keeping PHONETIC_CLASSES for ASCII and punctuation.  It also:

* Skips lower() on Bengali text in stats() (Bengali has no case).
* Returns zero case_flags / case_bitmaps for every token.
* Exposes bengali_inverse_map() — merged lookup for decompress().

BengaliJoiner.join() is a simple space-join that replaces the English
_join_words() logic; Bengali dandas are already attached to the
preceding token in the surface form.
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, List, Tuple

from compression.alphabet.bengali_phonetic_map import (
    BENGALI_PHONETIC_CLASSES,
    get_bengali_coords,
    is_bengali,
)
from compression.alphabet.morph_codes import BASE
from compression.alphabet.phonetic_map import (
    PHONETIC_CLASSES,
    char_to_triple as _en_char_to_triple,
)
from compression.pipeline.stage5_encode import (
    CharacterEncoder,
    StructuralEncoder,
    _char_classes_from_triples,
    _cumulative_from_deltas,
    _expand_morph_codes_for_chars,
    _expand_pos_tags_for_chars,
    _flat_char_id,
    _sequence_with_markers,
    compute_deltas,
    encode_factoradic,
)
from compression.pipeline.stage3_syntax import SyntaxResult

# ---------------------------------------------------------------------------
# Merged inverse map: Bengali + English/ASCII in one dict.
# ---------------------------------------------------------------------------
_MERGED_INVERSE: Dict[Tuple[int, int], str] = {
    coords: char for char, coords in PHONETIC_CLASSES.items()
}
_MERGED_INVERSE.update(
    {coords: char for char, coords in BENGALI_PHONETIC_CLASSES.items()}
)


def bengali_inverse_map() -> Dict[Tuple[int, int], str]:
    """Return a copy of the merged (English + Bengali) coords -> char map."""
    return dict(_MERGED_INVERSE)


# ---------------------------------------------------------------------------
# Bengali-aware triple builder
# ---------------------------------------------------------------------------

def _bengali_char_to_triple(
    char: str, pos: int, word_len: int
) -> Tuple[int, int, int]:
    """Map *char* to (class, position, role) using the correct phonetic map."""
    if is_bengali(char):
        cls, position = get_bengali_coords(char)
        return (cls, position, 0)
    # Fall back to English triple builder for ASCII / punctuation markers.
    return _en_char_to_triple(char, pos, word_len)


def _bengali_triples_for_sentence(
    words: List[str],
) -> List[Tuple[int, int, int]]:
    """Build the full triple stream for a Bengali sentence."""
    triples: List[Tuple[int, int, int]] = []
    for idx, word in enumerate(words):
        triples.append(_bengali_char_to_triple("^", 0, 1))
        word_len = len(word)
        for pos, char in enumerate(word):
            triples.append(_bengali_char_to_triple(char, pos, word_len))
        triples.append(_bengali_char_to_triple("$", 0, 1))
        if idx < len(words) - 1:
            triples.append(_bengali_char_to_triple("_", 0, 1))
    return triples


# ---------------------------------------------------------------------------
# BengaliCharacterEncoder
# ---------------------------------------------------------------------------

class BengaliCharacterEncoder(CharacterEncoder):
    """CharacterEncoder variant that handles Bengali Unicode correctly."""

    def __init__(self) -> None:
        super().__init__()
        # Override the inverse map to cover Bengali codepoints.
        self.inverse_map = _MERGED_INVERSE

    def encode_sentence(
        self, morphology_results: List[Tuple]
    ) -> Dict[str, List]:
        roots: List[str] = [
            item[1] if len(item) == 3 else item[0]
            for item in morphology_results
        ]
        triples = _bengali_triples_for_sentence(roots)
        class_deltas, pos_deltas, role_deltas = compute_deltas(triples)
        role_stream = [role for _, _, role in triples]
        return {
            "class_deltas": class_deltas,
            "pos_deltas": pos_deltas,
            "role_stream": role_stream,
            "factoriadic_class": [encode_factoradic(v) for v in class_deltas],
            "factoriadic_pos": [encode_factoradic(v) for v in pos_deltas],
            "role_deltas": role_deltas,
        }

    def encode_sentence_full(
        self,
        morphology_results: List[Tuple],
        syntax_result: SyntaxResult,
        structural_encoder: StructuralEncoder,
        freq_table: Dict[str, int],
    ) -> Dict[str, object]:
        roots: List[str] = []
        morph_codes: List[int] = []
        for item in morphology_results:
            if len(item) == 3:
                roots.append(item[1])
                morph_codes.append(int(item[2]))
            else:
                roots.append(item[0])
                morph_codes.append(BASE)

        char_encoding = self.encode_sentence(morphology_results)
        triples = _bengali_triples_for_sentence(roots)
        char_classes = _char_classes_from_triples(triples)
        char_morph_codes = _expand_morph_codes_for_chars(roots, morph_codes)
        char_pos_tags = _expand_pos_tags_for_chars(roots, syntax_result.pos_tags)
        pos_encoding = structural_encoder.encode_pos_sequence(
            syntax_result.pos_tags, freq_table
        )
        tree_shape_id = structural_encoder.encode_tree_shape(syntax_result.tree_shape)
        sentence_meta = structural_encoder.encode_sentence_meta(syntax_result)

        # Bengali has no case — emit zero flags for every root
        case_flags: List[int] = [0] * len(roots)
        case_bitmaps: List[int] = [0] * len(roots)

        return {
            **char_encoding,
            "roots": roots,
            "morph_codes": morph_codes,
            "pos_tags": syntax_result.pos_tags,
            "char_classes": char_classes,
            "char_morph_codes": char_morph_codes,
            "char_pos_tags": char_pos_tags,
            "pos_encoding": pos_encoding,
            "tree_shape_id": tree_shape_id,
            "sentence_meta": sentence_meta,
            "pos_huffman_bits": pos_encoding["pos_huffman_bits"],
            "pos_huffman_codes": pos_encoding["pos_huffman_codes"],
            "pos_n_tags": pos_encoding["tag_count"],
            "case_flags": case_flags,
            "case_bitmaps": case_bitmaps,
        }

    def decode_word(self, encoded: Dict[str, List[int]]) -> str:
        """Reconstruct word using the merged inverse map."""
        class_deltas = encoded.get("class_deltas", [])
        pos_deltas = encoded.get("pos_deltas", [])
        classes = _cumulative_from_deltas(class_deltas)
        positions = _cumulative_from_deltas(pos_deltas)
        return "".join(
            self.inverse_map.get((cls, pos), "?") for cls, pos in zip(classes, positions)
        )

    def stats(self, text: str) -> Dict[str, float]:
        """Stats for Bengali text — skip lower() since Bengali has no case."""
        words = [w for w in text.split() if w.strip()]
        if not words:
            return {
                "mean_class_delta": 0.0,
                "mean_pos_delta": 0.0,
                "mean_role_delta": 0.0,
                "mean_flat_delta": 0.0,
                "mean_decomp_magnitude": 0.0,
                "improvement_ratio": 0.0,
            }

        triples = _bengali_triples_for_sentence(words)
        class_deltas, pos_deltas, role_deltas = compute_deltas(triples)

        deltas_2d = [
            sqrt(dc * dc + dp * dp)
            for dc, dp in zip(class_deltas[1:], pos_deltas[1:])
        ]
        mean_decomp = sum(deltas_2d) / len(deltas_2d) if deltas_2d else 0.0
        mean_class = sum(abs(v) for v in class_deltas[1:]) / max(len(class_deltas) - 1, 1)
        mean_pos = sum(abs(v) for v in pos_deltas[1:]) / max(len(pos_deltas) - 1, 1)
        mean_role = sum(abs(v) for v in role_deltas[1:]) / max(len(role_deltas) - 1, 1)

        flat_sequence = _sequence_with_markers(words)
        flat_ids = [_flat_char_id(ch) for ch in flat_sequence]
        flat_deltas = [flat_ids[0]] + [b - a for a, b in zip(flat_ids, flat_ids[1:])]
        mean_flat = sum(abs(v) for v in flat_deltas[1:]) / max(len(flat_deltas) - 1, 1)

        improvement = (mean_flat / mean_decomp) if mean_decomp else 0.0
        return {
            "mean_class_delta": float(mean_class),
            "mean_pos_delta": float(mean_pos),
            "mean_role_delta": float(mean_role),
            "mean_flat_delta": float(mean_flat),
            "mean_decomp_magnitude": float(mean_decomp),
            "improvement_ratio": float(improvement),
        }


# ---------------------------------------------------------------------------
# BengaliJoiner — replaces _join_words for Bengali text
# ---------------------------------------------------------------------------

class BengaliJoiner:
    """Simple space-joiner for Bengali tokens.

    Bengali dandas (।॥) are already attached to the preceding token in
    the surface form, so no special punctuation logic is needed.
    """

    @staticmethod
    def join(words: List[str]) -> str:
        return " ".join(w for w in words if w)
