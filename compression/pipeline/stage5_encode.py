"""Character encoder integrating phonetic deltas and factoriadic encoding."""

from __future__ import annotations

from math import factorial, sqrt
from typing import Dict, List, Tuple, TypedDict

from compression.alphabet.morph_codes import BASE
from compression.alphabet.phonetic_map import (
    PHONETIC_CLASSES,
    PhoneticMap,
    char_to_triple,
    compute_deltas,
)
from compression.alphabet.symbol_alphabet import SymbolAlphabet
from compression.pipeline.stage3_syntax import SyntaxResult


def encode_factoradic(value: int) -> List[int]:
    """Encode a non-negative integer into factoriadic digits."""
    if value < 0:
        raise ValueError("value must be non-negative")
    if value == 0:
        return [0]
    digits: List[int] = []
    base = 1
    n = value
    while n > 0:
        n, rem = divmod(n, base)
        digits.append(rem)
        base += 1
    return list(reversed(digits))


def decode_factoradic(digits: List[int]) -> int:
    """Decode factoriadic digits into an integer."""
    if not digits:
        return 0
    total = 0
    length = len(digits)
    for idx, digit in enumerate(digits):
        total += digit * factorial(length - 1 - idx)
    return total


def _cumulative_from_deltas(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    values = [deltas[0]]
    for delta in deltas[1:]:
        values.append(values[-1] + delta)
    return values


def _flat_char_id(char: str) -> int:
    lower = char.lower()
    if "a" <= lower <= "z":
        return ord(lower) - ord("a")
    if lower == "^":
        return 50
    if lower == "$":
        return 51
    if lower == "_":
        return 52
    if lower == ".":
        return 53
    return 54


def _sequence_with_markers(words: List[str]) -> List[str]:
    sequence: List[str] = []
    for idx, word in enumerate(words):
        sequence.append("^")
        sequence.extend(list(word))
        sequence.append("$")
        if idx < len(words) - 1:
            sequence.append("_")
    return sequence


def _triples_for_sentence(words: List[str]) -> List[Tuple[int, int, int]]:
    triples: List[Tuple[int, int, int]] = []
    for idx, word in enumerate(words):
        triples.append(char_to_triple("^", 0, 1))
        word_length = len(word)
        for pos, char in enumerate(word):
            triples.append(char_to_triple(char, pos, word_length))
        triples.append(char_to_triple("$", 0, 1))
        if idx < len(words) - 1:
            triples.append(char_to_triple("_", 0, 1))
    return triples


def _char_classes_from_triples(triples: List[Tuple[int, int, int]]) -> List[int]:
    return [cls for cls, _, _ in triples]


def _expand_morph_codes_for_chars(
    roots: List[str], morph_codes: List[int]
) -> List[int]:
    stream: List[int] = []
    for idx, root in enumerate(roots):
        stream.append(BASE)  # start marker
        stream.extend([morph_codes[idx]] * len(root))
        stream.append(BASE)  # end marker
        if idx < len(roots) - 1:
            stream.append(BASE)  # whitespace marker
    return stream


def _expand_pos_tags_for_chars(roots: List[str], pos_tags: List[str]) -> List[str]:
    stream: List[str] = []
    for idx, root in enumerate(roots):
        pos_tag = pos_tags[idx] if idx < len(pos_tags) else "X"
        stream.append("X")  # start marker
        stream.extend([pos_tag] * len(root))
        stream.append("X")  # end marker
        if idx < len(roots) - 1:
            stream.append("X")  # whitespace marker
    return stream


class PosEncoding(TypedDict):
    pos_ids: List[int]
    pos_deltas: List[int]
    mean_abs_delta: float
    flat_mean_abs_delta: float


class PosDeltaReport(TypedDict):
    flat_mean: float
    delta_mean: float
    improvement_ratio: float
    conclusion: str


class StructuralEncoder:
    """Encodes Stage 3 outputs into integer symbol streams."""

    def __init__(self, symbol_alphabet: SymbolAlphabet):
        self.alphabet = symbol_alphabet

    def encode_tree_shape(self, tree_shape: str) -> int:
        """Convert a serialised tree shape string to its integer ID in the symbol alphabet."""
        return self.alphabet.get_id(tree_shape, add=True)

    def encode_pos_sequence(self, pos_tags: List[str]) -> PosEncoding:
        """Convert POS tag sequence to delta-encoded integer stream."""
        pos_ids = [self.alphabet.get_id(tag, add=True) for tag in pos_tags]
        if not pos_ids:
            return {
                "pos_ids": [],
                "pos_deltas": [],
                "mean_abs_delta": 0.0,
                "flat_mean_abs_delta": 0.0,
            }
        pos_deltas = [pos_ids[0]] + [b - a for a, b in zip(pos_ids, pos_ids[1:])]
        mean_abs_delta = sum(abs(v) for v in pos_deltas[1:]) / max(
            len(pos_deltas) - 1, 1
        )
        flat_mean_abs_delta = sum(abs(v) for v in pos_ids) / max(len(pos_ids), 1)
        return {
            "pos_ids": pos_ids,
            "pos_deltas": pos_deltas,
            "mean_abs_delta": float(mean_abs_delta),
            "flat_mean_abs_delta": float(flat_mean_abs_delta),
        }

    def encode_sentence_meta(self, syntax_result: SyntaxResult) -> Dict[str, int]:
        """Encode sentence-level metadata as single symbol codes."""
        sentence_type_map = {
            "DECLARATIVE": 0,
            "INTERROGATIVE": 1,
            "IMPERATIVE": 2,
            "EXCLAMATORY": 3,
        }
        voice_map = {"ACTIVE": 0, "PASSIVE": 1}
        return {
            "tree_shape_id": self.encode_tree_shape(syntax_result.tree_shape),
            "sentence_type_code": sentence_type_map.get(syntax_result.sentence_type, 0),
            "voice_code": voice_map.get(syntax_result.voice, 0),
        }

    def decode_pos_sequence(self, pos_deltas: List[int], first_id: int) -> List[str]:
        """Reconstruct POS tag strings from delta stream."""
        if not pos_deltas:
            return []
        ids = [first_id]
        for delta in pos_deltas[1:]:
            ids.append(ids[-1] + delta)
        return [self.alphabet.get_symbol(pos_id) for pos_id in ids]

    def decode_sentence_meta(self, meta: Dict[str, int]) -> Dict[str, str]:
        """Reconstruct sentence_type and voice strings from codes."""
        sentence_type_reverse = {
            0: "DECLARATIVE",
            1: "INTERROGATIVE",
            2: "IMPERATIVE",
            3: "EXCLAMATORY",
        }
        voice_reverse = {0: "ACTIVE", 1: "PASSIVE"}
        return {
            "sentence_type": sentence_type_reverse.get(
                meta.get("sentence_type_code", 0), "DECLARATIVE"
            ),
            "voice": voice_reverse.get(meta.get("voice_code", 0), "ACTIVE"),
        }

    def measure_pos_delta_improvement(self, pos_tags: List[str]) -> PosDeltaReport:
        """Measure POS delta effectiveness and print a conclusion."""
        encoded = self.encode_pos_sequence(pos_tags)
        flat_mean = float(encoded["flat_mean_abs_delta"])
        delta_mean = float(encoded["mean_abs_delta"])
        improvement = (flat_mean / delta_mean) if delta_mean else 0.0
        conclusion = "DELTA HELPS" if delta_mean < flat_mean else "DELTA DOES NOT HELP"
        print(
            f"POS delta test: flat={flat_mean:.4f} delta={delta_mean:.4f} "
            f"ratio={improvement:.3f} -> {conclusion}"
        )
        return {
            "flat_mean": flat_mean,
            "delta_mean": delta_mean,
            "improvement_ratio": improvement,
            "conclusion": conclusion,
        }


class CharacterEncoder:
    """Encoder for the Track A character layer."""

    def __init__(self) -> None:
        self.phonetic_map = PhoneticMap()
        self.inverse_map = {coords: char for char, coords in PHONETIC_CLASSES.items()}

    def encode_word(self, root: str) -> Dict[str, List]:
        """Encode a single root word."""
        triples = [
            self.phonetic_map.char_to_triple(char, pos, len(root))
            for pos, char in enumerate(root)
        ]
        class_deltas, pos_deltas, role_deltas = self.phonetic_map.compute_deltas(
            triples
        )
        role_stream = [role for _, _, role in triples]
        return {
            "class_deltas": class_deltas,
            "pos_deltas": pos_deltas,
            "role_stream": role_stream,
            "factoriadic_class": [encode_factoradic(v) for v in class_deltas],
            "factoriadic_pos": [encode_factoradic(v) for v in pos_deltas],
            "role_deltas": role_deltas,
        }

    def encode_sentence(self, morphology_results: List[Tuple]) -> Dict[str, List]:
        """Encode a full sentence given Stage 2 output."""
        roots: List[str] = []
        for item in morphology_results:
            if len(item) == 3:
                roots.append(item[1])
            else:
                roots.append(item[0])
        triples = _triples_for_sentence(roots)
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
    ) -> Dict[str, object]:
        """Encode a full sentence combining character and structural streams."""
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
        triples = _triples_for_sentence(roots)
        char_classes = _char_classes_from_triples(triples)
        char_morph_codes = _expand_morph_codes_for_chars(roots, morph_codes)
        char_pos_tags = _expand_pos_tags_for_chars(roots, syntax_result.pos_tags)
        pos_encoding = structural_encoder.encode_pos_sequence(syntax_result.pos_tags)
        tree_shape_id = structural_encoder.encode_tree_shape(syntax_result.tree_shape)
        sentence_meta = structural_encoder.encode_sentence_meta(syntax_result)
        pos_delta_report = structural_encoder.measure_pos_delta_improvement(
            syntax_result.pos_tags
        )

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
            "pos_delta_report": pos_delta_report,
        }

    def decode_word(self, encoded: Dict[str, List[int]]) -> str:
        """Reconstruct root word from encoded streams. Must round-trip perfectly."""
        class_deltas = encoded.get("class_deltas", [])
        pos_deltas = encoded.get("pos_deltas", [])
        classes = _cumulative_from_deltas(class_deltas)
        positions = _cumulative_from_deltas(pos_deltas)
        chars: List[str] = []
        for cls, pos in zip(classes, positions):
            chars.append(self.inverse_map.get((cls, pos), "?"))
        return "".join(chars)

    def stats(self, text: str) -> Dict[str, float]:
        """Run full encoding on text and return compression statistics."""
        words = [w.lower() for w in text.split() if w.strip()]
        if not words:
            return {
                "mean_class_delta": 0.0,
                "mean_pos_delta": 0.0,
                "mean_role_delta": 0.0,
                "mean_flat_delta": 0.0,
                "mean_decomp_magnitude": 0.0,
                "improvement_ratio": 0.0,
            }

        triples = _triples_for_sentence(words)
        class_deltas, pos_deltas, role_deltas = compute_deltas(triples)

        deltas_2d = [
            sqrt(dc * dc + dp * dp) for dc, dp in zip(class_deltas[1:], pos_deltas[1:])
        ]
        mean_decomp = sum(deltas_2d) / len(deltas_2d) if deltas_2d else 0.0
        mean_class = sum(abs(v) for v in class_deltas[1:]) / max(
            len(class_deltas) - 1, 1
        )
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
