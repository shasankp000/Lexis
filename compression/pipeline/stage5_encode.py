"""Character encoder integrating phonetic deltas and factoriadic encoding."""

from __future__ import annotations

import heapq
import json
import struct
from collections import Counter
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

# ---------------------------------------------------------------------------
# Case-flag constants
# ---------------------------------------------------------------------------
# 0  all-lower      e.g. "the", "quick"
# 1  Title-case     e.g. "The", "Project"  (first char upper, rest lower)
# 2  ALL-CAPS       e.g. "USA", "HTTP"
# 3  mixed          e.g. "eBook", "iPhone"  (char-level bitmap packed below)
#
# For flag 3 a secondary bitmap is stored: one bit per character of the
# *root* (not the surface form), starting from index 1 (char 0 is always
# lowercase in mixed mode).  bit 0 of bitmap = char index 1, bit 1 = char
# index 2, etc.  The bitmap is packed into an integer and stored alongside
# case_flags in the LEXI metadata.

CASE_LOWER = 0
CASE_TITLE = 1
CASE_UPPER = 2
CASE_MIXED = 3


def compute_case_flag(surface: str) -> Tuple[int, int]:
    """Return (flag, bitmap) for a single token surface form.

    bitmap is only meaningful when flag == CASE_MIXED; otherwise 0.
    """
    if not surface:
        return CASE_LOWER, 0
    if not any(ch.isalpha() for ch in surface):
        return CASE_LOWER, 0
    if surface.islower():
        return CASE_LOWER, 0
    if surface.isupper():
        return CASE_UPPER, 0
    if surface[0].isupper() and surface[1:].islower():
        return CASE_TITLE, 0
    # mixed — pack a per-char bitmap starting from index 1
    # bit 0 of bitmap = char at index 1, bit 1 = char at index 2, etc.
    # char at index 0 is always lowercase in CASE_MIXED.
    bitmap = 0
    for i, ch in enumerate(surface[1:], start=0):
        if ch.isupper():
            bitmap |= (1 << i)
    return CASE_MIXED, bitmap


def apply_case_flag(word: str, flag: int, bitmap: int = 0) -> str:
    """Apply a case flag (and optional bitmap) to a lowercase word."""
    if not word:
        return word
    if flag == CASE_LOWER:
        return word
    if flag == CASE_TITLE:
        return word[0].upper() + word[1:]
    if flag == CASE_UPPER:
        return word.upper()
    # CASE_MIXED — char 0 is always lowercase; bitmap bit 0 = char index 1
    chars = list(word.lower())
    for bit_pos, char_idx in enumerate(range(1, len(chars))):
        if bitmap & (1 << bit_pos):
            chars[char_idx] = chars[char_idx].upper()
    return "".join(chars)


# ---------------------------------------------------------------------------
# Factoriadic helpers
# ---------------------------------------------------------------------------

def _encode_factoradic_unsigned(value: int) -> List[int]:
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


def _decode_factoradic_unsigned(digits: List[int]) -> int:
    """Decode factoriadic digits into a non-negative integer."""
    if not digits:
        return 0
    total = 0
    length = len(digits)
    for idx, digit in enumerate(digits):
        total += digit * factorial(length - 1 - idx)
    return total


def encode_factoradic(value: int) -> List[int]:
    """Encode any integer into signed factoriadic digits."""
    if value < 0:
        digits = _encode_factoradic_unsigned(abs(value))
        return [-1] + digits
    return _encode_factoradic_unsigned(value)


def decode_factoradic(digits: List[int]) -> int:
    """Decode signed factoriadic digits into an integer."""
    if not digits:
        raise ValueError("empty digits list")
    if digits[0] < 0:
        value = _decode_factoradic_unsigned(digits[1:])
        return -value
    return _decode_factoradic_unsigned(digits)


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
    pos_huffman_bits: int
    pos_huffman_bitstring: str
    pos_huffman_codes: Dict[str, str]
    tag_count: int
    vocab_size: int


class StructuralEncoder:
    """Encodes Stage 3 outputs into integer symbol streams."""

    def __init__(self, symbol_alphabet: SymbolAlphabet):
        self.alphabet = symbol_alphabet

    def encode_tree_shape(self, tree_shape: str) -> int:
        """Convert a serialised tree shape string to its integer ID in the symbol alphabet."""
        return self.alphabet.get_id(tree_shape, add=True)

    def build_pos_frequency_table(self, sentences: List[List[str]]) -> Dict[str, int]:
        """Build a frequency table mapping POS tag -> count."""
        counts = Counter(tag for sentence in sentences for tag in sentence)
        return dict(counts)

    def build_pos_huffman_codes(self, freq_table: Dict[str, int]) -> Dict[str, str]:
        """Build Huffman codes for POS tags using heapq."""
        if not freq_table:
            return {}

        heap: List[Tuple[int, int, object]] = []
        order = 0
        for tag, count in freq_table.items():
            heapq.heappush(heap, (count, order, tag))
            order += 1

        if len(heap) == 1:
            _, _, tag = heap[0]
            return {str(tag): "0"}

        while len(heap) > 1:
            freq1, _, node1 = heapq.heappop(heap)
            freq2, _, node2 = heapq.heappop(heap)
            merged = (node1, node2)
            heapq.heappush(heap, (freq1 + freq2, order, merged))
            order += 1

        _, _, root = heap[0]
        codes: Dict[str, str] = {}

        def assign(node, prefix: str) -> None:
            if isinstance(node, str):
                codes[node] = prefix or "0"
                return
            left, right = node
            assign(left, prefix + "0")
            assign(right, prefix + "1")

        assign(root, "")
        return codes

    def encode_pos_sequence(
        self, pos_tags: List[str], freq_table: Dict[str, int]
    ) -> PosEncoding:
        """Encode POS tag sequence with Huffman coding."""
        codes = self.build_pos_huffman_codes(freq_table)
        if not pos_tags or not codes:
            return {
                "pos_huffman_bits": 0,
                "pos_huffman_bitstring": "",
                "pos_huffman_codes": codes,
                "tag_count": 0,
                "vocab_size": len(codes),
            }

        bitstring = "".join(codes.get(tag, "") for tag in pos_tags)
        return {
            "pos_huffman_bits": len(bitstring),
            "pos_huffman_bitstring": bitstring,
            "pos_huffman_codes": codes,
            "tag_count": len(pos_tags),
            "vocab_size": len(codes),
        }

    def decode_pos_sequence(
        self, bitstring: str, huffman_codes: Dict[str, str]
    ) -> List[str]:
        """Decode Huffman-encoded POS bitstring back to tag sequence."""
        if not bitstring or not huffman_codes:
            return []
        code_to_tag = {code: tag for tag, code in huffman_codes.items()}
        decoded: List[str] = []
        buffer = ""
        for bit in bitstring:
            buffer += bit
            if buffer in code_to_tag:
                decoded.append(code_to_tag[buffer])
                buffer = ""
        return decoded

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

    def serialise_freq_table(self, freq_table: Dict[str, int]) -> bytes:
        """Serialise frequency table to bytes for storage in archive header."""
        return json.dumps(freq_table, separators=(",", ":")).encode("utf-8")

    def deserialise_freq_table(self, data: bytes) -> Dict[str, int]:
        """Deserialise frequency table from archive header bytes."""
        return json.loads(data.decode("utf-8"))

    def write_archive(self, freq_table: Dict[str, int], payload: bytes) -> bytes:
        """Write a length-prefixed archive with freq table header."""
        table_bytes = self.serialise_freq_table(freq_table)
        header = struct.pack(">I", len(table_bytes))
        return header + table_bytes + payload

    def read_archive(self, data: bytes) -> Tuple[Dict[str, int], bytes]:
        """Read a length-prefixed archive and return freq table + payload."""
        table_len = struct.unpack(">I", data[:4])[0]
        table_bytes = data[4 : 4 + table_len]
        payload = data[4 + table_len :]
        return self.deserialise_freq_table(table_bytes), payload


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
        freq_table: Dict[str, int],
    ) -> Dict[str, object]:
        """Encode a full sentence combining character and structural streams."""
        roots: List[str] = []
        morph_codes: List[int] = []
        case_flags: List[int] = []
        case_bitmaps: List[int] = []

        for item in morphology_results:
            # item[0] is the original surface token (preserves case)
            # item[1] is the root (already lowercased by Stage 2)
            surface = item[0] if len(item) >= 1 else ""
            if len(item) == 3:
                roots.append(item[1])
                morph_codes.append(int(item[2]))
            else:
                roots.append(item[0].lower())
                morph_codes.append(BASE)
            flag, bitmap = compute_case_flag(surface)
            case_flags.append(flag)
            case_bitmaps.append(bitmap)

        char_encoding = self.encode_sentence(morphology_results)
        triples = _triples_for_sentence(roots)
        char_classes = _char_classes_from_triples(triples)
        char_morph_codes = _expand_morph_codes_for_chars(roots, morph_codes)
        char_pos_tags = _expand_pos_tags_for_chars(roots, syntax_result.pos_tags)
        pos_encoding = structural_encoder.encode_pos_sequence(
            syntax_result.pos_tags, freq_table
        )
        tree_shape_id = structural_encoder.encode_tree_shape(syntax_result.tree_shape)
        sentence_meta = structural_encoder.encode_sentence_meta(syntax_result)

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
            # case restoration
            "case_flags": case_flags,
            "case_bitmaps": case_bitmaps,
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
