"""Stage 8 decoding for the hierarchical compression pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

import msgpack

from compression.alphabet.morph_codes import apply_morph
from compression.alphabet.phonetic_map import PHONETIC_CLASSES
from compression.pipeline.stage5_encode import decode_factoradic
from compression.pipeline.stage6_probability import ContextMixingModel
from compression.pipeline.stage7_ans import ANSDecoder, FrequencyTable


def _cumulative_from_deltas(deltas: List[int]) -> List[int]:
    if not deltas:
        return []
    values = [deltas[0]]
    for delta in deltas[1:]:
        values.append(values[-1] + delta)
    return values


def decode_morphology(morphology: List[Mapping[str, Any]]) -> str:
    """Reconstruct surface text from morphology entries."""
    words: List[str] = []
    for entry in morphology:
        root = str(entry.get("root", ""))
        code_value = entry.get("code", 0)
        code = int(code_value) if isinstance(code_value, (int, str)) else 0
        words.append(apply_morph(root, code))
    return " ".join(words)


def decode_payload(payload: Mapping[str, Any]) -> str:
    """Decode a payload dictionary back into text."""
    morphology = payload.get("morphology", [])
    if isinstance(morphology, list):
        return decode_morphology(morphology)
    return ""


class FullDecoder:
    """Reverses the complete compression pipeline."""

    def __init__(self) -> None:
        self.inverse_map = {coords: char for char, coords in PHONETIC_CLASSES.items()}

    def decode(self, compressed: bytes, model_path: str) -> str:
        """
        Decode from compressed bytes + model path.
        Expects a msgpack payload with:
          - ans_state: bytes
          - num_symbols: int
          - stream_layout: {"order": [...], "lengths": {...}}
          - probabilities (optional): dict[int|str, float]
        """
        payload = msgpack.unpackb(compressed, raw=False)

        if "morphology" in payload:
            return decode_payload(payload)

        ans_state = payload.get("ans_state", b"")
        num_symbols = int(payload.get("num_symbols", 0))
        layout = payload.get("stream_layout", {})

        model = ContextMixingModel()
        model.load(model_path)

        probs_payload = payload.get("probabilities")
        if isinstance(probs_payload, dict):
            probabilities = {int(k): float(v) for k, v in probs_payload.items()}
        else:
            probabilities = model.global_char_distribution()

        table = FrequencyTable(probabilities)
        decoder = ANSDecoder(table)
        state = decoder.bytes_to_state(ans_state)
        stream = decoder.decode(state, num_symbols)

        streams = self._split_streams(stream, layout)

        class_deltas = self._decode_factoriadic_stream(
            streams.get("char_class_deltas", []),
            streams.get("factoriadic_class", []),
        )
        pos_deltas = self._decode_factoriadic_stream(
            streams.get("char_pos_deltas", []),
            streams.get("factoriadic_pos", []),
        )
        role_stream = streams.get("char_role_stream", [])

        char_stream = self._reconstruct_chars(class_deltas, pos_deltas, role_stream)
        roots = self._split_roots_from_stream(char_stream)

        morph_codes = streams.get("morph_code_stream", [])
        words = self._reconstruct_words(roots, morph_codes)

        sentence_meta = streams.get("sentence_meta", [])
        if sentence_meta:
            sentence_type_code = int(sentence_meta[0])
        else:
            sentence_type_code = 0

        return self._reconstruct_sentence(words, sentence_type_code)

    def _decode_factoriadic_stream(
        self, deltas: List[int], factoriadic: List[Any]
    ) -> List[int]:
        if deltas:
            return [int(v) for v in deltas]
        if not factoriadic:
            return []
        first = factoriadic[0]
        if isinstance(first, list):
            return [decode_factoradic(digits) for digits in factoriadic]
        return [int(v) for v in factoriadic]

    def _split_streams(
        self, stream: List[int], layout: Mapping[str, Any]
    ) -> Dict[str, List[int]]:
        order = layout.get(
            "order",
            [
                "tree_shape_ids",
                "sentence_meta",
                "pos_deltas",
                "morph_code_stream",
                "char_class_deltas",
                "char_pos_deltas",
                "char_role_stream",
            ],
        )
        lengths = layout.get("lengths", {})
        cursor = 0
        streams: Dict[str, List[int]] = {}
        for name in order:
            length = int(lengths.get(name, 0))
            if length <= 0:
                streams[name] = []
                continue
            streams[name] = stream[cursor : cursor + length]
            cursor += length
        return streams

    def _reconstruct_chars(
        self, class_deltas: List[int], pos_deltas: List[int], role_stream: List[int]
    ) -> str:
        classes = _cumulative_from_deltas(class_deltas)
        positions = _cumulative_from_deltas(pos_deltas)
        chars: List[str] = []
        for cls, pos in zip(classes, positions):
            chars.append(self.inverse_map.get((cls, pos), "?"))
        return "".join(chars)

    def _split_roots_from_stream(self, char_stream: str) -> List[str]:
        roots: List[str] = []
        current: List[str] = []
        for char in char_stream:
            if char == "^":
                current = []
            elif char == "$":
                if current:
                    roots.append("".join(current))
                current = []
            elif char == "_":
                continue
            else:
                current.append(char)
        if current:
            roots.append("".join(current))
        return roots

    def _reconstruct_words(self, roots: List[str], morph_codes: List[int]) -> List[str]:
        words: List[str] = []
        for idx, root in enumerate(roots):
            code = morph_codes[idx] if idx < len(morph_codes) else 0
            words.append(apply_morph(root, int(code)))
        return words

    def _reconstruct_sentence(self, words: List[str], sentence_type_code: int) -> str:
        sentence = " ".join(words)
        if not sentence:
            return ""
        sentence = sentence[0].upper() + sentence[1:] if sentence else sentence
        if sentence_type_code == 1 and not sentence.endswith("?"):
            return sentence + "?"
        if sentence_type_code == 3 and not sentence.endswith("!"):
            return sentence + "!"
        if not sentence.endswith((".", "?", "!")):
            return sentence + "."
        return sentence
