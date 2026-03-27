"""Context-mixing probability model for symbolic compression."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, cast

import msgpack

from compression.config import (
    CHAR_CONTEXT_SIZE,
    MORPH_CONTEXT_SIZE,
    STRUCT_CONTEXT_SIZE,
)


def _prepare_for_pack(data: dict) -> dict:
    """Recursively convert integer keys to strings for msgpack compatibility."""
    return {
        str(k): (_prepare_for_pack(v) if isinstance(v, dict) else v)
        for k, v in data.items()
    }


def _restore_from_pack(data: dict) -> dict:
    """Recursively restore string keys to integers after msgpack unpack."""
    result = {}
    for k, v in data.items():
        int_key = int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k
        result[int_key] = _restore_from_pack(v) if isinstance(v, dict) else v
    return result


def bits_for_symbol(prob: float) -> float:
    return -math.log2(max(prob, 1e-10))


def total_bits(symbol_stream: List[int], prob_fn) -> float:
    return sum(bits_for_symbol(prob_fn(symbol)) for symbol in symbol_stream)


def bpb(total_bits_value: float, original_text: str) -> float:
    return total_bits_value / len(original_text.encode("utf-8"))


class UnigramModel:
    """Baseline unigram model for comparison."""

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()
        self.total: int = 0
        self.vocab: List[int] = []

    def train(self, sequences: Iterable[List[int]]) -> None:
        for sequence in sequences:
            self.counts.update(sequence)
            self.total += len(sequence)
        self.vocab = sorted(self.counts.keys())

    def distribution(self) -> Dict[int, float]:
        vocab = self.vocab or [0]
        total = self.total + len(vocab)
        return {sym: (self.counts.get(sym, 0) + 1) / total for sym in vocab}

    def probability(self, symbol: int) -> float:
        dist = self.distribution()
        return dist.get(symbol, 1.0 / max(len(dist), 1))


class ContextMixingModel:
    """
    Combines evidence from three context levels to estimate symbol probability.

    Level 1 — Character context:   P(next_char_class | prev_char_class)
    Level 2 — Morphological context: P(char_class | current_morph_code)
    Level 3 — Structural context:  P(morph_code | current_pos_tag)

    Final probability = weighted average of all three levels.
    """

    def __init__(self) -> None:
        self.char_context: Dict[int, Counter[int]] = defaultdict(Counter)
        self.morph_context: Dict[int, Counter[int]] = defaultdict(Counter)
        self.struct_context: Dict[str, Counter[int]] = defaultdict(Counter)
        self.weights: List[float] = [1 / 3, 1 / 3, 1 / 3]
        self.char_vocab: List[int] = []
        self.morph_vocab: List[int] = []
        self.pos_vocab: List[str] = []

    def train(self, encoded_sentences: List[Dict]) -> None:
        """
        Two-pass training:
          Pass 1 — Build all context tables with full counts (no sliding window).
          Pass 2 — Update weights on every char position using the fully-built tables.
        This ensures weight updates see accurate distributions rather than empty
        or partially-populated tables.
        """
        char_vocab: set = set()
        morph_vocab: set = set()
        pos_vocab: set = set()

        # ── Pass 1: populate context tables ──────────────────────────────────
        for sentence in encoded_sentences:
            char_classes = sentence.get("char_classes", [])
            char_morph_codes = sentence.get("char_morph_codes", [])
            pos_tags = sentence.get("pos_tags", [])
            morph_codes = sentence.get("morph_codes", [])

            # char bigram: P(curr | prev)
            for prev, curr in zip(char_classes, char_classes[1:]):
                self.char_context[int(prev)][int(curr)] += 1

            # morph → char: P(char_class | morph_code)
            for char_class, morph_code in zip(char_classes, char_morph_codes):
                self.morph_context[int(morph_code)][int(char_class)] += 1

            # pos → morph: P(morph_code | pos_tag)
            for pos_tag, morph_code in zip(pos_tags, morph_codes):
                self.struct_context[str(pos_tag)][int(morph_code)] += 1

            char_vocab.update(int(c) for c in char_classes)
            morph_vocab.update(int(m) for m in morph_codes)
            pos_vocab.update(str(p) for p in pos_tags)

        self.char_vocab = sorted(char_vocab) if char_vocab else list(range(7))
        self.morph_vocab = sorted(morph_vocab) if morph_vocab else [0]
        self.pos_vocab = sorted(pos_vocab)

        # ── Pass 2: update weights on every char position ────────────────────
        char_vocab_list = self.char_vocab
        morph_vocab_list = self.morph_vocab

        for sentence in encoded_sentences:
            char_classes = sentence.get("char_classes", [])
            char_morph_codes = sentence.get("char_morph_codes", [])
            char_pos_tags = sentence.get("char_pos_tags", [])

            for idx in range(1, len(char_classes)):
                symbol = int(char_classes[idx])
                prev_char = int(char_classes[idx - 1])
                current_morph = (
                    int(char_morph_codes[idx]) if idx < len(char_morph_codes) else 0
                )
                current_pos = (
                    str(char_pos_tags[idx]) if idx < len(char_pos_tags) else "X"
                )

                p1 = self._smoothed_distribution(
                    self.char_context[prev_char], char_vocab_list
                )
                p2 = self._smoothed_distribution(
                    self.morph_context[current_morph], char_vocab_list
                )
                p3 = self._char_distribution_from_struct_context(
                    current_pos, char_vocab_list, morph_vocab_list
                )

                self._update_weights(symbol, p1, p2, p3)

    def _update_weights(self, symbol: int, p1: Dict, p2: Dict, p3: Dict) -> None:
        """Nudge weights toward the stream that assigned the highest probability."""
        scores = [
            p1.get(symbol, 1e-10),
            p2.get(symbol, 1e-10),
            p3.get(symbol, 1e-10),
        ]
        best = max(range(3), key=lambda i: scores[i])
        lr = 0.0001
        floor = 0.05
        for i in range(3):
            if i == best:
                self.weights[i] += lr
            else:
                self.weights[i] = max(self.weights[i] - lr / 2, floor)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

    def probability(self, symbol: int, context: Dict) -> float:
        """Estimate P(symbol | context)."""
        distribution = self.probability_distribution(context)
        return distribution.get(symbol, 1.0 / max(len(distribution), 1))

    def probability_distribution(self, context: Dict) -> Dict[int, float]:
        char_history = context.get("char_history", [])
        current_morph_code = int(context.get("current_morph_code", 0))
        current_pos_tag = str(context.get("current_pos_tag", "X"))
        struct_prob = float(context.get("struct_prob", 1.0))

        char_vocab = self.char_vocab or list(range(7))
        morph_vocab = self.morph_vocab or [0]

        prev_char = int(char_history[-1]) if char_history else 0
        p1 = self._smoothed_distribution(self.char_context[prev_char], char_vocab)
        p2 = self._smoothed_distribution(
            self.morph_context[current_morph_code], char_vocab
        )
        p3 = self._char_distribution_from_struct_context(
            current_pos_tag, char_vocab, morph_vocab
        )

        w3 = self.weights[2] * struct_prob
        w_total = self.weights[0] + self.weights[1] + w3

        combined = {
            char_class: (
                self.weights[0] * p1[char_class]
                + self.weights[1] * p2[char_class]
                + w3 * p3[char_class]
            )
            / w_total
            for char_class in char_vocab
        }
        return combined

    def bpb(self, text: str, pipeline) -> float:
        """Run full pipeline on text, compute bits per byte."""
        encoded_sentences = pipeline.encode_for_model(text)
        total_bits_value = 0.0

        infer_char_history: deque = deque(maxlen=CHAR_CONTEXT_SIZE)

        for sentence in encoded_sentences:
            char_classes = sentence.get("char_classes", [])
            char_morph_codes = sentence.get("char_morph_codes", [])
            char_pos_tags = sentence.get("char_pos_tags", [])
            n_tags = int(sentence.get("pos_n_tags", 0))
            huffman_bits = float(sentence.get("pos_huffman_bits", 0.0))

            if n_tags > 0 and huffman_bits > 0:
                struct_prob = 2 ** (-huffman_bits / n_tags)
            else:
                pos_vocab_size = max(len(self.pos_vocab), 1)
                struct_prob = 1.0 / pos_vocab_size

            for idx, symbol in enumerate(char_classes):
                current_morph = (
                    int(char_morph_codes[idx]) if idx < len(char_morph_codes) else 0
                )
                current_pos = (
                    str(char_pos_tags[idx]) if idx < len(char_pos_tags) else "X"
                )

                context = {
                    "char_history": list(infer_char_history),
                    "current_morph_code": current_morph,
                    "current_pos_tag": current_pos,
                    "struct_prob": struct_prob,
                }

                prob = self.probability(int(symbol), context)
                total_bits_value += bits_for_symbol(prob)
                infer_char_history.append(int(symbol))

        result = bpb(total_bits_value, text)
        print(f"Context-mixing bpb: {result:.4f}")
        return result

    def serialise(self, path: str) -> int:
        """Save model to binary file. Return file size in bytes."""
        payload = {
            "char_context": {
                int(key): dict(counter) for key, counter in self.char_context.items()
            },
            "morph_context": {
                int(key): dict(counter) for key, counter in self.morph_context.items()
            },
            "struct_context": {
                str(key): dict(counter) for key, counter in self.struct_context.items()
            },
            "weights": self.weights,
            "char_vocab": self.char_vocab,
            "morph_vocab": self.morph_vocab,
            "pos_vocab": self.pos_vocab,
        }
        prepared = _prepare_for_pack(payload)
        packed = msgpack.packb(prepared, use_bin_type=True)
        packed_bytes = cast(bytes, packed)
        Path(path).write_bytes(packed_bytes)
        return len(packed_bytes)

    def load(self, path: str) -> None:
        """Load model from binary file saved by serialise()."""
        data = msgpack.unpackb(Path(path).read_bytes(), raw=False, strict_map_key=False)
        self.char_context = defaultdict(Counter)
        self.morph_context = defaultdict(Counter)
        self.struct_context = defaultdict(Counter)

        char_context = _restore_from_pack(data.get("char_context", {}))
        morph_context = _restore_from_pack(data.get("morph_context", {}))
        struct_context = _restore_from_pack(data.get("struct_context", {}))

        for key, counter in char_context.items():
            self.char_context[int(key)] = Counter(counter)
        for key, counter in morph_context.items():
            self.morph_context[int(key)] = Counter(counter)
        for key, counter in struct_context.items():
            self.struct_context[str(key)] = Counter(counter)

        self.weights = list(data.get("weights", [1 / 3, 1 / 3, 1 / 3]))
        self.char_vocab = list(data.get("char_vocab", list(range(7))))
        self.morph_vocab = list(data.get("morph_vocab", [0]))
        self.pos_vocab = list(data.get("pos_vocab", []))

    def global_char_distribution(self) -> Dict[int, float]:
        """Return a smoothed global distribution over character classes."""
        char_vocab = self.char_vocab or list(range(7))
        totals: Counter[int] = Counter()
        for counter in self.morph_context.values():
            totals.update(counter)
        if not totals:
            return {symbol: 1.0 / len(char_vocab) for symbol in char_vocab}
        total = sum(totals.values())
        vocab_size = max(len(char_vocab), 1)
        denom = total + vocab_size
        return {symbol: (totals.get(symbol, 0) + 1) / denom for symbol in char_vocab}

    def _smoothed_distribution(
        self, counter: Counter[int], vocab: List[int]
    ) -> Dict[int, float]:
        total = sum(counter.values())
        vocab_size = max(len(vocab), 1)
        denom = total + vocab_size
        return {symbol: (counter.get(symbol, 0) + 1) / denom for symbol in vocab}

    def _char_distribution_from_struct_context(
        self, pos_tag: str, char_vocab: List[int], morph_vocab: List[int]
    ) -> Dict[int, float]:
        morph_counter = self.struct_context.get(pos_tag, Counter())
        morph_total = sum(morph_counter.values())
        morph_vocab_size = max(len(morph_vocab), 1)
        morph_probs = {
            morph_code: (morph_counter.get(morph_code, 0) + 1)
            / (morph_total + morph_vocab_size)
            for morph_code in morph_vocab
        }

        char_dist: Dict[int, float] = {char_class: 0.0 for char_class in char_vocab}
        for morph_code, morph_prob in morph_probs.items():
            char_probs = self._smoothed_distribution(
                self.morph_context.get(morph_code, Counter()), char_vocab
            )
            for char_class, char_prob in char_probs.items():
                char_dist[char_class] += morph_prob * char_prob

        total = sum(char_dist.values()) or 1.0
        return {char_class: value / total for char_class, value in char_dist.items()}
