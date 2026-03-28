"""spaCy-backed morphological analyser with rule-based fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from compression.alphabet.morph_codes import (
    ADVERBIAL,
    BASE,
    COMPARATIVE,
    IRREGULAR,
    NEGATION,
    NOMINALIZE,
    PAST_PART,
    PAST_TENSE,
    PLURAL,
    PRESENT_PART,
    SUPERLATIVE,
    THIRD_SING,
)
from compression.config import SPACY_MAX_LENGTH, SPACY_MODEL
from compression.pipeline.utils import chunk_text, split_sentences

_IRREGULAR_ROOT_TO_SURFACE: Dict[str, str] = {
    "be": "was",
    "begin": "began",
    "break": "broke",
    "come": "came",
    "do": "did",
    "drive": "drove",
    "eat": "ate",
    "get": "got",
    "give": "gave",
    "go": "went",
    "have": "had",
    "leave": "left",
    "make": "made",
    "run": "ran",
    "say": "said",
    "swim": "swam",
    "see": "saw",
    "sit": "sat",
    "speak": "spoke",
    "take": "took",
    "write": "wrote",
}
_IRREGULAR_SURFACE_TO_ROOT: Dict[str, str] = {
    **{surface: root for root, surface in _IRREGULAR_ROOT_TO_SURFACE.items()},
    "are": "are",
    "were": "were",
    "been": "been",
    "me": "me",
    "him": "him",
    "her": "her",
    "us": "us",
    "them": "them",
    "my": "my",
    "his": "his",
    "its": "its",
    "our": "our",
    "their": "their",
    "your": "your",
    "myself": "myself",
    "himself": "himself",
    "herself": "herself",
    "itself": "itself",
    "ourselves": "ourselves",
    "themselves": "themselves",
    "yourself": "yourself",
    "yourselves": "yourselves",
    "most": "most",
    "best": "best",
    "worst": "worst",
    "least": "least",
    "more": "more",
    "better": "better",
    "worse": "worse",
    "less": "less",
    "further": "further",
    "furthest": "furthest",
    "many": "many",
    "much": "much",
}

# Words whose surface == root: must always encode as BASE, never IRREGULAR,
# because the IRREGULAR decode path calls lemminflect(VBD) and would corrupt them.
_IDENTITY_WORDS: frozenset = frozenset(
    word for word, root in _IRREGULAR_SURFACE_TO_ROOT.items() if word == root
)


@dataclass(frozen=True)
class MorphResult:
    original: str
    root: str
    code: int


class MorphologicalAnalyser:
    """Analyse words into (root, morph_code) pairs."""

    def __init__(self, use_spacy: bool = True, model_name: str | None = None) -> None:
        self.use_spacy = use_spacy
        self.nlp = None
        model_to_load = model_name or SPACY_MODEL
        if use_spacy:
            try:
                import spacy  # type: ignore

                self.nlp = spacy.load(model_to_load)
                self.nlp.max_length = SPACY_MAX_LENGTH
            except Exception:
                self.nlp = None

    def analyse(self, word: str) -> Tuple[str, int]:
        """Return (root, morph_code) for a single word."""
        if not word or word.isspace():
            return (word, BASE)

        if self.nlp is not None:
            doc = self.nlp(word)
            for token in doc:
                if token.is_space:
                    continue
                return self._analyse_token_spacy(token)
            return (word, BASE)

        return self._analyse_rule_based(word)

    def analyse_sentence(self, sentence: str) -> List[Tuple[str, str, int]]:
        """Return list of (original_word, root, morph_code) for all words in sentence."""
        results: List[Tuple[str, str, int]] = []
        if self.nlp is not None:
            doc = self.nlp(sentence)
            for token in doc:
                if token.is_space:
                    continue
                root, code = self._analyse_token_spacy(token)
                results.append((token.text, root, code))
            return results

        for raw in sentence.split():
            root, code = self._analyse_rule_based(raw)
            results.append((raw, root, code))
        return results

    def char_savings(self, text: str) -> Dict[str, float]:
        """Compute morphological character savings across the full text."""
        total_original = 0
        total_root = 0

        for chunk in self._chunk_text(text):
            for sentence in self._split_sentences(chunk):
                if not sentence.strip():
                    continue
                try:
                    results = self.analyse_sentence(sentence)
                    total_original += sum(len(original) for original, _, _ in results)
                    total_root += sum(len(root) for _, root, _ in results)
                except Exception:
                    total_original += len(sentence)
                    total_root += len(sentence)

        chars_saved = total_original - total_root
        pct_saved = (chars_saved / total_original * 100.0) if total_original else 0.0

        return {
            "original_chars": float(total_original),
            "root_chars": float(total_root),
            "chars_saved": float(chars_saved),
            "pct_saved": float(pct_saved),
        }

    def _chunk_text(self, text: str, chunk_size: int = 500_000) -> List[str]:
        """Split text into chunks of ~chunk_size chars, breaking at newlines."""
        return chunk_text(text, chunk_size)

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using a heuristic."""
        return split_sentences(text)

    def _analyse_token_spacy(self, token) -> Tuple[str, int]:
        lower_text = token.text.lower()
        lemma = token.lemma_.lower() if token.lemma_ else lower_text
        morph = token.morph

        # Non-alpha tokens (digits, URLs, punctuation, mixed symbols) can never
        # be inflected meaningfully. Return them verbatim as BASE so that
        # apply_morph is a no-op and round-trip is guaranteed.
        # Example: spaCy assigns Number=Plur to bare cardinals like '2701',
        # which would otherwise fall through to the PLURAL branch and produce
        # '2701s' via lemminflect.
        if not any(ch.isalpha() for ch in lower_text):
            return (lower_text, BASE)

        # Identity-mapped words (surface == root) must stay BASE.
        # Returning IRREGULAR for them causes the decoder to call
        # lemminflect(VBD) and corrupt the word (e.g. "most" -> "mosted").
        if lower_text in _IDENTITY_WORDS:
            return (lower_text, BASE)

        if lower_text in _IRREGULAR_SURFACE_TO_ROOT:
            return (_IRREGULAR_SURFACE_TO_ROOT[lower_text], IRREGULAR)

        if lower_text.startswith("un") and lemma == lower_text[2:]:
            return (lemma, NEGATION)

        if token.pos_ == "ADV" and lower_text.endswith("ly") and lemma != lower_text:
            return (lemma, ADVERBIAL)

        if (
            token.tag_ == "VBG"
            or (
                token.pos_ in {"VERB", "AUX"}
                and (
                    "Aspect=Prog" in morph
                    or ("VerbForm=Part" in morph and "Tense=Pres" in morph)
                )
            )
            or (
                token.pos_ in {"VERB", "AUX"}
                and lower_text.endswith("ing")
                and lemma != lower_text
            )
        ):
            return (lemma, PRESENT_PART)

        if "Tense=Past" in morph:
            if token.tag_ == "VBN" or (
                "VerbForm=Part" in morph and token.tag_ == "VBN"
            ):
                return (lemma, PAST_PART)
            return (lemma, PAST_TENSE)

        if token.pos_ == "PRON":
            return (lower_text, BASE)

        # NUM must be checked before Number=Plur / Degree / Person=3 because
        # spaCy occasionally assigns Number=Plur morphology to cardinal numbers.
        if token.pos_ == "NUM":
            return (lower_text, BASE)

        if "Number=Plur" in morph:
            return (lemma, PLURAL)

        if "Degree=Cmp" in morph:
            return (lemma, COMPARATIVE)

        if "Degree=Sup" in morph:
            return (lemma, SUPERLATIVE)

        if (
            token.pos_ in {"VERB", "AUX"}
            and "Person=3" in morph
            and "Number=Sing" in morph
        ):
            return (lemma, THIRD_SING)

        return (lemma, BASE)

    def _analyse_rule_based(self, word: str) -> Tuple[str, int]:
        lower = word.lower()

        # Non-alpha tokens bypass all inflection logic.
        if not any(ch.isalpha() for ch in lower):
            return (lower, BASE)

        # Identity-mapped words must stay BASE in rule-based path too.
        if lower in _IDENTITY_WORDS:
            return (lower, BASE)

        if lower in _IRREGULAR_SURFACE_TO_ROOT:
            return (_IRREGULAR_SURFACE_TO_ROOT[lower], IRREGULAR)

        if lower.startswith("un") and len(lower) > 2:
            return (lower[2:], NEGATION)

        if lower.endswith("ing") and len(lower) > 4:
            base = lower[:-3]
            base = _undouble_final_consonant(base)
            return (base, PRESENT_PART)

        if lower.endswith("ied") and len(lower) > 3:
            return (lower[:-3] + "y", PAST_TENSE)

        if lower.endswith("ed") and len(lower) > 3:
            base = lower[:-2]
            base = _undouble_final_consonant(base)
            return (base, PAST_TENSE)

        if lower.endswith("ies") and len(lower) > 3:
            return (lower[:-3] + "y", PLURAL)

        if (
            lower.endswith("es")
            and len(lower) > 2
            and lower[:-2].endswith(("s", "x", "z", "ch", "sh"))
        ):
            return (lower[:-2], PLURAL)

        if lower.endswith("s") and len(lower) > 2:
            return (lower[:-1], PLURAL)

        if lower.endswith("ly") and len(lower) > 3:
            base = lower[:-2]
            if base.endswith("i"):
                base = base[:-1] + "y"
            return (base, ADVERBIAL)

        if lower.endswith("est") and len(lower) > 4:
            base = lower[:-3]
            base = _undouble_final_consonant(base)
            if base.endswith("i"):
                base = base[:-1] + "y"
            return (base, SUPERLATIVE)

        if lower.endswith("er") and len(lower) > 3:
            base = lower[:-2]
            base = _undouble_final_consonant(base)
            if base.endswith("i"):
                base = base[:-1] + "y"
            return (base, COMPARATIVE)

        if lower.endswith("ness") and len(lower) > 5:
            base = lower[:-4]
            if base.endswith("i"):
                base = base[:-1] + "y"
            if base.endswith("e"):
                base = base[:-1]
            return (base, NOMINALIZE)

        return (lower, BASE)


def _undouble_final_consonant(word: str) -> str:
    if len(word) >= 2 and word[-1] == word[-2] and word[-1].isalpha():
        return word[:-1]
    return word
