"""Bengali morphological analyser for the Lexis pipeline.

Bengali is an agglutinative language with no English-style suffix
inflection table and no publicly available spaCy model.  This analyser
therefore operates as an *identity* pass: every whitespace-delimited
token is returned as (surface, surface, BASE) — original form, root
(same), morph code 0.

Sentence splitting uses Bengali dandas (। ॥) and newlines rather than
English full-stops.

Public API mirrors MorphologicalAnalyser:
    analyse(word)          -> (surface, root, code)
    analyse_sentence(text) -> list[(surface, root, code)]
    char_savings(text)     -> dict
    split_sentences(text)  -> list[str]
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from compression.alphabet.morph_codes import BASE

# Bengali sentence-ending markers (danda, double-danda) plus newlines
_DANDA_RE = re.compile(r"[।॥\n]+")


class BengaliMorphologicalAnalyser:
    """Identity-root morphological analyser for Bengali text."""

    def __init__(self, **_kwargs) -> None:
        # Accept and silently ignore any kwargs that MorphologicalAnalyser
        # accepts (use_spacy, model_name, …) so call-sites can swap them
        # without changing keyword arguments.
        self.nlp = None  # no spaCy model needed

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def analyse(self, word: str) -> Tuple[str, str, int]:
        """Return (surface, root=surface, BASE) for a single token."""
        return (word, word, BASE)

    def analyse_sentence(self, text: str) -> List[Tuple[str, str, int]]:
        """Tokenise *text* on whitespace; return one triple per token."""
        tokens = text.split()
        return [(tok, tok, BASE) for tok in tokens]

    def char_savings(self, text: str) -> Dict[str, float]:
        """Return morphological compression statistics.

        For Bengali the identity root means zero char savings from
        morphology.  We still return the expected dict shape.
        """
        chars = len(text.replace(" ", ""))
        return {
            "original_chars": float(chars),
            "root_chars": float(chars),
            "chars_saved": 0.0,
            "pct_saved": 0.0,
        }

    def split_sentences(self, text: str) -> List[str]:
        """Split Bengali text on dandas and newlines."""
        parts = [p.strip() for p in _DANDA_RE.split(text)]
        return [p for p in parts if p]
