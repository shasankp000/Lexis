"""Discourse analysis placeholder stage.

This module provides minimal scaffolding for discourse and coreference analysis.
It is intentionally conservative and can be expanded once research decisions are
finalized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class DiscourseResult:
    sentence: str
    coreference_links: List[Tuple[int, int]]
    discourse_relations: List[Tuple[int, int, str]]
    literary_devices: List[Tuple[int, int, str]]


class DiscourseAnalyser:
    """Placeholder analyser for discourse-level signals.

    The current implementation returns empty structures. It exists to keep the
    pipeline wiring stable until a full discourse model is specified.
    """

    def analyse_sentence(self, sentence: str) -> DiscourseResult:
        """Return discourse annotations for a single sentence."""
        return DiscourseResult(
            sentence=sentence,
            coreference_links=[],
            discourse_relations=[],
            literary_devices=[],
        )

    def analyse_document(self, sentences: List[str]) -> List[DiscourseResult]:
        """Return discourse annotations for a list of sentences."""
        return [self.analyse_sentence(sentence) for sentence in sentences]
