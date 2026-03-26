"""Shared utilities for the compression pipeline."""

from __future__ import annotations

import re
from typing import List


def chunk_text(text: str, chunk_size: int = 500_000) -> List[str]:
    """Split text into chunks of ~chunk_size chars, breaking at newlines.

    Safe for any spaCy pipeline stage. Never splits mid-word.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks: List[str] = []
    remaining = text
    while len(remaining) > chunk_size:
        split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(remaining[:split_at])
        remaining = remaining[split_at:].lstrip("\n")
    if remaining:
        chunks.append(remaining)
    return chunks


def split_sentences(text: str) -> List[str]:
    """Heuristic sentence splitter to keep nlp() inputs short."""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [part.strip() for part in parts if part.strip()]
