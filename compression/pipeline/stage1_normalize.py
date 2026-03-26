"""Simple text normalization stage for the compression pipeline."""

from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """Normalize whitespace and line endings without altering semantic content."""
    if not text:
        return ""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
