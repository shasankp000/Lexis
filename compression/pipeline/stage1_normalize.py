"""Simple text normalization stage for the compression pipeline."""

from __future__ import annotations

import re

# Typographic / Unicode quote and dash characters that are not in the phonetic
# map. Normalise them to their ASCII equivalents so the entire downstream
# pipeline (character encoder, phonetic map, _join_words) only ever sees
# straight ASCII punctuation.
_UNICODE_REPLACEMENTS: list[tuple[str, str]] = [
    # Single quotes / apostrophes
    ("\u2018", "'"),  # left single quotation mark
    ("\u2019", "'"),  # right single quotation mark / apostrophe
    ("\u201a", "'"),  # single low-9 quotation mark
    ("\u201b", "'"),  # single high-reversed-9 quotation mark
    ("\u2032", "'"),  # prime
    # Double quotes
    ("\u201c", '"'),  # left double quotation mark
    ("\u201d", '"'),  # right double quotation mark
    ("\u201e", '"'),  # double low-9 quotation mark
    ("\u201f", '"'),  # double high-reversed-9 quotation mark
    ("\u2033", '"'),  # double prime
    ("\u00ab", '"'),  # left-pointing double angle quotation mark
    ("\u00bb", '"'),  # right-pointing double angle quotation mark
    # Dashes
    ("\u2014", "--"),  # em dash
    ("\u2013", "-"),   # en dash
]


def normalize_text(text: str) -> str:
    """Normalize whitespace, line endings, and typographic punctuation.

    Typographic quote and dash characters that fall outside the phonetic
    alphabet are mapped to their ASCII equivalents before any other stage
    processes the text.  This guarantees lossless round-trips for possessives
    (e.g. "ahab\u2019s" -> "ahab's") and similar constructs.
    """
    if not text:
        return ""

    normalized = text

    # Typographic -> ASCII punctuation (must come before whitespace collapse
    # so that any incidental spaces introduced are handled below).
    for src, dst in _UNICODE_REPLACEMENTS:
        normalized = normalized.replace(src, dst)

    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.replace("\u00a0", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
