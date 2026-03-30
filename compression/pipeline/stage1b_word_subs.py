"""Stage 1b: Frequency-based whole-word substitution.

Replaces words that appear >= MIN_FREQ times in the input with compact
§W{n} tokens. This reduces the character stream length before morphological
analysis, directly shrinking char_classes and pos_deltas.

Design decisions
----------------
- Pure alpha tokens only ([A-Za-z]+). No apostrophes, hyphens, or digits.
  This guarantees \\b word boundaries work correctly for both counting
  and substitution — no mismatch between what is counted and replaced.
- Case-sensitive: 'The' and 'the' tracked and substituted independently.
- Symbols assigned in descending frequency order (§W1 = most common).
- ALL occurrences substituted (unlike entity stage which keeps anchor).
- Words of length <= 2 excluded: §W1 is 3 chars, never a net saving.
- Substitution order: longest words first to prevent prefix collisions.

Round-trip guarantee
--------------------
    text == decode_word_subs(encode_word_subs(text, min_freq)[0],
                             encode_word_subs(text, min_freq)[1])
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

MIN_FREQ: int = 5

_SYMBOL_RE = re.compile(r"§[EWR]\d+")
# Pure alpha tokens only — guarantees \b boundaries are unambiguous
_WORD_RE   = re.compile(r"\b[A-Za-z]+\b")


def encode_word_subs(
    text: str,
    min_freq: int = MIN_FREQ,
) -> Tuple[str, Dict[str, str]]:
    """
    Substitute high-frequency pure-alpha words with §W{n} tokens.

    Returns
    -------
    compressed : str
        Text with frequent words replaced.
    word_table : Dict[str, str]
        Mapping {"§W1": "the", ...} for all substituted words.
    """
    # Count pure-alpha word frequencies, skip existing symbols
    words_in_text = [
        m.group() for m in _WORD_RE.finditer(text)
        if not _SYMBOL_RE.match(m.group())
    ]
    freq: Counter = Counter(words_in_text)

    # Candidates: freq >= min_freq, length > 2 (so symbol §Wn saves bytes),
    # not already a symbol
    candidates: List[Tuple[str, int]] = [
        (word, count)
        for word, count in freq.items()
        if count >= min_freq
        and len(word) > 2
        and not _SYMBOL_RE.match(word)
    ]

    # Sort descending by frequency — §W1 = most common word
    candidates.sort(key=lambda x: -x[1])

    # Assign symbols — only if symbol length < word length
    word_table: Dict[str, str] = {}
    symbol_idx = 1
    for word, _ in candidates:
        sym = f"\u00a7W{symbol_idx}"
        if len(sym) < len(word):
            word_table[sym] = word
            symbol_idx += 1

    if not word_table:
        return text, {}

    # Reverse map: word -> symbol; sort longest-first to avoid prefix issues
    word_to_sym: Dict[str, str] = {v: k for k, v in word_table.items()}
    sorted_words = sorted(word_to_sym.keys(), key=lambda w: -len(w))

    compressed = text
    for word in sorted_words:
        sym = word_to_sym[word]
        # Pure alpha word: \b boundaries are unambiguous
        pattern = r"\b" + re.escape(word) + r"\b"
        compressed = re.sub(pattern, sym, compressed)

    return compressed, word_table


def decode_word_subs(text: str, word_table: Dict[str, str]) -> str:
    """
    Restore substituted words. Replace longest symbols first.
    """
    result = text
    for sym, word in sorted(word_table.items(), key=lambda x: -len(x[0])):
        result = result.replace(sym, word)
    return result


def merge_symbol_tables(
    entity_table: Dict[str, str],
    word_table:   Dict[str, str],
) -> Dict[str, str]:
    """Merge entity + word substitution tables into one payload dict."""
    merged = dict(entity_table)
    merged.update(word_table)
    return merged
