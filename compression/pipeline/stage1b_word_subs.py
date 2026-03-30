"""Stage 1b: Frequency-based whole-word substitution.

Replaces words that appear >= MIN_FREQ times in the input with compact
§W{n} tokens. This reduces the character stream length before morphological
analysis, directly shrinking char_classes and pos_deltas.

Design decisions
----------------
- Whole-word substitution only (regex word boundaries) to avoid partial
  matches inside longer words.
- Case-sensitive matching: 'The' and 'the' are tracked separately.
  Both are substituted if each independently meets MIN_FREQ.
- Symbols are assigned in descending frequency order so the most common
  words get the shortest symbols (§W1, §W2, ...).
- The anchor occurrence (first occurrence) is also substituted — unlike
  the entity coreference stage which preserves the anchor. Here every
  occurrence is replaced because the word form is fully recoverable from
  the symbol table.
- Excluded: words that are already §E or §R symbols, and any word whose
  substitution symbol is longer than the word itself (no-win cases).

Round-trip guarantee
--------------------
    text == decode_word_subs(encode_word_subs(text, min_freq)[0],
                             encode_word_subs(text, min_freq)[1])

Wire format
-----------
    Stored as a plain dict {"§W1": "the", "§W2": "of", ...} in the payload
    alongside the entity symbol_table. decode_symbols in
    stage5_discourse_symbols handles both transparently.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, List, Tuple

# Minimum occurrences for a word to be substituted
MIN_FREQ: int = 5

# Never substitute these even if frequent — they are structural tokens
# or single chars whose symbol would be longer than the word.
_STOPLIST: frozenset = frozenset({
    "a", "A",   # 1-char: §W1 (3 chars) is longer
    "I",        # same
})

_SYMBOL_RE = re.compile(r"§[EWR]\d+")
_WORD_RE   = re.compile(r"\b\w[\w'\-]*\b")  # whole word, allows apostrophes


def encode_word_subs(
    text: str,
    min_freq: int = MIN_FREQ,
) -> Tuple[str, Dict[str, str]]:
    """
    Substitute high-frequency words with §W{n} tokens.

    Returns
    -------
    compressed : str
        Text with frequent words replaced.
    word_table : Dict[str, str]
        Mapping {"§W1": "the", ...} for all substituted words.
    """
    # Count word frequencies, ignoring already-substituted symbols
    words_in_text = [
        m.group() for m in _WORD_RE.finditer(text)
        if not _SYMBOL_RE.match(m.group())
    ]
    freq: Counter = Counter(words_in_text)

    # Select candidates: freq >= min_freq, not in stoplist,
    # and symbol must be shorter than the word
    candidates: List[Tuple[str, int]] = [
        (word, count)
        for word, count in freq.items()
        if count >= min_freq
        and word not in _STOPLIST
        and not _SYMBOL_RE.match(word)
    ]

    # Sort by frequency descending so §W1 = most common
    candidates.sort(key=lambda x: -x[1])

    # Assign symbols — only include if symbol is strictly shorter than word
    word_table: Dict[str, str] = {}
    symbol_idx = 1
    for word, _ in candidates:
        sym = f"\u00a7W{symbol_idx}"  # §W{n}
        if len(sym) < len(word):       # only substitute if we save bytes
            word_table[sym] = word
            symbol_idx += 1

    if not word_table:
        return text, {}

    # Build reverse map word -> symbol, longest word first to avoid
    # partial replacement (e.g. 'there' before 'the')
    word_to_sym: Dict[str, str] = {v: k for k, v in word_table.items()}
    sorted_words = sorted(word_to_sym.keys(), key=lambda w: -len(w))

    # Replace each word using whole-word regex
    compressed = text
    for word in sorted_words:
        sym = word_to_sym[word]
        # Use word boundary — but \b fails on non-alphanumeric chars,
        # so use lookahead/lookbehind for non-word chars or string edges
        pattern = r"(?<![\w'\-])" + re.escape(word) + r"(?![\w'\-])"
        compressed = re.sub(pattern, sym, compressed)

    return compressed, word_table


def decode_word_subs(text: str, word_table: Dict[str, str]) -> str:
    """
    Restore substituted words. Symbols replaced longest-first.
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
