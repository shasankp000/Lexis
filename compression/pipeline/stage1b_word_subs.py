"""Stage 1b: Frequency-based whole-word substitution.

Replaces words that appear >= MIN_FREQ times in the input with compact
§W{n} tokens. This reduces the character stream length before morphological
analysis, directly shrinking char_classes and pos_deltas.

Design decisions
----------------
- Pure alpha tokens only ([A-Za-z]+). No apostrophes, hyphens, or digits.
- Case-sensitive: 'The' and 'the' tracked and substituted independently.
- Symbols assigned in descending frequency order (§W1 = most common).
- ALL occurrences substituted (unlike entity stage which keeps anchor).
- Net-saving guard: only substitute if the total char saving across all
  occurrences exceeds SLOT_OVERHEAD_CHARS. This prevents marginal subs
  from increasing bpb due to slot placeholder + slot_map payload cost.

Net-saving formula
------------------
    net = freq * (len(word) - SLOT_PLACEHOLDER_LEN) - SLOT_OVERHEAD_CHARS
    Substitute only if net > 0.

    SLOT_PLACEHOLDER_LEN = 3  (zza..zzz for slots 0-25)
    SLOT_OVERHEAD_CHARS  = 20  (conservative: slot_map entry + zstd context)

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

# Slot placeholder length for slots 0-25 (zza..zzz = 3 chars).
# For slots 26+ it's 4 chars (zzaa..), but 3 is the common case.
SLOT_PLACEHOLDER_LEN: int = 3

# Conservative estimate of per-word-type overhead from adding a slot:
# slot_map msgpack entry (~6 bytes) + zstd context disruption (~14 chars equiv)
SLOT_OVERHEAD_CHARS: int = 20

_SYMBOL_RE = re.compile(r"§[EWR]\d+")
_WORD_RE   = re.compile(r"\b[A-Za-z]+\b")


def encode_word_subs(
    text: str,
    min_freq: int = MIN_FREQ,
) -> Tuple[str, Dict[str, str]]:
    """
    Substitute high-frequency pure-alpha words with §W{n} tokens.

    Only substitutes when net char saving > 0 after accounting for slot
    placeholder length and slot_map payload overhead.

    Returns
    -------
    compressed : str
        Text with frequent words replaced.
    word_table : Dict[str, str]
        Mapping {"§W1": "the", ...} for all substituted words.
    """
    words_in_text = [
        m.group() for m in _WORD_RE.finditer(text)
        if not _SYMBOL_RE.match(m.group())
    ]
    freq: Counter = Counter(words_in_text)

    candidates: List[Tuple[str, int]] = [
        (word, count)
        for word, count in freq.items()
        if count >= min_freq
        and len(word) > SLOT_PLACEHOLDER_LEN   # must save at least 1 char/occurrence
        and not _SYMBOL_RE.match(word)
        # Net-saving guard: total saving across all occurrences > overhead
        and count * (len(word) - SLOT_PLACEHOLDER_LEN) > SLOT_OVERHEAD_CHARS
    ]

    candidates.sort(key=lambda x: -x[1])

    word_table: Dict[str, str] = {}
    symbol_idx = 1
    for word, _ in candidates:
        sym = f"\u00a7W{symbol_idx}"
        word_table[sym] = word
        symbol_idx += 1

    if not word_table:
        return text, {}

    word_to_sym: Dict[str, str] = {v: k for k, v in word_table.items()}
    sorted_words = sorted(word_to_sym.keys(), key=lambda w: -len(w))

    compressed = text
    for word in sorted_words:
        sym = word_to_sym[word]
        pattern = r"\b" + re.escape(word) + r"\b"
        compressed = re.sub(pattern, sym, compressed)

    return compressed, word_table


def decode_word_subs(text: str, word_table: Dict[str, str]) -> str:
    """Restore substituted words. Replace longest symbols first."""
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
