"""Stage 9 — Post-decompress rule-based autocorrector.

Applied as the very last step after _join_words() and decode_symbols().
Zero external dependencies; pure regex over the joined string.

Artifact classes targeted
--------------------------
1. Opening-quote gap      :  '.  "word'  →  '. "word'
   _join_words emits ' "' (space + quote) for opening quotes, but the
   *next* word then also gets a leading space → double gap removed here.
2. Closing-quote gap      :  'word " ,'  →  'word",'
3. Hyphenated compounds   :  'whale - fish'  →  'whale-fish'
4. Em-dash spacing        :  ' — '  →  '—'
5. Bracket inner spacing  :  '( word )'  →  '(word)'
6. Apostrophe contractions:  "do n't"  →  "don't"
7. Collapsed whitespace   :  multiple spaces → single space
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Compiled patterns (order matters — applied top-to-bottom)
# ---------------------------------------------------------------------------

_RULES: list[tuple[re.Pattern, str]] = [
    # 1. Opening quote: remove the extra space BETWEEN the quote and the
    #    following word.  Pattern: '"' followed by one or more spaces then
    #    a word character.
    #    e.g. '" while'  →  '"while'
    (re.compile(r'" +([\w])'), r'"\1'),

    # 2. Closing quote: remove space BEFORE '"' when followed by punctuation
    #    or end-of-string.
    #    e.g. 'whale " ,'  →  'whale",'
    (re.compile(r'(\w) "([,\.!\?;:\)])'), r'\1"\2'),
    # Also: bare closing quote glued to next punctuation with spurious space
    #    e.g. 'mortality ."'  — already handled by _ATTACH_LEFT, but catch
    #    any residual 'word " ' at end of clause.
    (re.compile(r'(\w) "( )'), r'\1" '),

    # 3. Hyphenated compounds: spaCy splits 'whale-fish' into
    #    ['whale', '-', 'fish'].  _join_words inserts spaces either side
    #    of '-' when it appears as a standalone token.
    #    Pattern: word SPACE hyphen SPACE word  →  word-word
    #    Guard: do NOT collapse '- ' at start of a list item (line-initial).
    (re.compile(r'(?<=[\w]) - (?=[\w])'), '-'),

    # 4. Em-dash: normalise ' — ' → '—'  (already in _join_words but may
    #    survive if the em-dash came through a different path).
    (re.compile(r' — '), '—'),

    # 5a. Opening bracket inner space: '( word'  →  '(word'
    (re.compile(r'\( +'), '('),
    # 5b. Closing bracket inner space: 'word )'  →  'word)'
    (re.compile(r' +\)'), ')'),

    # 6. Apostrophe contractions split by spaCy:
    #    "do n't" → "don't",  "I 'm" → "I'm",  "it 's" → "it's"
    #    Pattern: word boundary, space, apostrophe-prefixed suffix.
    (re.compile(r"(\w) (n't|'s|'re|'ve|'ll|'d|'m|'em)\b"), r"\1\2"),

    # 7. Collapse any run of multiple spaces to a single space.
    (re.compile(r'  +'), ' '),
]


def autocorrect(text: str) -> str:
    """Apply all correction rules in sequence and return the cleaned text."""
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text.strip()
