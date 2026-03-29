"""Stage 9 — Post-decompress rule-based autocorrector.

Applied as the very last step after _join_words() and decode_symbols().
Zero external dependencies; pure regex over the joined string.

Artifact classes targeted
--------------------------
1. Opening-quote gap      :  '" word'  ->  '"word'  (space after opening quote removed)
2. Closing-quote gap      :  'word " ,'  ->  'word",'
2c. Period+quote+word     :  'end."next'  ->  'end. "next'  (opening quote after sentence)
3. Hyphenated compounds   :  'whale - fish'  ->  'whale-fish'
4. Em-dash / double-hyphen:  ' -- '  ->  '--'
5. Bracket inner spacing  :  '( word )'  ->  '(word)'
6. Apostrophe contractions:  "do n't"  ->  "don't"
7. Collapsed whitespace   :  multiple spaces -> single space
   (guarded: '* * *' separators are preserved)
"""

from __future__ import annotations

import re

# Placeholder used to temporarily protect '* * *' from space-collapse.
_STAR_SEP_PLACEHOLDER = "\x00STARSEP\x00"
_STAR_SEP_RE = re.compile(r'\* \* \*')
_STAR_SEP_RESTORE = re.compile(re.escape(_STAR_SEP_PLACEHOLDER))

# ---------------------------------------------------------------------------
# Compiled patterns (order matters — applied top-to-bottom)
# ---------------------------------------------------------------------------

_RULES: list[tuple[re.Pattern, str]] = [
    # 1. Opening-quote gap: '" word' -> '"word'
    #    Only collapse when followed by a lowercase letter — uppercase
    #    means it is a sentence-opening quote which keeps its space.
    (re.compile(r'" +([a-z])'), r'"\1'),

    # 2a. Closing quote before punctuation: 'word " ,' -> 'word",'
    (re.compile(r'(\w) "([,\.!\?;:])'), r'\1"\2'),
    # 2b. Closing quote followed by space (end of quoted clause):
    #     'word " ' -> 'word" '
    (re.compile(r'(\w) " '), r'\1" '),
    # 2c. Period+opening-quote glued to next word: 'end."next' -> 'end. "next'
    #     _join_words glues '"' left after '.' (correct for closing quotes)
    #     but when '"' is actually an opening quote for the next sentence
    #     we need to re-insert the space between '"' and the following word.
    (re.compile(r'(\.")(\w)'), r'\1 \2'),

    # 3. Hyphenated compounds: 'whale - fish' -> 'whale-fish'
    #    Guard: only between word characters, not list-item dashes.
    (re.compile(r'(?<=[\w]) - (?=[\w])'), '-'),

    # 4. Double-hyphen (ASCII em-dash from normalize_text): ' -- ' -> '--'
    #    normalize_text maps em-dash -> '--', so restore tight spacing.
    (re.compile(r' -- '), '--'),

    # 5a. Opening bracket inner space: '( word' -> '(word'
    (re.compile(r'\( +'), '('),
    # 5b. Closing bracket inner space: 'word )' -> 'word)'
    (re.compile(r' +\)'), ')'),

    # 6. Apostrophe contractions split by spaCy:
    #    "do n't" -> "don't",  "I 'm" -> "I'm",  "it 's" -> "it's"
    (re.compile(r"(\w) (n't|'s|'re|'ve|'ll|'d|'m|'em)\b"), r"\1\2"),

    # 7. Collapse multiple spaces to single space.
    #    ('* * *' has been replaced by placeholder before this runs.)
    (re.compile(r'  +'), ' '),
]


def autocorrect(text: str) -> str:
    """Apply all correction rules in sequence and return the cleaned text."""
    # Protect '* * *' section separators before space-collapse.
    text = _STAR_SEP_RE.sub(_STAR_SEP_PLACEHOLDER, text)

    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)

    # Restore '* * *' separators.
    text = _STAR_SEP_RESTORE.sub('* * *', text)
    return text.strip()
