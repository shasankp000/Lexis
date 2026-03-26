"""Symbol alphabet scaffolding for structural and lexical symbols."""

from __future__ import annotations

import string
from typing import Dict, Iterable, List, Tuple

from compression.alphabet.morph_codes import MORPH_CODE_NAMES

SENTENCE_TYPES: Tuple[str, ...] = (
    "DECLARATIVE",
    "INTERROGATIVE",
    "IMPERATIVE",
    "EXCLAMATORY",
)

PHRASE_LABELS: Tuple[str, ...] = (
    "NP",
    "VP",
    "PP",
    "ADJP",
    "ADVP",
    "SBAR",
    "S",
)

DISCOURSE_RELATIONS: Tuple[str, ...] = (
    "CAUSE",
    "CONTRAST",
    "SEQUENCE",
    "ELABORATION",
    "CONDITION",
    "ADDITION",
    "SUMMARY",
)

SPECIAL_SYMBOLS: Tuple[str, ...] = (
    "^",
    "$",
    "_",
    ".",
)

CHAR_SYMBOLS: Tuple[str, ...] = tuple(string.ascii_lowercase) + SPECIAL_SYMBOLS

POS_TAGS: Tuple[str, ...] = (
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
)

DEP_LABELS: Tuple[str, ...] = (
    "ROOT",
    "acl",
    "acomp",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "auxpass",
    "case",
    "cc",
    "ccomp",
    "compound",
    "conj",
    "cop",
    "csubj",
    "csubjpass",
    "dep",
    "det",
    "dobj",
    "iobj",
    "mark",
    "neg",
    "nmod",
    "nsubj",
    "nsubjpass",
    "nummod",
    "obj",
    "obl",
    "parataxis",
    "pcomp",
    "pobj",
    "poss",
    "prep",
    "prt",
    "punct",
    "relcl",
    "xcomp",
)

MORPH_CODE_SYMBOLS: Tuple[str, ...] = tuple(
    MORPH_CODE_NAMES[code] for code in sorted(MORPH_CODE_NAMES)
)

DEFAULT_SYMBOLS: Tuple[str, ...] = (
    *SENTENCE_TYPES,
    *PHRASE_LABELS,
    *DISCOURSE_RELATIONS,
    *SPECIAL_SYMBOLS,
)

FULL_SYMBOLS: Tuple[str, ...] = (
    *DEFAULT_SYMBOLS,
    *MORPH_CODE_SYMBOLS,
    *POS_TAGS,
    *DEP_LABELS,
    *CHAR_SYMBOLS,
)

__all__ = [
    "SENTENCE_TYPES",
    "PHRASE_LABELS",
    "DISCOURSE_RELATIONS",
    "SPECIAL_SYMBOLS",
    "CHAR_SYMBOLS",
    "POS_TAGS",
    "DEP_LABELS",
    "MORPH_CODE_SYMBOLS",
    "DEFAULT_SYMBOLS",
    "FULL_SYMBOLS",
    "SymbolAlphabet",
    "build_default_alphabet",
    "build_character_alphabet",
    "build_full_alphabet",
    "get_symbol_id",
    "get_symbol_name",
]


class SymbolAlphabet:
    """Mutable symbol alphabet with ID registration."""

    def __init__(self, symbols: Iterable[str] | None = None) -> None:
        self.symbol_to_id: Dict[str, int] = {}
        self.id_to_symbol: Dict[int, str] = {}
        if symbols:
            for symbol in symbols:
                self.register(symbol)

    def register(self, symbol: str) -> int:
        """Register a symbol and return its ID."""
        if symbol in self.symbol_to_id:
            return self.symbol_to_id[symbol]
        symbol_id = len(self.symbol_to_id)
        self.symbol_to_id[symbol] = symbol_id
        self.id_to_symbol[symbol_id] = symbol
        return symbol_id

    def get_id(self, symbol: str, add: bool = False) -> int:
        """Lookup a symbol ID. Register if add=True, else return -1."""
        if symbol in self.symbol_to_id:
            return self.symbol_to_id[symbol]
        return self.register(symbol) if add else -1

    def get_symbol(self, symbol_id: int) -> str:
        """Lookup a symbol name by ID."""
        return self.id_to_symbol.get(symbol_id, "UNKNOWN")

    @property
    def size(self) -> int:
        """Return number of registered symbols."""
        return len(self.symbol_to_id)


def _dedupe(symbols: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for symbol in symbols:
        if symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return ordered


def build_default_alphabet(symbols: Iterable[str] | None = None) -> Dict[str, int]:
    """Build a deterministic symbol-to-ID mapping."""
    symbol_list: List[str] = (
        list(symbols) if symbols is not None else list(DEFAULT_SYMBOLS)
    )
    return {symbol: idx for idx, symbol in enumerate(symbol_list)}


def build_character_alphabet() -> Dict[str, int]:
    """Build a character-level alphabet including special symbols."""
    return {symbol: idx for idx, symbol in enumerate(_dedupe(CHAR_SYMBOLS))}


def build_full_alphabet() -> Dict[str, int]:
    """Build a full symbol alphabet covering structural and lexical symbols."""
    return {symbol: idx for idx, symbol in enumerate(_dedupe(FULL_SYMBOLS))}


def get_symbol_id(symbol: str, alphabet: Dict[str, int]) -> int:
    """Lookup a symbol ID with a safe fallback."""
    return alphabet.get(symbol, -1)


def get_symbol_name(symbol_id: int, alphabet: Dict[str, int]) -> str:
    """Reverse lookup from ID to symbol name."""
    reverse = {idx: symbol for symbol, idx in alphabet.items()}
    return reverse.get(symbol_id, "UNKNOWN")
