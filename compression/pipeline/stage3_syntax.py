"""spaCy-based syntax analysis and tree-shape serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class SyntaxResult:
    sentence: str
    tokens: List[str]
    pos_tags: List[str]
    dep_labels: List[str]
    tree_shape: str
    sentence_type: str
    voice: str
    phrase_boundaries: List[Tuple[int, int, str]]


def analyse_sentence(doc) -> SyntaxResult:
    """Given a spaCy Doc, return full SyntaxResult."""
    tokens = [token.text for token in doc if not token.is_space]
    pos_tags = [token.pos_ for token in doc if not token.is_space]
    dep_labels = [token.dep_ for token in doc if not token.is_space]
    tree_shape = serialise_tree_shape(doc)

    sentence_text = doc.text.strip()
    sentence_type = _detect_sentence_type(doc, sentence_text)
    voice = "PASSIVE" if any(token.dep_ == "auxpass" for token in doc) else "ACTIVE"
    phrase_boundaries = _extract_phrase_boundaries(doc)

    return SyntaxResult(
        sentence=sentence_text,
        tokens=tokens,
        pos_tags=pos_tags,
        dep_labels=dep_labels,
        tree_shape=tree_shape,
        sentence_type=sentence_type,
        voice=voice,
        phrase_boundaries=phrase_boundaries,
    )


def serialise_tree_shape(doc) -> str:
    """Return a string encoding only the structural shape of the parse tree."""
    if not doc:
        return "S[]"

    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break
    if root is None:
        root = doc[0]

    children_map = {token: [] for token in doc}
    for token in doc:
        if token is root:
            continue
        children_map[token.head].append(token)

    def build_shape(token) -> str:
        children = children_map.get(token, [])
        children.sort(key=lambda t: t.i)
        label = token.pos_ or token.dep_ or "X"
        if not children:
            return label
        inner = " ".join(build_shape(child) for child in children)
        return f"{label}[{inner}]"

    return f"S[{build_shape(root)}]"


def get_pos_delta_sequence(pos_tags: List[str], pos_vocab: Dict[str, int]) -> List[int]:
    """Convert POS tag sequence to integer IDs using pos_vocab, then delta-encode."""
    if not pos_tags:
        return []
    local_vocab = dict(pos_vocab)
    next_id = max(local_vocab.values(), default=-1) + 1
    ids: List[int] = []
    for tag in pos_tags:
        if tag not in local_vocab:
            local_vocab[tag] = next_id
            next_id += 1
        ids.append(local_vocab[tag])

    deltas = [ids[0]]
    deltas.extend(curr - prev for prev, curr in zip(ids, ids[1:]))
    return deltas


def _detect_sentence_type(doc, sentence_text: str) -> str:
    if sentence_text.endswith("?"):
        return "INTERROGATIVE"
    if sentence_text.endswith("!"):
        return "EXCLAMATORY"

    root = None
    for token in doc:
        if token.dep_ == "ROOT":
            root = token
            break
    if root is not None and root.pos_ == "VERB":
        has_subject = any(token.dep_ in {"nsubj", "nsubjpass"} for token in doc)
        if not has_subject:
            return "IMPERATIVE"

    return "DECLARATIVE"


def _extract_phrase_boundaries(doc) -> List[Tuple[int, int, str]]:
    boundaries: List[Tuple[int, int, str]] = []
    try:
        for chunk in doc.noun_chunks:
            boundaries.append((chunk.start, chunk.end - 1, "NP"))
    except Exception:
        pass
    return boundaries
