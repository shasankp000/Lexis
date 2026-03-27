"""spaCy-based syntax analysis and tree-shape serialization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

_PASSIVE_SUBJ_DEPS = frozenset({"nsubjpass", "nsubj:pass"})
_PASSIVE_AUX_DEPS = frozenset({"auxpass", "aux:pass"})


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
    try:
        sent = next(doc.sents)
    except Exception:
        sent = doc

    tokens = [token.text for token in sent if not token.is_space]
    pos_tags = [token.pos_ for token in sent if not token.is_space]
    dep_labels = [token.dep_ for token in sent if not token.is_space]
    tree_shape = serialise_tree_shape(sent)

    sentence_text = sent.text.strip() if hasattr(sent, "text") else doc.text.strip()
    sentence_type = _detect_sentence_type(sent, sentence_text)
    voice = _detect_voice(sent)
    phrase_boundaries = _extract_phrase_boundaries(sent)

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


def serialise_tree_shape(sent) -> str:
    """Return a string encoding only the structural shape of the parse tree."""
    if not sent:
        return "S[]"

    sent_token_indices = {token.i for token in sent}

    def build_shape(token, visited: set[int]) -> str:
        if token.i in visited:
            return "*"
        visited = visited | {token.i}

        children = [c for c in token.children if c.i in sent_token_indices]
        children.sort(key=lambda t: t.i)

        if not children:
            return token.dep_

        inner = " ".join(build_shape(child, visited) for child in children)
        return f"{token.dep_}[{inner}]"

    root = sent.root if hasattr(sent, "root") else None
    if root is None:
        try:
            root = next(token for token in sent if token.dep_ == "ROOT")
        except StopIteration:
            root = sent[0]

    return f"S[{build_shape(root, set())}]"


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


def _detect_voice(doc) -> str:
    """Detect active or passive voice from a spaCy doc or sentence span."""
    for token in doc:
        if token.dep_ in _PASSIVE_SUBJ_DEPS:
            return "PASSIVE"
        if token.dep_ in _PASSIVE_AUX_DEPS:
            return "PASSIVE"
    return "ACTIVE"


def _extract_phrase_boundaries(doc) -> List[Tuple[int, int, str]]:
    boundaries: List[Tuple[int, int, str]] = []
    try:
        for chunk in doc.noun_chunks:
            boundaries.append((chunk.start, chunk.end - 1, "NP"))
    except Exception:
        pass
    return boundaries
