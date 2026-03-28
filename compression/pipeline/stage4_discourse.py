"""Stage 4: Discourse analysis — coreference resolution and discourse relations."""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

# Discourse connective lookup table
DISCOURSE_CONNECTIVES: Dict[str, str] = {
    # CAUSE
    "because": "CAUSE",
    "since": "CAUSE",
    "due to": "CAUSE",
    # CONTRAST
    "but": "CONTRAST",
    "however": "CONTRAST",
    "though": "CONTRAST",
    "although": "CONTRAST",
    "yet": "CONTRAST",
    "whereas": "CONTRAST",
    "while": "CONTRAST",
    "nevertheless": "CONTRAST",
    "nonetheless": "CONTRAST",
    # SEQUENCE
    "then": "SEQUENCE",
    "next": "SEQUENCE",
    "after": "SEQUENCE",
    "before": "SEQUENCE",
    "subsequently": "SEQUENCE",
    "finally": "SEQUENCE",
    "first": "SEQUENCE",
    "second": "SEQUENCE",
    "lastly": "SEQUENCE",
    # ELABORATION
    "for example": "ELABORATION",
    "for instance": "ELABORATION",
    "in particular": "ELABORATION",
    "such as": "ELABORATION",
    "namely": "ELABORATION",
    "specifically": "ELABORATION",
    # ADDITION
    "also": "ADDITION",
    "furthermore": "ADDITION",
    "moreover": "ADDITION",
    "additionally": "ADDITION",
    "besides": "ADDITION",
    # CONDITION
    "if": "CONDITION",
    "unless": "CONDITION",
    "provided": "CONDITION",
    # CONCLUSION
    "therefore": "CONCLUSION",
    "thus": "CONCLUSION",
    "hence": "CONCLUSION",
    "consequently": "CONCLUSION",
    "as a result": "CONCLUSION",
    "so": "CONCLUSION",
}

_MULTIWORD_CONNECTIVES = [c for c in DISCOURSE_CONNECTIVES if " " in c]
_MULTIWORD_CONNECTIVES.sort(key=lambda x: len(x.split()), reverse=True)

BOILERPLATE_TERMS: Set[str] = {
    "gutenberg",
    "ebook",
    "project",
    "tm",
    "trademark",
    "foundation",
    "license",
    "copyright",
    "electronic",
}

PRONOUNS: Set[str] = {
    "he",
    "she",
    "it",
    "they",
    "him",
    "her",
    "his",
    "their",
    "them",
    "we",
    "i",
}


def is_compression_worthy(mentions: List[Tuple]) -> bool:
    """
    Decide whether a coreference chain is worth encoding.

    A chain is worthy if:
      - It has 2+ mentions (required for any substitution)
      - It is not dominated by boilerplate terms
      - Average mention length is reasonable (< 25 chars)
      - It has at least 2 distinct surface forms OR at least 2 identical
        named-entity repeats (pure-named chains like [Ahab, Ahab] are valid)

    NOTE: The old requirement that a chain MUST contain a pronoun has been
    removed. Pure named-entity repeat chains are compression-worthy too.
    """
    surfaces = [m[2].lower().strip() for m in mentions]

    if len(mentions) < 2:
        return False

    # Skip boilerplate-dominated chains
    boilerplate_count = sum(
        any(term in s for term in BOILERPLATE_TERMS) for s in surfaces
    )
    if boilerplate_count / len(surfaces) > 0.5:
        return False

    # Skip very long average mention surfaces
    avg_len = sum(len(s) for s in surfaces) / len(surfaces)
    if avg_len > 25:
        return False

    return True


def _chunk_at_sentences(text: str, max_chars: int) -> List[Tuple[str, int]]:
    """
    Split text into chunks of ~max_chars, breaking at sentence boundaries.

    Returns list of (chunk_text, chunk_char_offset) so callers can map
    chunk-relative char positions back to absolute document positions.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[Tuple[str, int]] = []
    current = ""
    current_offset = 0
    pos = 0

    for sent in sentences:
        # Find where this sentence starts in the original text
        sent_pos = text.index(sent, pos)
        if len(current) + len(sent) > max_chars and current:
            chunks.append((current.strip(), current_offset))
            current = sent
            current_offset = sent_pos
        else:
            if not current:
                current_offset = sent_pos
            current += " " + sent if current else sent
        pos = sent_pos + len(sent)

    if current:
        chunks.append((current.strip(), current_offset))

    return chunks


def extract_coreference_chains(
    doc: Any,
    chunk_char_offset: int = 0,
) -> Dict[int, List[Tuple[int, int, str, int, int]]]:
    """
    Extract coreference chains from a fastcoref-processed spaCy doc.

    fastcoref stores clusters as character offset tuples on doc._.coref_clusters:
        [[(char_start, char_end), ...], ...]

    Returns:
        Dict mapping entity_id -> [
            (sent_idx, token_idx, surface_text, abs_char_start, abs_char_end),
            ...
        ]

    char offsets are ABSOLUTE (relative to the full document), computed by
    adding chunk_char_offset to the chunk-relative fastcoref offsets.
    """
    chains: Dict[int, List[Tuple[int, int, str, int, int]]] = {}

    if not hasattr(doc._, "coref_clusters") or not doc._.coref_clusters:
        return chains

    for entity_id, cluster in enumerate(doc._.coref_clusters):
        seen: set = set()
        mentions: List[Tuple[int, int, str, int, int]] = []

        for char_start, char_end in cluster:
            span = doc.char_span(char_start, char_end, alignment_mode="expand")
            if span is None:
                continue

            sents = list(doc.sents)
            sent_idx = next(
                (i for i, s in enumerate(sents) if s.start <= span.start < s.end), 0
            )
            token_idx = span.start - sents[sent_idx].start
            surface = doc.text[char_start:char_end]

            # Absolute char offsets in the full document
            abs_start = chunk_char_offset + char_start
            abs_end = chunk_char_offset + char_end

            key = (sent_idx, token_idx, surface.lower())
            if key not in seen:
                seen.add(key)
                mentions.append((sent_idx, token_idx, surface, abs_start, abs_end))

        if len(mentions) > 1 and is_compression_worthy(mentions):
            chains[entity_id] = mentions

    return chains


def extract_discourse_relations(doc: Any) -> List[Tuple[int, int, str, str]]:
    """
    Extract discourse connectives and their relation types.

    Returns:
        List of (sent_idx, token_idx, connective_text, relation_type)
    """
    relations: List[Tuple[int, int, str, str]] = []

    for sent_idx, sent in enumerate(doc.sents):
        tokens = [t for t in sent]
        lowered = [t.text.lower() for t in tokens]

        # Single-token connectives
        for token_idx, token in enumerate(tokens):
            text_lower = token.text.lower()
            if text_lower in DISCOURSE_CONNECTIVES:
                rel_type = DISCOURSE_CONNECTIVES[text_lower]
                relations.append((sent_idx, token_idx, token.text, rel_type))

        # Multiword connectives
        n = len(tokens)
        for token_idx in range(n):
            for phrase in _MULTIWORD_CONNECTIVES:
                parts = phrase.split()
                m = len(parts)
                if token_idx + m > n:
                    continue
                if lowered[token_idx : token_idx + m] == parts:
                    rel_type = DISCOURSE_CONNECTIVES[phrase]
                    surface = " ".join(
                        t.text for t in tokens[token_idx : token_idx + m]
                    )
                    relations.append((sent_idx, token_idx, surface, rel_type))

    return relations


class DiscourseAnalyser:
    """Analyse text for coreference chains and discourse relations."""

    def __init__(
        self,
        use_spacy: bool = True,
        model_name: str | None = None,
        device: str = "cuda:0",
    ) -> None:
        self.use_spacy = use_spacy
        self.nlp: Any | None = None

        if use_spacy:
            try:
                import spacy
                from fastcoref import spacy_component  # noqa: F401 (registers factory)

                model_to_load = model_name or "en_core_web_lg"
                self.nlp = spacy.load(model_to_load)

                if "fastcoref" not in self.nlp.pipe_names:
                    self.nlp.add_pipe(
                        "fastcoref",
                        config={
                            "model_architecture": "FCoref",
                            "device": device,
                        },
                        last=True,
                    )
            except Exception as exc:
                print(f"[Stage 4] Warning: Could not initialise fastcoref: {exc}")
                self.nlp = None

    def analyse_document(self, text: str) -> Dict:
        """Chunk long texts to stay within model token limits."""
        MAX_CHARS = 3000  # ~750 tokens, safety ceiling for model context windows

        if not self.nlp:
            return {"coreference_chains": {}, "discourse_relations": []}

        all_chains: Dict[int, List[Tuple]] = {}
        all_relations: List[Tuple[int, int, str, str]] = []
        entity_offset = 0

        chunks = _chunk_at_sentences(text, MAX_CHARS)
        for chunk_text, chunk_char_offset in chunks:
            try:
                doc = self.nlp(
                    chunk_text,
                    component_cfg={"fastcoref": {"resolve_text": True}},
                )
            except Exception:
                doc = self.nlp(chunk_text)

            chunk_chains = extract_coreference_chains(doc, chunk_char_offset)
            for eid, mentions in chunk_chains.items():
                all_chains[entity_offset + eid] = mentions
            entity_offset += len(chunk_chains)
            all_relations.extend(extract_discourse_relations(doc))

        return {"coreference_chains": all_chains, "discourse_relations": all_relations}

    def analyse_sentence(self, text: str) -> Dict:
        """
        Analyse a single sentence.

        Note: coreference across sentences won't be detected in single-sentence mode.
        """
        if not self.nlp:
            return {"coreference_chains": {}, "discourse_relations": []}

        try:
            doc = self.nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
        except Exception:
            doc = self.nlp(text)

        return {
            "coreference_chains": extract_coreference_chains(doc, 0),
            "discourse_relations": extract_discourse_relations(doc),
        }
