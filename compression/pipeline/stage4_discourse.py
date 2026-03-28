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

    A chain is worthy if it has 2+ mentions, is not boilerplate-dominated,
    and has a reasonable average mention length. Pure named-entity repeat
    chains are valid — no pronoun requirement.
    """
    surfaces = [m[2].lower().strip() for m in mentions]

    if len(mentions) < 2:
        return False

    boilerplate_count = sum(
        any(term in s for term in BOILERPLATE_TERMS) for s in surfaces
    )
    if boilerplate_count / len(surfaces) > 0.5:
        return False

    avg_len = sum(len(s) for s in surfaces) / len(surfaces)
    if avg_len > 25:
        return False

    return True


def _chunk_at_sentences(text: str, max_chars: int) -> List[Tuple[str, int]]:
    """
    Split `text` into chunks of ~max_chars, breaking only at sentence boundaries.

    Returns list of (chunk_text, chunk_char_offset) where:
      - chunk_text   is a verbatim slice of `text` (no synthetic spaces added)
      - chunk_char_offset is the absolute index of chunk_text[0] in `text`

    By slicing the original text directly we guarantee that
    fastcoref's chunk-relative char offsets + chunk_char_offset == absolute
    document offsets with no off-by-one from join separators.
    """
    import re

    # Find the start position of every sentence in the original text
    sentence_starts: List[int] = []
    pos = 0
    for sent in re.split(r"(?<=[.!?])\s+", text):
        idx = text.index(sent, pos)
        sentence_starts.append(idx)
        pos = idx + len(sent)

    chunks: List[Tuple[str, int]] = []
    chunk_start_idx = 0       # index into sentence_starts for current chunk
    chunk_char_start = sentence_starts[0] if sentence_starts else 0

    for i, sent_start in enumerate(sentence_starts):
        # Determine end of this sentence in the original text
        if i + 1 < len(sentence_starts):
            next_sent_start = sentence_starts[i + 1]
        else:
            next_sent_start = len(text)

        chunk_so_far_len = next_sent_start - chunk_char_start

        # Flush when adding this sentence would exceed max_chars
        # (but always include at least one sentence per chunk)
        if chunk_so_far_len > max_chars and i > chunk_start_idx:
            chunk_end = sent_start  # end of previous sentence
            chunks.append((text[chunk_char_start:chunk_end], chunk_char_start))
            chunk_start_idx = i
            chunk_char_start = sent_start

    # Final chunk
    if chunk_start_idx < len(sentence_starts):
        chunks.append((text[chunk_char_start:], chunk_char_start))

    return chunks


def extract_coreference_chains(
    doc: Any,
    chunk_char_offset: int = 0,
) -> Dict[int, List[Tuple[int, int, str, int, int]]]:
    """
    Extract coreference chains from a fastcoref-processed spaCy doc.

    Returns:
        Dict mapping entity_id -> [
            (sent_idx, token_idx, surface_text, abs_char_start, abs_char_end),
            ...
        ]

    surface_text and abs char offsets use raw fastcoref char_start/char_end.
    char_span(expand) is used ONLY for sent_idx/token_idx bookkeeping.
    """
    chains: Dict[int, List[Tuple[int, int, str, int, int]]] = {}

    if not hasattr(doc._, "coref_clusters") or not doc._.coref_clusters:
        return chains

    for entity_id, cluster in enumerate(doc._.coref_clusters):
        seen: set = set()
        mentions: List[Tuple[int, int, str, int, int]] = []

        for char_start, char_end in cluster:
            # Raw surface — exact, no token-boundary expansion
            surface = doc.text[char_start:char_end]

            # char_span only for sent/token index (not for position)
            span = doc.char_span(char_start, char_end, alignment_mode="expand")
            if span is not None:
                sents = list(doc.sents)
                sent_idx = next(
                    (i for i, s in enumerate(sents) if s.start <= span.start < s.end),
                    0,
                )
                token_idx = span.start - sents[sent_idx].start
            else:
                sent_idx = 0
                token_idx = 0

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

    Returns list of (sent_idx, token_idx, connective_text, relation_type).
    """
    relations: List[Tuple[int, int, str, str]] = []

    for sent_idx, sent in enumerate(doc.sents):
        tokens = [t for t in sent]
        lowered = [t.text.lower() for t in tokens]

        for token_idx, token in enumerate(tokens):
            text_lower = token.text.lower()
            if text_lower in DISCOURSE_CONNECTIVES:
                rel_type = DISCOURSE_CONNECTIVES[text_lower]
                relations.append((sent_idx, token_idx, token.text, rel_type))

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
                from fastcoref import spacy_component  # noqa: F401

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
        MAX_CHARS = 3000

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
        """Analyse a single sentence (no cross-sentence coreference)."""
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
