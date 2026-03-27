"""Central configuration for the compression pipeline."""

# spaCy model to use for NLP processing.
SPACY_MODEL = "en_core_web_lg"

# Maximum character length per nlp() call.
SPACY_MAX_LENGTH = 2_000_000

# Context window sizes for Stage 6 context mixing.
CHAR_CONTEXT_SIZE = 32
MORPH_CONTEXT_SIZE = 24
STRUCT_CONTEXT_SIZE = 34
