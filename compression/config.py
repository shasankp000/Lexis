"""Central configuration for the compression pipeline."""

# spaCy model to use for NLP processing.
SPACY_MODEL = "en_core_web_lg"

# Maximum character length per nlp() call.
SPACY_MAX_LENGTH = 2_000_000
