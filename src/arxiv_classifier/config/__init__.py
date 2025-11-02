# src/arxiv_classifier/config/__init__.py

from .preprocessing_config import SPLITS_CONFIG
from .models_config import FMLP_CONFIG, FMLP_PROFILES_CONFIG
from .vectorization_config import TFIDF_CONFIG, NGRAM_CONFIG, EMBEDDINGS_CONFIG


__all__ = [
    "SPLITS_CONFIG",
    "FMLP_CONFIG",
    "FMLP_PROFILES_CONFIG",
    "TFIDF_CONFIG",
    "NGRAM_CONFIG",
    "EMBEDDINGS_CONFIG",
]
