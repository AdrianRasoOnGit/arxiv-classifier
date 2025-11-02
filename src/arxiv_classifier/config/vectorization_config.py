# src/arxiv_classifier/config/vectorization_config.py

from .preprocessing_config import SPLITS_CONFIG

TFIDF_CONFIG = {
    "max_features": 30_000,
    "min_df": 2,
    "max_df": 0.8,

    "sample_limit": 60_000,

    "batch_size": 10_000,
}

NGRAM_CONFIG = {
    "max_features": 30_000,
    "min_df": 2,
    "max_df": 0.8,

    "ngram_range": (1, 3),

    "sample_limit": 40_000,

    "batch_size": 10_000,
}

EMBEDDINGS_CONFIG = {
    "model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", # alleanai-specter2, or all-MiniLM-L6-v2, paraphrase-MiniLM-L3-v2

    "sample_limit": 10_000,

    "batch_size": 2048,
}
