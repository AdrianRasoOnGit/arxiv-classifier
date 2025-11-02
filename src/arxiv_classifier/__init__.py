# src/arxiv_classifier/__init__.py

# project file paths
from .paths import (
    # Preprocessing
    raw_json_path,
    raw_parquet_path,
    dataset_path,
    clean_dataset_path,
    taxonomy_path,
    labels_path,
    ds_labels_path,

    # Splits
    train_path,
    test_path,

    # TF-IDF vectors
    X_train_tfidf_path,
    y_train_tfidf_path,
    X_test_tfidf_path,
    y_test_tfidf_path,
    vectorizer_tfidf_path,

    # N-gram vectors
    X_train_ngram_path,
    y_train_ngram_path,
    X_test_ngram_path,
    y_test_ngram_path,
    vectorizer_ngram_path,

    # Embeddings vectors
    X_train_embeddings_path,
    y_train_embeddings_path,
    X_test_embeddings_path,
    y_test_embeddings_path,
    vectorizer_embeddings_path,

    # Weights
    weights_tfidf_path,
    weights_ngram_path,
    weights_embeddings_path,

    # Report summary
    summary_path,
    
    # Chart reports
    tfidf_training_loss_path,
    tfidf_confusion_matrix_path,

    # Metrics reports
    tfidf_classification_report_path,
)

# config/
from . import config

# utils/
from .utils.restart_file import restart_file
from .utils.batch_load import batch_load
from .utils.adam_gradient import AdamOptimizer
from .utils.load_training_data import load_training_data
from .utils.log_training_summary import log_training_summary


# preprocessing/
from .preprocessing.get_raw import get_raw
from .preprocessing.json_to_parquet import convert_json_to_parquet
from .preprocessing.taxonomy_scraper import extract_taxonomy, save_taxonomy
from .preprocessing.grab_labels import extract_labels, save_labels, build_labels
from .preprocessing.raw_parquet_to_dataset import prepare_batch, drop_duplicates, convert_parquet_to_dataset
from .preprocessing.dataset_to_clean_dataset import(
    load_canonical_labels,
    get_observed_labels,
    summarize_labels,
    filter_dataset_labels,
)

from .preprocessing.split_dataset import split_dataset, perform_split


# vectorizations/
from .vectorizations.tfidf_vectorization import build_tfidf_vectorizer
from .vectorizations.ngram_vectorization import build_ngram_vectorizer
from .vectorizations.embeddings_vectorization import build_embeddings_vectorizer

# models/
from .models.fmlp import NeuralNetwork


# trainings/
from .trainings.train_tfidf import train_tfidf
from .trainings.train_ngram import train_ngram
from .trainings.train_embeddings import train_embeddings

# all project functions
__all__ = [
    # Paths
    "get_raw",
    "raw_json_path",
    "raw_parquet_path",
    "dataset_path",
    "clean_dataset_path",
    "taxonomy_path",
    "labels_path",
    "train_path",
    "test_path",
    
    # Utils
    "restart_file",
    "batch_load",
    "load_training_data",
    "log_training_summary",
    
    # Preprocessing
    "convert_json_to_parquet",
    "extract_taxonomy",
    "save_taxonomy",
    "build_taxonomy",
    "prepare_batch",
    "drop_duplicates",
    "convert_parquet_to_dataset",
    "load_canonical_labels",
    "get_observed_labels",
    "summarize_labels",
    "filter_dataset_labels",
    "split_dataset",
    "perform_split",
    "extract_labels",
    "save_labels",
    "build_labels",
    
    # Vectorization
    "build_tfidf_vectorizer",
    "build_ngram_vectorizer",
    "build_embeddings_vectorizer",
    
    # Training
    "train_tfidf",
    "train_ngram",
    "train_embeddings",

    # Config
    "config",
]
