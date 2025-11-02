#!/usr/bin/python3

from pathlib import Path
import numpy as np
from scipy.sparse import load_npz
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse
import inspect

from ..config.models_config import FMLP_CONFIG
from ..paths import (
    X_train_tfidf_path,
    y_train_tfidf_path,
    X_test_tfidf_path,
    y_test_tfidf_path,
    X_train_ngram_path,
    y_train_ngram_path,
    X_test_ngram_path,
    y_test_ngram_path,
    ds_labels_path,
)


def load_training_data(config = FMLP_CONFIG, vector_type = None, verbose = True):
    
    # Detect the type of vectorization used in the pipeline with inspect
    if vector_type is None:
        caller = inspect.stack()[1].function.lower()
        if "tfidf" in caller:
            vector_type = "tfidf"
        elif "ngram" in caller:
            vector_type = "ngram"
        elif "embeddings" in caller:
            vector_type = "embeddings"

    # Report the type of vector detected
    if verbose:
        print(f"Loading {vector_type.upper()} features and labels.")

    # Prepare the loading of the particular set of vectors
    if vector_type == "tfidf":
        X_train, y_train = X_train_tfidf_path, y_train_tfidf_path
        X_test, y_test = X_test_tfidf_path, y_test_tfidf_path
    elif vector_type == "ngram":
        X_train, y_train = X_train_ngram_path, y_train_ngram_path
        X_test, y_test = X_test_ngram_path, y_test_ngram_path
    elif vector_type == "embeddings":
        X_train, y_train = X_train_embeddings_path, y_train_embeddings_path
        X_test, y_test = X_test_embeddings_path, y_test_embeddings_path

    else:
        raise ValueError(f"Unknown vector type '{vector_type}'. Currently, the accepted structures are 'tfidf', 'ngram', and 'embeddings'.")

    # Load the vectors as arrays
    X_train = load_npz(X_train)
    y_train = np.load(y_train)
    X_test = load_npz(X_test)
    y_test = np.load(y_test)

    # Ensure label arrays are strings before encoding (just a sanity check)
    y_train = y_train.astype(str)
    y_test = y_test.astype(str)

    # Apply limits to avoid OOM
    max_train = config.get("max_train", len(y_train))
    max_test = config.get("max_test", len(y_test))

    if X_train.shape[0] > max_train:
        X_train, y_train = X_train[:max_train], y_train[:max_train]
    if X_test.shape[0] > max_test:
        X_test, y_test = X_test[:max_test], y_test[:max_test]

    if verbose:
        print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")

    # Prepare labels for encoding
    with open(ds_labels_path, "r", encoding = "utf-8") as f:
        ds_labels = [line.strip() for line in f if line.strip()]

    encoder = LabelEncoder()
    encoder.fit(ds_labels)

    missing_train = set(y_train) - set(encoder.classes_)
    missing_test = set(y_test) - set(encoder.classes_)

    if missing_train or missing_test:
        print(
            f" Warning: {len(missing_train)} unseen labels in training and"
            f"{len(missing_test)} unseen in test, that is, these are nowhere"
            "to be found in labels.txt, the set of labels from the arxiv"
            "taxonomy."
        )

    # Encode labels
    y_train_enc = encoder.transform(y_train)
    y_test_enc = encoder.transform(y_test)

    if verbose:
        print(f"Unique labels in train: {len(np.unique(y_train_enc)):,} | "
              f"Unique labels in test: {len(np.unique(y_test_enc)):,}"
              f"Total dataset labels: {len(encoder.classes_):,}"
        )
              

    return X_train, y_train_enc, X_test, y_test_enc, encoder
