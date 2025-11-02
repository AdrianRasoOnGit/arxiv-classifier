#!/usr/bin/python3


from pathlib import Path
import joblib
import numpy as np
from tqdm import tqdm
from scipy.sparse import vstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import gc

from arxiv_classifier import restart_file, batch_load

from ..paths import (
    train_path,
    test_path,
    vectorizer_ngram_path,
    X_train_ngram_path,
    y_train_ngram_path,
    X_test_ngram_path,
    y_test_ngram_path,
)

from arxiv_classifier.config.vectorization_config import NGRAM_CONFIG


def build_ngram_vectorizer(
    train_src: Path = train_path,
    test_src: Path = test_path,
    vectorizer_dst: Path = vectorizer_ngram_path,
    X_train_dst: Path = X_train_ngram_path,
    y_train_dst: Path = y_train_ngram_path,
    X_test_dst: Path = X_test_ngram_path,
    y_test_dst: Path = y_test_ngram_path,
    config: dict = NGRAM_CONFIG,
) -> None:

    batch_size = config.get("batch_size", 10_000)
    max_features = config.get("max_features", 30_000)
    min_df = config.get("min_df", 2)
    max_df = config.get("max_df", 0.8)
    sample_limit = config.get("sample_limit", 10_000)
    ngram_range = config.get("ngram_range", (1, 2))

    

    for f in [vectorizer_dst, X_train_dst, y_train_dst, X_test_dst, y_test_dst]:
        restart_file(f)

    print("Started transformation of training data.")

    text_chunks, label_chunks = [], []
    for tbl in batch_load._iter_batches(train_src, columns = ["text", "label"], batch_size = batch_size):
        df = tbl.to_pandas()
        text_chunks.extend(df["text"].tolist())
        label_chunks.extend(df["label"].tolist())
        del df, tbl
        gc.collect()

    print("Total training chunks: ", len(text_chunks))

    if sample_limit and len(text_chunks) > sample_limit:
        print(f"Limiting to random {sample_limit:,} samples for memory efficiency.")
        idx = np.random.choice(len(text_chunks), sample_limit, replace=False)
        text_chunks = [text_chunks[i] for i in idx]
        label_chunks = [label_chunks[i] for i in idx]


    vectorizer = TfidfVectorizer(
        sublinear_tf = True,
        stop_words = "english",
        max_features = max_features,
        min_df = min_df,
        max_df = max_df,
        ngram_range = ngram_range
    )

    X_train = vectorizer.fit_transform(text_chunks)
    y_train = np.array(label_chunks, dtype = str)

    print("Size of the vocabulary: ", len(vectorizer.vocabulary_))
    print("N-grams TF-IDF matrix built successfully.")

    joblib.dump(vectorizer, vectorizer_dst)
    save_npz(X_train_dst, X_train)
    np.save(y_train_dst, y_train)
    
    print("Training vectors saved.")

    del text_chunks, label_chunks, X_train, y_train
    gc.collect()
                
    
    print("Started transformation of test data.")

    test_matrices, labels = [], []
    for tbl in tqdm(
        batch_load._iter_batches(test_src, columns = ["text", "label"], batch_size = batch_size),
        desc = "Vectorizing test batches",
    ):
        df = tbl.to_pandas()
        X_batch = vectorizer.transform(df["text"].tolist())
        test_matrices.append(X_batch)
        labels.extend(df["label"].tolist())

        del df, tbl, X_batch
        gc.collect()

    X_test = vstack(test_matrices)
    y_test = np.array(labels, dtype = str)
    print("Test vectorized.")

    print("Saving vectors.")
    save_npz(X_test_dst, X_test)
    np.save(y_test_dst, y_test)

    del X_test, y_test, vectorizer
    gc.collect()

    print("TF-IDF with n-grams vectorization completed successfully.")
