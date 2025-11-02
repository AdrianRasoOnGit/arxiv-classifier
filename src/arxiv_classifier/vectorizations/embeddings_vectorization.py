#!/usr/bin/python3


from pathlib import Path
import joblib
import numpy as np
from tqdm import tqdm
from scipy.sparse import vstack, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
import gc

from sentence_transformers import SentenceTransformer

from arxiv_classifier import restart_file, batch_load

from ..paths import (
    train_path,
    test_path,
    vectorizer_embeddings_path,
    X_train_embeddings_path,
    y_train_embeddings_path,
    X_test_embeddings_path,
    y_test_embeddings_path,
)

from arxiv_classifier.config.vectorization_config import EMBEDDINGS_CONFIG


def build_embeddings_vectorizer(
    train_src: Path = train_path,
    test_src: Path = test_path,
    vectorizer_dst: Path = vectorizer_embeddings_path,
    X_train_dst: Path = X_train_embeddings_path,
    y_train_dst: Path = y_train_embeddings_path,
    X_test_dst: Path = X_test_embeddings_path,
    y_test_dst: Path = y_test_embeddings_path,

    config: dict = EMBEDDINGS_CONFIG,
) -> None:

    # Access to config
    batch_size = config.get("batch_size", 10_000)
    max_features = config.get("max_features", 30_000)
    min_df = config.get("min_df", 2)
    max_df = config.get("max_df", 0.8)
    sample_limit = config.get("sample_limit", 10_000)

    # Initilize the model
    model_name = config.get("model_name", "all-MiniLM-L6-v2")
    print(f"Loading SentenceTransformer model: {model_name}")
    model = SentenceTransformer(model_name)

    # Restart the files so they can be regenerated
    for f in [vectorizer_dst, X_train_dst, y_train_dst, X_test_dst, y_test_dst]:
        restart_file(f)

    print("Started encoding of training data.")
    texts, labels = [], []

    # Loop for encoding the training data
    for tbl in batch_load._iter_batches(train_src, columns = ["text", "label"], batch_size = batch_size):
        # Get data from with the columns (I will change it to Polars!)
        df = tbl.to_pandas()
        texts.extend(df["text"].tolist())
        labels.extend(df["label"].tolist())
        # Delete temp objects to clean memory
        del df, tbl
        gc.collect()

    print(f"Loaded {len(texts):,} training samples.")

    # Limit the amount of data we will be dealing with
    if sample_limit and len(texts) > sample_limit:
        print(f"Limiting to random {sample_limit:,} samples for memory efficiency.")
        # Here, we choose a random subset of the split. This can damage stratification, but note that not using sample_limit we keep the stratification
        idx = np.random.choice(len(texts), sample_limit, replace = False)
        text = [texts[i] for i in idx]
        label = [labels[i] for i in idx]


    # Encode in batches
    X_train = []
    for i in tqdm(range(0, len(texts), batch_size), desc = "Enconding train batches"):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(
            batch,
            show_progress_bar = False,
            convert_to_numpy = True,
            batch_size = 32,
        ).astype(np.float16)
        X_train.append(embeddings)
        
    X_train = np.vstack(X_train)
    y_train = np.array(labels, dtype = str)

    print(f"Train embeddings shape: {X_train.shape}")

    # Finally we save the train vectors and the train vectorizer
    joblib.dump(model ,vectorizer_dst)
    np.save(X_train_dst, X_train)
    np.save(y_train_dst, y_train)
    print("Training embeddings saved.")

    del texts, labels, X_train, y_train
    gc.collect()

    # Test split
    print("Started encoding of test data.")
    texts, labels = [], []

    for tbl in tqdm(
        batch_load._iter_batches(test_src, columns = ["text", "label"], batch_size = batch_size),
        desc = "Loading test batches",
    ):
        df = tbl.to_pandas()
        texts.extend(df["text"].tolist())
        labels.extend(df["label"].tolist())
        del df, tbl
        gc.collect()

    print(f"Loaded {len(texts):,} test samples.")

    if sample_limit and len(texts) > sample_limit:
        idx = np.random.choice(len(texts), sample_limit, replace = False)
        texts = [texts[i] for i in idx]
        labels = [labels[i] for i in idx]

    X_test = []
    for i in tqdm(range(0, len(texts), batch_size), desc = "Encoding test batches"):
        batch = texts[i:i + batch_size]
        embeddings = model.encode(
            batch,
            show_progress_bar = False,
            convert_to_numpy = True,
            batch_size = 32,
        ).astype(np.float16)
        X_test.append(embeddings)
        
    X_test = np.vstack(X_test)
    y_test = np.array(labels, dtype = str)

    print(f"Test embeddings shape: {X_test.shape}")

    np.save(X_test_dst, X_test)
    np.save(y_test_dst, y_test)

    del texts, labels, X_test, y_test, model
    gc.collect()

    print("Embeddings vectorization completed successfully.")
