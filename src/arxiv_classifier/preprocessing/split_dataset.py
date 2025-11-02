#!/usr/bin/python3

import os
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split

from arxiv_classifier import (
    clean_dataset_path,
    train_path,
    test_path,
    restart_file,
)

from arxiv_classifier.config import SPLITS_CONFIG

def split_dataset(src: Path = clean_dataset_path):

    # Load config
    get_val = SPLITS_CONFIG.get("get_validation", False)
    train_ratio = SPLITS_CONFIG.get("train_ratio", 0.8)
    batch_size = SPLITS_CONFIG.get("batch_size", 50_000)
    seed = SPLITS_CONFIG.get("random_state", 24)

    # In order to regenerate the data, we remove any previous splits made
    restart_file(train_path)
    restart_file(test_path)

    # We load the clean dataset and check the number of rows we have
    reader = pq.ParquetFile(src)
    total_rows = reader.metadata.num_rows

    # Read label column
    labels = []
    for batch in reader.iter_batches(columns = ["label"], batch_size = batch_size):
        labels.extend(batch.column("label").to_pylist())
    labels = np.array(labels)

    # Stratify split setup. I can see that all this approach to the splitting task can be a bit too much, but since we are following a batch strategy, we need to get done all these workarounds in order to make this simple things working
    all_indices = np.arange(total_rows)

    try:
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size = 1 - train_ratio,
            stratify = labels,
            random_state = seed,
        )
    except ValueError as e:
        print(f"Stratification could not be performed ({e}). Falling back to random split.")
        train_idx, test_idx = train_test_split(
            all_indices,
            test_size = 1 - train_ratio,
            random_state = seed,
        )

    # Diagnostics of labels preservation
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    all_unique = set(labels)
    train_unique = set(train_labels)
    test_unique = set(test_labels)

    missing_from_train = all_unique - train_unique
    missing_from_test = all_unique - test_unique

    print("\nLabel coverage check: ")
    print(f" Total unique labels: {len(all_unique)}")
    print(f" Train labels: {len(train_unique)}")
    print(f" Test labels: {len(test_unique)}")
    print(f" Missing from train: {len(missing_from_train)}")
    print(f" Missing from test: {len(missing_from_test)}")

    if missing_from_test:
        print("\n The following labels are missing from the test split:")
        print(", ".join(list(sorted(missing_from_test))[:20]))
        if len(missing_from_test) > 20:
            print(f" ... and {len(missing_from_test) - 20} more")

    # Transform to set to check stratification
    train_idx_set = set(train_idx)
    test_idx_set = set(test_idx)

    # Writers
    train_writer = test_writer = None
    offset = train_rows = test_rows = 0


    for batch in reader.iter_batches(batch_size = batch_size):
        pdf = batch.to_pandas().reset_index(drop = True)
        if "__index_level_0__" in pdf.columns:
            pdf = pdf.drop(columns = "__index_level_0__")

        batch_indices = np.arange(offset, offset + len(pdf))
        train_mask = [i in train_idx_set for i in batch_indices]
        test_mask = np.logical_not(train_mask)

        train_df = pdf[train_mask].reset_index(drop = True)
        test_df = pdf[test_mask].reset_index(drop = True)

        train_tbl = pa.Table.from_pandas(train_df, preserve_index = False)
        test_tbl = pa.Table.from_pandas(test_df, preserve_index = False)

        if train_writer is None:
            train_writer = pq.ParquetWriter(train_path, train_tbl.schema, compression = "snappy")
        if test_writer is None:
            test_writer = pq.ParquetWriter(test_path, test_tbl.schema, compression = "snappy")

        if len(train_tbl) > 0:
            train_writer.write_table(train_tbl)
            train_rows += len(train_tbl)
        if len(test_tbl) > 0:
            test_writer.write_table(test_tbl)
            test_rows += len(test_tbl)

        offset += len(pdf)
        print(
            f"Processed {offset:,}/{total_rows:,} rows | "
            f"Train: {train_rows:,} | "
            f"Test: {test_rows:,}"
        )

    # Close writers
    for writer in [train_writer, test_writer]:
        if writer:
            writer.close()

            
   # Summary
    total_written = train_rows + test_rows
    print("\nSplitting completed.")
    print(f"Train rows: {train_rows:,} ({train_rows / total_rows:.2%})")
    print(f"Test rows: {test_rows:,} ({test_rows / total_rows:.2%})")


def perform_split():
    split_dataset()
