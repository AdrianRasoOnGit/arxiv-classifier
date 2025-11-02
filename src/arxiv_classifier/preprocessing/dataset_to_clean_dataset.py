#!/usr/bin/python3

"""
Clean the Parquet dataset

In this module, the dataset’s labels are validated and filtered so that we have a consistent set of labels that is also suitable as training data. Labels that are not included in the current arXiv taxonomy will be removed, as well as those that occur only once, thus allowing us later to stratify the splits and, moreover, ensuring that there are no labels with too few observations, which would prevent effective exposure of the model to them.

Steps:
1. Load canonical label list.
2. Scan the dataset to collect observed labels.
3. Compare and summarize canonical vs. observed sets.
4. Remove singleton labels.
5. Filter and rebuild the dataset, just with the valid albels.
6. Save the cleaned dataset, and also the final label list, as a reference.

Usage:
    (CLI) arxiv-classifier clean

Dependencies:
    PyArrow, Pathlib, TQDM

"""

import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
from pathlib import Path
from tqdm import tqdm

from arxiv_classifier import labels_path, ds_labels_path, dataset_path, clean_dataset_path, restart_file

def load_canonical_labels(path: Path) -> set[str]:
    """
    Load the canonical set of labels from `labels.txt`.

    Args:
        path (Path): Path to the canonical label set (`labels.txt`),

    Returns:
        set[str]: labels from the taxonomy.
    """

    with open(path, "r", encoding = "utf-8") as f:
        return {line.strip() for line in f if line.strip()}

    
def get_observed_labels(path: Path, batch_size: int = 100_000) -> set[str]:
    """
    Gather all labels present in the dataset.

    We use a batched approach to limit memory use.

    Args:
        path (Path): Path to the Parquet dataset.
        batch_size (int): Number of rows to consider in each reading batch.

    Returns:
        set[str]: Unique labels found in the dataset.
    """
    
    dataset_ds = ds.dataset(path, format="parquet")
    observed = set()

    for batch in tqdm(dataset_ds.to_batches(columns=["label"], batch_size=batch_size),
                      desc="Scanning observed labels"):
        col = batch.column("label")
        observed.update(col.to_pylist())

    return observed


def summarize_labels(canonical_labels: set[str], observed_labels: set[str]) -> None:
    """
    Print a summary of the canonical and observed label sets.

    Shows counts for recognized, missing, and extra labels in the dataset.

    Args:
        canonical_labels(set[str]): The reference label set.
        observed_labels (set[str]): The observed label set.
    """
    
    print("Dataset scanned. A total of ", len(observed_labels), "unique instances of labels were recorded.")

    not_recognised_labels = observed_labels - canonical_labels
    missing_labels = canonical_labels - observed_labels

    print("Label cross-check summary")
    print("-" * 10)
    print(f"Labels in labels.txt: {len(canonical_labels):,}")
    print(f"Extra labels in dataset (not in labels.txt): {len(not_recognised_labels):,}")
    print(f"Missing labels from dataset (present in labels.txt): {len(missing_labels):,}")
    print("-" * 10)


def filter_dataset_labels(
    src: Path = dataset_path,
    dst: Path = clean_dataset_path,
    canonical_labels_path: Path = labels_path,
    batch_size: int = 100_000,
) -> None:

    """
    Filter the dataset, keeping just canonical labels that have a reasonable amount of observations.

    Here we ensure label coherence and stratification for the next stage of the preprocessing.

    Args:
        src (Path): Parquet dataset, the, say, source here.
        dst (Path): Clean parquet dataset, the, say, destination here.
        canonical_labels_path (Path): Path to canonical label list.
        batch_size (int): Number of rows per batch.

    Returns:
        None        
    """
    restart_file(dst)

    canonical_labels = load_canonical_labels(canonical_labels_path)
    observed_labels = get_observed_labels(src, batch_size)
    summarize_labels(canonical_labels, observed_labels)

    # Remove label entries where we are only getting one observation. This is due because having just one element for a category makes impossible to stratify the sample when the splits are done. We want at least two observations of each label, since we will be making two splits.

    print("Removing singleton labels, in order to ensure that proper stratification is possible in the dataset.")

    label_counts = {}
    reader = pq.ParquetFile(src)
    for batch in reader.iter_batches(batch_size = batch_size, columns = ["label"]):
        
        if "label" not in batch.schema.names:
            continue
    
        col = batch.column("label").to_pylist()
        for lbl in col:
            label_counts[lbl] = label_counts.get(lbl, 0) + 1

    labels_to_keep = {l for l, c in label_counts.items() if c > 1}

    print(f"{len(label_counts)} unique lables recorded.")
    print(f"{len(labels_to_keep)} labels with more than one instance recorded.")
    print(f"{len(label_counts) - len(labels_to_keep)} singleton labels removed.")

    print("Filter started.")

    canonical_array = pa.array(list(canonical_labels))
    labels_to_keep_array = pa.array(list(labels_to_keep))
    
    reader = pq.ParquetFile(src)
    writer = None

    kept_rows = 0
    total_rows = 0

    with tqdm(total = reader.metadata.num_rows, desc = "Processing dataset", unit = "rows") as pbar:
        for batch in reader.iter_batches(batch_size = batch_size):
            tbl = pa.Table.from_batches([batch])
            
            mask_canonical = pc.is_in(tbl["label"], value_set = canonical_array)
            mask_non_singleton = pc.is_in(tbl["label"], value_set = labels_to_keep_array)
            mask = pc.and_(mask_canonical, mask_non_singleton)
            
            filtered = tbl.filter(mask)

            total_rows += len(tbl)
            kept_rows += len(filtered)

            if filtered.num_rows == 0:
                pbar.update(len(tbl))
                continue

            if writer is None:
                writer = pq.ParquetWriter(dst, filtered.schema, compression = "snappy")

            writer.write_table(filtered)
            pbar.update(len(tbl))

    if writer:
        writer.close()

    # Check directly the final amount of labels contained in the dataset
    final_labels = set()
    final_ds = ds.dataset(dst, format = "parquet")

    for batch in tqdm(final_ds.to_batches(columns = ["label"], batch_size = batch_size), desc = "Checking final labels"):
        col = batch.column("label")
        final_labels.update(col.to_pylist())
        
    dropped = total_rows - kept_rows
    pct = (dropped / total_rows) * 100

    print("Labels successfully filtered from the dataset.")
    print(f"Total rows: {total_rows:,}")
    print(f"Kept rows: {kept_rows:,}")
    print(f"Dropped rows: {dropped:,} ({pct:.2f}%)")
    print(f"Final amount of labels: {len(final_labels):,}")

    print(f"Saving the actual dataset label set to {ds_labels_path}")
    with open(ds_labels_path, "w", encoding = "utf-8") as f:
        for label in sorted(final_labels):
            f.write(f"{label}\n")

    print(f"Dataset labels have been saved with {len(final_labels)} labels.")
