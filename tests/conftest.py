
import os
from pathlib import Path
import numpy as np
import pytest
from scipy import sparse
import pyarrow as pa
import pyarrow.parquet as pq

@pytest.fixture
def tmp_vectors(tmp_path):
    """ Create tiny TF-IDF-like sparse matrices + label arrays, and a ds_labels.txt. """
    d = tmp_path / "vectors"
    d.mkdir()

    X_train = sparse.csr_matrix(np.array([
        [1,0,2,0,0],
        [0,1,0,0,3],
        [1,1,0,0,0],
        [0,0,0,4,0],
    ], dtype=np.float32))
    y_train = np.array(["math.AC", "cs.AI", "math.AC", "cs.AI"])

    X_test = sparse.csr_matrix(np.array([[0,1,0,0,1],[1,0,0,1,0]], dtype=np.float32))
    y_test = np.array(["cs.AI", "math.AC"])

    sparse.save_npz(d / "X_train.npz", X_train)
    np.save(d / "y_train.npy", y_train)
    sparse.save_npz(d / "X_test.npz", X_test)
    np.save(d / "y_test.npy", y_test)

    labels_file = tmp_path / "ds_labels.txt"
    labels_file.write_text("cs.AI\nmath.AC\n", encoding="utf-8")

    return {
        "X_train": d / "X_train.npz",
        "y_train": d / "y_train.npy",
        "X_test": d / "X_test.npz",
        "y_test": d / "y_test.npy",
        "ds_labels": labels_file,
    }

@pytest.fixture
def tiny_dataset_parquet(tmp_path):
    """ Create a small parquet dataset with columns text,label (no singletons). """
    tbl = pa.table({
        "text": pa.array(["t0", "t1", "t2", "t3", "t4"]),
        "label": pa.array(["cs.AI", "math.AC", "cs.AI", "math.AC", "math.AC"])
    })
    path = tmp_path / "dataset.parquet"
    pq.write_table(tbl, path)
    return path

@pytest.fixture
def canonical_labels_file(tmp_path):
    p = tmp_path / "labels.txt"
    p.write_text("cs.AI\nmath.AC\nphysics.optics\n", encoding="utf-8")
    return p

@pytest.fixture
def clean_dataset_dest(tmp_path):
    return tmp_path / "clean_dataset.parquet"
