#!/usr/bin/python3

from pathlib import Path
import os
from typing import Optional, List, Union, Generator
import pyarrow.parquet as pq
import pyarrow as pa
import polars as pl
import pandas as pd
import numpy as np
from tqdm import tqdm
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report



# Reproducibility helpers

def restart_file(path):
    """
    Remove a file we plan to build if it already exists, so we can regenerate data without double dumping.
    """
    path = Path(path)
    try:
        os.remove(path)
        print(f"{path} has been removed. Regeneration of the data ready.")
    except FileNotFoundError:
        pass

# On-the-go batching, especially useful to just loading data, in a notebook for instance

class batch_load:

    @staticmethod
    def _iter_batches(
        src: Path,
        columns: Optional[List[str]] = None,
        batch_size: int = 100_000,
        progress: bool = True,
    ):

        reader = pq.ParquetFile(src)
        iterator = reader.iter_batches(columns = columns, batch_size = batch_size)
        total_rows = reader.metadata.num_rows

        if progress:
            iterator = tqdm(
                iterator,
                total = (total_rows // batch_size) + 1,
                desc = f"Loading {src.name}",
                unit = "batch",
            )

        for batch in iterator:
            yield pa.Table.from_batches([batch])

    
    @staticmethod
    def as_tbl(
        src: Path,
        columns: Optional[List[str]] = None,
        batch_size: int = 100_000,
        progress: bool = True,
    ) -> pa.Table:

        tables = [tbl for tbl in batch_load._iter_batches(src, columns, batch_size, progress)]
        return pa.concat_tables(tables)


    @staticmethod
    def as_pd(
        src: Path,
        columns: Optional[List[str]] = None,
        batch_size: int = 100_000,
        progress: bool = True,
    ) -> pd.DataFrame:

        tbl = batch_load.as_tbl(src, columns, batch_size, progress)

        return tbl.to_pandas()


    @staticmethod
    def as_pl(
        src: Path,
        columns: Optional[List[str]] = None,
        batch_size: int = 100_000,
        progress: bool = True,
    ) -> pl.DataFrame:

        tbl = batch_load.as_tbl(src, columns, batch_size, progress)

        return pl.from_arrow(tbl)

# Adam Optimizer

class AdamOptimizer:
    def __init__(self, params, lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8):
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params.keys():
            # Moving averages of gradient and squared gradient
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * grads[f"d{k}"]
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (grads[f"d{k}"] ** 2)

            # Bias correction
            m_hat = self.m[k] / (1 - self.beta1 ** self.t)
            v_hat = self.v[k] / (1 - self.beta2 ** self.t)

            # Parameter update
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
    
