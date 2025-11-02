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


class batch_load:
    """
    Allows to, instead of fully loading data files, introduce them through batches, so we avoid any OOMs. It has implemented a few different formats so it can have more than just a pair of use cases through the project.
    """
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
