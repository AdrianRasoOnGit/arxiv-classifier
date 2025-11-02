#!/usr/bin/python3


from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc
import polars as pl
from tqdm import tqdm

from arxiv_classifier import raw_parquet_path, dataset_path, restart_file

def prepare_batch(batch: pa.RecordBatch) -> pa.Table:
    
    tbl = pa.Table.from_batches([batch])

    # The next three lines ensure contiguous arrays in the columns. This is pretty fascinating, because this step protects us from a fundamental behavior of Apache Arrow. When calling the columns of the dataset, due to the compression used to store the data, the column values might be fragmented into different chunks. Below, in case that happens, we merge the chunks.
    title = tbl["title"].combine_chunks()
    abstract = tbl["abstract"].combine_chunks()
    categories = tbl["categories"].combine_chunks()

    # Just in case, we will fill nulls. Again, this step is really advisable since the code won't run if it encounters one! And it's free to add, I believe.
    title = pc.fill_null(title, "")
    abstract = pc.fill_null(abstract, "")
    categories = pc.fill_null(categories, "")

    # Merge title + abstract to get text
    space = pa.array([" "] * len(title))
    text = pc.binary_join_element_wise(title, space, abstract)
    text = pc.utf8_trim_whitespace(text)

    # Extract just the first category (label from now on) from categories
    label_list = pc.utf8_split_whitespace(categories)
    label = pc.list_element(label_list, 0)

    # Filter null entries (empty strings, because due to Parquet we already have removed nulls)
    mask = pc.and_(
        pc.greater(pc.utf8_length(text), pa.scalar(0)),
        pc.greater(pc.utf8_length(label), pa.scalar(0))
    )

    tbl = pa.table({"text": text, "label": label}).filter(mask)

    return pa.table({"text": text, "label": label}).filter(mask)

def drop_duplicates(tbl: pa.Table) -> pa.Table:

    # Drop duplicates. We are not interested in two observations with the same values. We will use Polars for this task
    df = pl.from_arrow(tbl)
    df = df.unique(subset = ["text", "label"], keep = "first")
    tbl = df.to_arrow()

    # Because we have used Polars, now we can get some type mismatches between batches at Arrow level, so we need to enforce string types in the schema since it is requested by Arrow
    tbl = tbl.set_column(0, "text", pc.cast(tbl["text"], pa.string()))
    tbl = tbl.set_column(1, "label", pc.cast(tbl["label"], pa.string()))

    return tbl

def convert_parquet_to_dataset(
        src: Path = raw_parquet_path,
    dst: Path = dataset_path,
    batch_size: int = 100_000,
    original_rows: int = 2_848_279,
) -> None:

    restart_file(dst)

    # Parameters for the batch
    reader = pq.ParquetFile(src)
    writer = None
    total_rows = 0
    written_rows = 0

    for batch in tqdm(
        reader.iter_batches(columns=["title", "abstract", "categories"], batch_size=batch_size),
        desc="Processing Parquet batches",
    ):
        tbl = prepare_batch(batch)
        tbl = drop_duplicates(tbl)

        if tbl.num_rows == 0:
            continue

        if writer is None:
            writer = pq.ParquetWriter(dst, tbl.schema, compression="snappy")

        writer.write_table(tbl)
        total_rows += len(batch)
        written_rows += tbl.num_rows
        drop_pct = (1 - (written_rows / total_rows)) * 100
        tqdm.write(f"Processed {total_rows:,} rows. Kept {written_rows:,}. Dropped {drop_pct:.2f}%.")

    if writer:
        writer.close()

    lost_rows = original_rows - written_rows
    lost_pct = (lost_rows / original_rows) * 100

    print("The dataset has been created!")
    print("Total written rows: ", written_rows)
    print("Original total rows: ", original_rows)
    print("Entries dropped: ", original_rows - written_rows, f"| Percentage of data lost: {lost_pct:.3f}%.")
      
    
