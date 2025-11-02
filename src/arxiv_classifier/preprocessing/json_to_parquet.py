#!/usr/bin/python3

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import json
from tqdm import tqdm
import functools

print(functools.partial(print, flush = True))

from arxiv_classifier import raw_json_path, raw_parquet_path, restart_file

def convert_json_to_parquet(
        src: Path = raw_json_path,
        dst: Path = raw_parquet_path,
        batch_size: int = 100_000,
):
    """
    Read the raw arXiv JSON and write it as a lightweight Apache Parquet file, in batches so we avoid OOMs.
    """

    # Restart the file so we don't duplicate the data
    restart_file(dst)

    # Parameters of the batch loop
    written_rows = 0
    total_rows = 0
    writer = None
    batch = []

    # Batch loop for reading the JSON and writting the Parquet
    with open(src, "r", encoding = "utf-8") as f:
        for line in tqdm(f, desc = "Reading JSON entries"):
            try:
                entry = json.loads(line)
                batch.append(entry)
            except json.JSONDecodeError:
                continue

            # Here we manage the completion of batch reading, and we deallocate the contents so we can resuse the batch in the next iteration 
            if len(batch) >= batch_size:
                tbl = pa.Table.from_pylist(batch)
                batch.clear()

                # Check the existence of the columns, just in case. We look for "title", "abstract", and "categories"
                if not all(c in tbl.column_names for c in ["title", "abstract", "categories"]):
                    continue

                # In Arrow, an actual null value can nuke out the process if our plan is to dump it into Parquet, since this format is strict with type handling. For this reason, we are going to fill, just in case, the nulls with an emptry string. We will clean those later anyway
                title = pc.fill_null(tbl["title"], "")
                abstract = pc.fill_null(tbl["abstract"], "")
                categories = pc.fill_null(tbl["categories"], "")
            

            # Finally, we write to Parquet the readings
                if writer is None:
                    writer = pq.ParquetWriter(dst, tbl.schema, compression = "snappy")

                writer.write_table(tbl)

            # With the next two simple variables, and especially with the print, we can follow in the whole loop if we are losing any entries in the conversion. Naturally, if total_rows > written_rows, we dropped something in the way
                total_rows += len(tbl)
                written_rows += tbl.num_rows
                print(f"Processed total {total_rows:,} | written {written_rows:,}")

        print("Batch reading complete!")

    # Because it is not likely that with our batch_size configuration we are dividing in exact parts the data, we need to process now the remaining entries
    if batch:
        print(f"Reading remaining entries: {len(batch)} rows.")
        tbl = pa.Table.from_pylist(batch)
        if writer is None:
            writer = pq.ParquetWriter(dst, tbl.schema, compression="snappy")
    writer.write_table(tbl)
    written_rows += tbl.num_rows

    print("Remaining entries read!", flush = True)

    if writer:
        writer.close()

        print("Dataset built successfully.", flush = True)
        print(f"Total written rows: {written_rows:,}", flush = True)

    return dst
