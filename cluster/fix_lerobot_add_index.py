"""
fix_lerobot_add_index.py
------------------------
Adds a global 'index' column to every data parquet file in a LeRobot v3.0 dataset.

LeRobot v3.0 uses the layout:
    data/chunk-000/file-000.parquet   (all frames, or chunked across multiple files)

The dataset_reader.py requires each row to have an 'index' key that contains
the absolute/global frame index across the entire dataset (0-based).

Usage:
    python fix_lerobot_add_index.py --dataset_dir /path/to/dataset
"""

import argparse
import glob
import os
import re

import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd


def find_data_parquets(dataset_dir: str) -> list[str]:
    """Return data parquet paths sorted by chunk then file index."""
    # v3.0 layout: data/chunk-XXX/file-XXX.parquet
    pattern = os.path.join(dataset_dir, "data", "chunk-*", "file-*.parquet")
    paths = glob.glob(pattern)

    if not paths:
        # Fallback: v2.1 layout episode_XXXXXX.parquet (shouldn't happen for v3.0)
        pattern = os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet")
        paths = glob.glob(pattern)

    if not paths:
        raise FileNotFoundError(
            f"No data parquet files found under {dataset_dir}/data/\n"
            f"Expected: data/chunk-XXX/file-XXX.parquet"
        )

    def _sort_key(p):
        chunk_m = re.search(r"chunk-(\d+)", p)
        file_m  = re.search(r"file-(\d+)\.parquet$", p)
        ep_m    = re.search(r"episode_(\d+)\.parquet$", p)
        chunk_i = int(chunk_m.group(1)) if chunk_m else 0
        file_i  = int(file_m.group(1)) if file_m else (int(ep_m.group(1)) if ep_m else 0)
        return (chunk_i, file_i)

    return sorted(paths, key=_sort_key)


def add_index_column(dataset_dir: str) -> None:
    paths = find_data_parquets(dataset_dir)
    print(f"Found {len(paths)} data parquet file(s).")

    global_idx = 0
    for path in paths:
        table = pq.read_table(path)
        df = table.to_pandas()
        n = len(df)

        print(f"  {os.path.relpath(path, dataset_dir)}: {n} rows, columns: {list(df.columns)}")

        if "index" in df.columns:
            existing_start = int(df["index"].iloc[0])
            if existing_start == global_idx:
                print(f"    → 'index' already correct (start={global_idx}, end={global_idx+n-1}), skipping")
                global_idx += n
                continue
            else:
                print(f"    → 'index' wrong start (expected {global_idx}, got {existing_start}) — rewriting")

        df["index"] = range(global_idx, global_idx + n)

        # Put 'index' first in column order
        cols = ["index"] + [c for c in df.columns if c != "index"]
        df = df[cols]

        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
        print(f"    → wrote 'index' {global_idx}..{global_idx + n - 1}")
        global_idx += n

    print(f"Done. Total frames indexed: {global_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="Root of the LeRobot dataset (contains meta/ and data/).")
    args = parser.parse_args()
    add_index_column(args.dataset_dir)
