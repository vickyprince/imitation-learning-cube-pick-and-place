"""
scripts/fix_lerobot_episodes.py
================================
Fixes meta/episodes parquet so that videos/*/from_timestamp and
videos/*/to_timestamp are computed from episode frame counts, not from
whatever values were written during dataset creation.

Root cause of "Invalid frame index=31486 for streamIndex=0; must be < 30854":
  LeRobot computes shifted_ts = from_timestamp[ep] + parquet_timestamp,
  then frame_index = round(shifted_ts * video_fps).  If from_timestamp was
  derived incorrectly the frame index can exceed the video length.

Fix:
  from_timestamp[ep] = cumulative_frames_before_ep / fps
  to_timestamp[ep]   = (cumulative_frames_before_ep + ep_length) / fps

This is consistent with the v3.0 convention.  We derive episode lengths from
the data parquet directly (sum of rows per episode_index), so no dependency
on dataset_from_index.

Run ONCE before (re)submitting training:
    python3 fix_lerobot_episodes.py --dataset_dir /work/vvicto2s/aic/data/datasets/xarm_lift_v1
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def fix_episodes(dataset_dir: pathlib.Path):
    # ------------------------------------------------------------------ #
    # fps from info.json
    # ------------------------------------------------------------------ #
    info_path = dataset_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    fps = float(info["fps"])
    print(f"Dataset fps: {fps}")

    video_keys = [
        k for k, v in info.get("features", {}).items()
        if isinstance(v, dict) and v.get("dtype") == "video"
    ]
    print(f"Video keys: {video_keys}")

    # ------------------------------------------------------------------ #
    # Compute episode lengths from the DATA parquet
    # ------------------------------------------------------------------ #
    data_files = sorted((dataset_dir / "data").glob("**/*.parquet"))
    if not data_files:
        print(f"ERROR: no data parquet files found under {dataset_dir}/data")
        return

    frames = []
    for p in data_files:
        t = pq.read_table(p, columns=["episode_index"])
        frames.append(t.to_pandas())
    df_data = pd.concat(frames, ignore_index=True)

    ep_lengths = (
        df_data.groupby("episode_index")["episode_index"]
        .count()
        .sort_index()
        .rename("length")
    )
    # Cumulative frame offset before each episode
    cum_frames = ep_lengths.cumsum().shift(1, fill_value=0)

    print(f"\nEpisode frame counts (first 5 / last 5):")
    print(ep_lengths.head(5).to_string())
    print("...")
    print(ep_lengths.tail(5).to_string())
    print(f"Total frames: {ep_lengths.sum()}")

    # ------------------------------------------------------------------ #
    # Fix episodes parquet
    # ------------------------------------------------------------------ #
    episodes_dir = dataset_dir / "meta" / "episodes"
    ep_files = sorted(episodes_dir.glob("**/*.parquet"))
    if not ep_files:
        print(f"\nERROR: no episodes parquet found in {episodes_dir}")
        return

    for ep_path in ep_files:
        print(f"\nProcessing: {ep_path}")
        table = pq.read_table(ep_path)
        df = table.to_pandas()
        print(f"  Episodes rows: {len(df)}")

        changed = False
        for vid_key in video_keys:
            from_col = f"videos/{vid_key}/from_timestamp"
            to_col   = f"videos/{vid_key}/to_timestamp"

            # Compute correct values
            correct_from = df["episode_index"].map(lambda e: cum_frames.get(e, 0) / fps)
            correct_to   = df["episode_index"].map(lambda e: (cum_frames.get(e, 0) + ep_lengths.get(e, 0)) / fps)

            old_str = ""
            if from_col in df.columns:
                old_str = f" (was {df[from_col].iloc[0]:.3f}..{df[from_col].iloc[-1]:.3f})"

            df[from_col] = correct_from.astype(float)
            df[to_col]   = correct_to.astype(float)
            changed = True
            print(f"  {vid_key}: from_timestamp[0]={correct_from.iloc[0]:.3f} "
                  f"to_timestamp[-1]={correct_to.iloc[-1]:.3f}{old_str}")

        # Also fix dataset_from_index / dataset_to_index if present
        if "dataset_from_index" in df.columns:
            correct_di_from = df["episode_index"].map(lambda e: int(cum_frames.get(e, 0)))
            correct_di_to   = df["episode_index"].map(lambda e: int(cum_frames.get(e, 0) + ep_lengths.get(e, 0)))
            df["dataset_from_index"] = correct_di_from
            df["dataset_to_index"]   = correct_di_to
            print(f"  dataset_from/to_index updated")

        if changed:
            new_table = pa.Table.from_pandas(df)
            pq.write_table(new_table, ep_path)
            print(f"  ✓ Written: {ep_path.name}")

    print("\nDone.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="Root of the LeRobot dataset (contains meta/, data/, videos/)")
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)
    if not dataset_dir.exists():
        print(f"ERROR: {dataset_dir} does not exist")
        return

    fix_episodes(dataset_dir)


if __name__ == "__main__":
    main()
