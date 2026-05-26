"""
scripts/fix_lerobot_parquet.py
==============================
Prepares our rosbag2_to_lerobot Parquet files for lerobot-train.

What it does (in order):
  1. Merges individual float columns → fixed-size list columns:
       observation.state.ee_x … → observation.state  (float32[32])
       action.0 … action.3    → action              (float32[4])
  2. Adds task_index = 0 if missing.
  3. Rewrites timestamp column as (local_frame_index / fps) so video frame
     lookup is deterministic.  ROS timestamps have jitter and occasional
     large gaps that cause torchcodec to request out-of-bounds frame indices.

Run this ONCE on the cluster before training:
    python3 fix_lerobot_parquet.py --dataset_dir /work/vvicto2s/aic/data/datasets/xarm_lift_v1
"""

import argparse
import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


STATE_NAMES_32D = [
    "ee_x", "ee_y", "ee_z",
    "ee_qx", "ee_qy", "ee_qz", "ee_qw",
    "ee_vx", "ee_vy", "ee_vz",
    "ee_wx", "ee_wy", "ee_wz",
    "gripper",
    "j1", "j2", "j3", "j4", "j5", "j6",
    "dj1", "dj2", "dj3", "dj4", "dj5", "dj6",
    "ft_fx", "ft_fy", "ft_fz", "ft_tx", "ft_ty", "ft_tz",
]
ACTION_DIM = 4

FEATURE_DIMS = {
    "observation.state": 32,
    "action": 4,
}


def _to_fixed_size_list(table: pa.Table, col_name: str, size: int) -> pa.Table:
    """Cast a list column to fixed-size list<float32>[size]."""
    idx = table.schema.get_field_index(col_name)
    if idx < 0:
        return table
    col = table.column(col_name)
    target_type = pa.list_(pa.float32(), size)
    if col.type == target_type:
        return table
    data = np.array(col.to_pylist(), dtype=np.float32)  # (N, size)
    fixed_col = pa.array(data.tolist(), type=target_type)
    return table.set_column(idx, col_name, fixed_col)


def fix_parquet(path: pathlib.Path, fps: float):
    table = pq.read_table(path)
    df    = table.to_pandas()
    changed = False

    # --- 1. observation.state: merge individual columns into list column ---
    state_cols = [f"observation.state.{n}" for n in STATE_NAMES_32D]
    present    = [c for c in state_cols if c in df.columns]
    if present and "observation.state" not in df.columns:
        arr = df[present].values.astype(np.float32)
        df["observation.state"] = [row.tolist() for row in arr]
        df = df.drop(columns=present)
        changed = True

    # --- 2. action: merge individual columns into list column ---
    action_cols = [f"action.{i}" for i in range(ACTION_DIM)]
    present_a   = [c for c in action_cols if c in df.columns]
    if present_a and "action" not in df.columns:
        arr = df[present_a].values.astype(np.float32)
        df["action"] = [row.tolist() for row in arr]
        df = df.drop(columns=present_a)
        changed = True

    # --- 3. task_index: required by lerobot-train ---
    if "task_index" not in df.columns:
        df["task_index"] = 0
        changed = True

    # --- 4. Rewrite timestamps as (local_frame_index / fps) ---
    # ROS timestamps have jitter and occasional multi-second gaps that push
    # frame_index = round(shifted_ts * video_fps) beyond the video length.
    # Using frame-index-based timestamps is deterministic and gap-free.
    if "timestamp" in df.columns and "episode_index" in df.columns:
        new_ts = (df.groupby("episode_index", sort=False).cumcount() / fps).astype(np.float64)
        if not (df["timestamp"] - new_ts).abs().lt(1e-9).all():
            df["timestamp"] = new_ts
            changed = True
            print(f"    timestamp rewritten (frame-index based, fps={fps})")

    # Rebuild Arrow table
    table = pa.Table.from_pandas(df)

    # --- 5. Force list columns to fixed-size float32 (LeRobot requirement) ---
    for col_name, size in FEATURE_DIMS.items():
        if col_name in table.schema.names:
            new_table = _to_fixed_size_list(table, col_name, size)
            if new_table is not table:
                table = new_table
                changed = True

    if changed:
        pq.write_table(table, path)
        print(f"  Fixed: {path.name}")
    else:
        print(f"  OK (already correct): {path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    args = parser.parse_args()

    dataset_dir = pathlib.Path(args.dataset_dir)

    # Read fps from info.json
    info_path = dataset_dir / "meta" / "info.json"
    fps = 30.0
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)
        fps = float(info.get("fps", 30))
    print(f"fps={fps}")

    parquet_files = sorted(dataset_dir.glob("**/*.parquet"))
    # Skip meta/ parquet files (episodes, tasks, stats)
    data_parquets = [p for p in parquet_files if "/meta/" not in str(p)]
    if not data_parquets:
        print(f"No data parquet files found in {dataset_dir}")
        return

    print(f"Fixing {len(data_parquets)} data parquet files in {dataset_dir}...")
    for p in data_parquets:
        fix_parquet(p, fps)
    print("Done.")


if __name__ == "__main__":
    main()
