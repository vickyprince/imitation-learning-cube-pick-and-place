"""
fix_lerobot_episodes_stats.py
------------------------------
Generates meta/episodes_stats.jsonl required by convert_dataset_v21_to_v30.

Each line in the file is a JSON object:
    {"episode_index": N, "stats": {feat: {"mean": ..., "std": ..., "min": ..., "max": ..., "count": [...]}}}

Shape requirements (validated later by _validate_stat_value):
  - scalar features (state, action): mean/std/min/max shape (D,) → list of D floats
  - image features (video): mean/std/min/max shape (3, 1, 1) → [[r], [g], [b]]
  - count: shape (1,) → [N]

Usage:
    python fix_lerobot_episodes_stats.py --dataset_dir /path/to/dataset
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Scalar stats from parquet
# ---------------------------------------------------------------------------

def compute_scalar_stats(arr: np.ndarray) -> dict:
    """arr shape: (N, D) or (N,). Returns JSON-serialisable dict."""
    if arr.ndim == 1:
        arr = arr[:, None]
    n = len(arr)
    return {
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).clip(min=1e-8).tolist(),
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "count": [n],
    }


# ---------------------------------------------------------------------------
# Image stats from video
# ---------------------------------------------------------------------------

def compute_image_stats_from_video(video_path: str, n_frames: int = 30) -> dict | None:
    """
    Sample n_frames from a video and return per-channel stats shaped (3,1,1).
    Returns None if cv2 is unavailable or the video can't be opened.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            return None
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        pixels = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                pixels.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).reshape(-1, 3))
        cap.release()
        if not pixels:
            return None
        px = np.concatenate(pixels, axis=0).astype(np.float32) / 255.0  # (M, 3)
        mean = px.mean(axis=0)   # (3,)
        std  = px.std(axis=0).clip(min=1e-8)
        mn   = px.min(axis=0)
        mx   = px.max(axis=0)
        # Shape must be (3, 1, 1) → stored as nested list [[[r]], [[g]], [[b]]]
        def to_3x1x1(v): return [[[float(v[c])]] for c in range(3)]
        return {
            "mean":  to_3x1x1(mean),
            "std":   to_3x1x1(std),
            "min":   to_3x1x1(mn),
            "max":   to_3x1x1(mx),
            "count": [int(len(px))],
        }
    except Exception:
        return None


def default_image_stats(n: int = 1) -> dict:
    """Fallback: ImageNet-like stats, correct shape (3,1,1)."""
    return {
        "mean":  [[[0.485]], [[0.456]], [[0.406]]],
        "std":   [[[0.229]], [[0.224]], [[0.225]]],
        "min":   [[[0.0]],   [[0.0]],   [[0.0]]],
        "max":   [[[1.0]],   [[1.0]],   [[1.0]]],
        "count": [n],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_episodes_stats(dataset_dir: str) -> None:
    info_path   = os.path.join(dataset_dir, "meta", "info.json")
    output_path = os.path.join(dataset_dir, "meta", "episodes_stats.jsonl")

    with open(info_path) as f:
        info = json.load(f)

    features   = info.get("features", {})
    data_path_tpl = info.get("data_path",
        "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    video_path_tpl = info.get("video_path",
        "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    total_episodes = info.get("total_episodes", 0)

    print(f"Generating episodes_stats.jsonl for {total_episodes} episodes ...")

    lines = []
    for ep_idx in range(total_episodes):
        # ---- locate parquet for this episode ----
        parquet_rel = data_path_tpl.format(
            episode_chunk=ep_idx // info.get("chunks_size", 1000),
            episode_index=ep_idx
        )
        parquet_path = os.path.join(dataset_dir, parquet_rel)
        if not os.path.exists(parquet_path):
            print(f"  WARNING: {parquet_rel} not found, skipping ep {ep_idx}")
            continue

        table = pq.read_table(parquet_path)
        df    = table.to_pandas()
        n_frames = len(df)

        ep_stats: dict[str, dict] = {}

        for feat_name, feat_meta in features.items():
            dtype = feat_meta.get("dtype", "")

            if dtype in ("float32", "float64", "int32", "int64"):
                if feat_name in df.columns:
                    col_data = df[feat_name].tolist()
                    arr = np.array(col_data, dtype=np.float32)
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    ep_stats[feat_name] = compute_scalar_stats(arr)
                else:
                    # Might be a column that hasn't been merged yet — skip
                    pass

            elif dtype in ("video", "image"):
                video_rel = video_path_tpl.format(
                    episode_chunk=ep_idx // info.get("chunks_size", 1000),
                    episode_index=ep_idx,
                    video_key=feat_name
                )
                video_path = os.path.join(dataset_dir, video_rel)
                img_stats = None
                if os.path.exists(video_path):
                    img_stats = compute_image_stats_from_video(video_path, n_frames=30)
                if img_stats is None:
                    img_stats = default_image_stats(n=n_frames)
                ep_stats[feat_name] = img_stats

        lines.append({"episode_index": ep_idx, "stats": ep_stats})

        if (ep_idx + 1) % 10 == 0 or ep_idx == total_episodes - 1:
            print(f"  {ep_idx + 1}/{total_episodes} episodes done")

    with open(output_path, "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")

    print(f"\nWritten: {output_path} ({len(lines)} episodes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="Root of the v2.1 LeRobot dataset.")
    args = parser.parse_args()
    generate_episodes_stats(args.dataset_dir)
