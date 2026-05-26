"""
fix_lerobot_stats.py
--------------------
Recomputes and rewrites meta/stats.json for a LeRobot v3.0 dataset.

LeRobot v3.0 normalize_processor.py requires stats with correct shapes:
  - Non-image features (state, action): mean/std/min/max shape (D,), count shape (1,)
  - Image features: mean/std/min/max shape (3, 1, 1), count shape (1,)

If stats.json has empty arrays [] the normalizer raises:
  RuntimeError: The size of tensor a (32) must match the size of tensor b (0)

This script reads the actual data parquet, computes proper per-feature stats,
and writes them back to meta/stats.json.

Image stats are estimated from a sample of frames (configurable) to avoid
decoding every video frame.

Usage:
    python fix_lerobot_stats.py --dataset_dir /path/to/dataset
    python fix_lerobot_stats.py --dataset_dir /path/to/dataset --image_sample_episodes 5
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_parquet_column(dataset_dir: str, col: str) -> np.ndarray:
    """Load a single column from all data parquet files, returned as (N, D) array."""
    pattern = os.path.join(dataset_dir, "data", "chunk-*", "file-*.parquet")
    paths = sorted(glob.glob(pattern))
    if not paths:
        # v2.1 fallback
        pattern = os.path.join(dataset_dir, "data", "chunk-*", "episode_*.parquet")
        paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {dataset_dir}/data/")

    arrays = []
    for p in paths:
        t = pq.read_table(p, columns=[col])
        col_data = t[col].to_pylist()
        arr = np.array(col_data, dtype=np.float32)
        arrays.append(arr)
    return np.concatenate(arrays, axis=0)  # (N, D) or (N,)


def scalar_stats(arr: np.ndarray) -> dict:
    """Compute mean/std/min/max/count for a (N, D) float array."""
    if arr.ndim == 1:
        arr = arr[:, None]  # make 2D
    return {
        "mean":  arr.mean(axis=0).tolist(),
        "std":   arr.std(axis=0).clip(min=1e-8).tolist(),
        "min":   arr.min(axis=0).tolist(),
        "max":   arr.max(axis=0).tolist(),
        "count": [int(len(arr))],
    }


def sample_video_frames(video_path: str, n_frames: int = 200) -> np.ndarray:
    """
    Sample n_frames evenly from a video file.
    Returns (n_frames, H, W, 3) uint8 array, or None if decoding fails.
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total == 0:
            cap.release()
            return None
        indices = np.linspace(0, total - 1, min(n_frames, total), dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
        cap.release()
        return np.stack(frames).astype(np.float32) / 255.0 if frames else None
    except Exception as e:
        print(f"    Warning: video decode failed: {e}")
        return None


def image_stats_from_videos(dataset_dir: str, video_key: str,
                             sample_episodes: int = 8,
                             frames_per_episode: int = 50) -> dict:
    """
    Estimate per-channel image stats by sampling frames from video files.
    Returns stats with mean/std/min/max shape (3, 1, 1) and count (1,).
    """
    video_dir = os.path.join(dataset_dir, "videos", video_key)
    pattern = os.path.join(video_dir, "chunk-*", "*.mp4")
    all_videos = sorted(glob.glob(pattern))

    # Sample a subset of episodes
    step = max(1, len(all_videos) // sample_episodes)
    sampled = all_videos[::step][:sample_episodes]

    print(f"    Sampling {len(sampled)} video(s) from {video_key} ...")
    all_pixels = []  # list of (C,) per-channel means from each video
    total_frames = 0

    for vpath in sampled:
        frames = sample_video_frames(vpath, n_frames=frames_per_episode)
        if frames is None:
            continue
        # frames: (N, H, W, 3) float32 in [0,1]
        all_pixels.append(frames.reshape(-1, 3))  # (N*H*W, 3)
        total_frames += len(frames)

    if not all_pixels:
        print(f"    Warning: no video frames decoded for {video_key}, using ImageNet defaults")
        return {
            "mean":  [[0.485], [0.456], [0.406]],
            "std":   [[0.229], [0.224], [0.225]],
            "min":   [[0.0],   [0.0],   [0.0]],
            "max":   [[1.0],   [1.0],   [1.0]],
            "count": [total_frames if total_frames > 0 else 1],
        }

    pixels = np.concatenate(all_pixels, axis=0)  # (M, 3)
    mean = pixels.mean(axis=0)   # (3,)
    std  = pixels.std(axis=0).clip(min=1e-8)
    mn   = pixels.min(axis=0)
    mx   = pixels.max(axis=0)

    # Shape must be (3, 1, 1) → stored as nested list [[[r]], [[g]], [[b]]]
    def to_3_1_1(v):
        return [[[float(v[c])]] for c in range(3)]

    return {
        "mean":  to_3_1_1(mean),
        "std":   to_3_1_1(std),
        "min":   to_3_1_1(mn),
        "max":   to_3_1_1(mx),
        "count": [int(total_frames)],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def recompute_stats(dataset_dir: str, image_sample_episodes: int = 8) -> None:
    info_path = os.path.join(dataset_dir, "meta", "info.json")
    stats_path = os.path.join(dataset_dir, "meta", "stats.json")

    with open(info_path) as f:
        info = json.load(f)

    features = info.get("features", {})
    stats = {}

    for feat_name, feat_meta in features.items():
        dtype = feat_meta.get("dtype", "")
        print(f"  Computing stats for: {feat_name} (dtype={dtype})")

        if dtype == "video" or dtype == "image":
            stats[feat_name] = image_stats_from_videos(
                dataset_dir, feat_name,
                sample_episodes=image_sample_episodes,
            )
            s = stats[feat_name]
            print(f"    mean(R,G,B) = {[s['mean'][c][0] for c in range(3)]}")

        elif dtype in ("float32", "float64", "int32", "int64"):
            try:
                arr = load_parquet_column(dataset_dir, feat_name)
                stats[feat_name] = scalar_stats(arr)
                s = stats[feat_name]
                print(f"    shape={arr.shape}, mean[0]={s['mean'][0]:.4f}, std[0]={s['std'][0]:.4f}")
            except Exception as e:
                print(f"    Warning: could not compute stats for {feat_name}: {e}")

        else:
            print(f"    Skipping (unhandled dtype: {dtype})")

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nStats written to {stats_path}")
    print(f"Features covered: {list(stats.keys())}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True,
                        help="Root of the LeRobot dataset (contains meta/ and data/).")
    parser.add_argument("--image_sample_episodes", type=int, default=8,
                        help="Number of episodes to sample for image stats (default: 8).")
    args = parser.parse_args()
    recompute_stats(args.dataset_dir, image_sample_episodes=args.image_sample_episodes)
