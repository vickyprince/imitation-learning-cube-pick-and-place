"""
dataset_pipeline/rosbag2_to_lerobot.py
=======================================
Converts rosbag2 recordings from the RecordingManager into the
LeRobot HuggingFace dataset format (Apache Parquet + Arrow IPC).

Observation space — 32D proprioceptive (mirrors AIC UR5e approach):
  [0:3]   EEF Position      ee_x, ee_y, ee_z           (m)
  [3:7]   EEF Orientation   ee_qx, ee_qy, ee_qz, ee_qw (quaternion)
  [7:10]  EEF Linear Vel    ee_vx, ee_vy, ee_vz        (m/s)
  [10:13] EEF Angular Vel   ee_wx, ee_wy, ee_wz        (rad/s)
  [13]    Gripper            gripper_opening             [0, 1]
  [14:20] Joint Positions   j1..j6                      (rad)
  [20:26] Joint Velocities  dj1..dj6                   (rad/s)
  [26:32] F/T Wrench        ft_fx..ft_tz               (N, N·m)

Action space — 4D EEF delta command:
  [dx, dy, dz, gripper]   (Cartesian EEF Δ + gripper target)

Usage:
    python3 rosbag2_to_lerobot.py \\
        --bags_dir /data/bags \\
        --output_dir /data/datasets/xarm_lift_v1 \\
        --task_name "xarm_lift"

Output structure:
    /data/datasets/xarm_lift_v1/
        meta/
            info.json          dataset metadata
            episodes.jsonl     per-episode stats
            tasks.jsonl        task descriptions
        data/
            chunk-000/
                episode_000000.parquet
                ...
        videos/
            chunk-000/
                observation.images.top/
                    episode_000000.mp4
                    ...

LeRobot format reference:
    https://github.com/huggingface/lerobot/blob/main/examples/7_get_started_with_real_robot.md
"""

import argparse
import json
import os
import pathlib
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================================
# 32D state channel layout (must match xarm_sim_node.py STATE_NAMES)
# ============================================================================
STATE_NAMES_32D = [
    # EEF Pose (7)
    "ee_x", "ee_y", "ee_z",
    "ee_qx", "ee_qy", "ee_qz", "ee_qw",
    # EEF Velocity (6)
    "ee_vx", "ee_vy", "ee_vz",
    "ee_wx", "ee_wy", "ee_wz",
    # Gripper (1)
    "gripper",
    # Joint Positions xARM6 (6)
    "j1", "j2", "j3", "j4", "j5", "j6",
    # Joint Velocities (6)
    "dj1", "dj2", "dj3", "dj4", "dj5", "dj6",
    # F/T Wrench (6)
    "ft_fx", "ft_fy", "ft_fz",
    "ft_tx", "ft_ty", "ft_tz",
]
STATE_DIM  = len(STATE_NAMES_32D)   # 32
ACTION_DIM = 4                       # [dx, dy, dz, gripper]


# ============================================================================
# ROS2 bag reading
# ============================================================================
def read_bag(bag_path: str) -> dict[str, list]:
    """
    Read a rosbag2 recording.  Uses rosbag2_py when available (inside the
    ROS2 container), falls back to raw SQLite3 otherwise.
    """
    try:
        import rosbag2_py
        return _read_bag_rosbag2py(bag_path)
    except ImportError:
        return _read_bag_sqlite(bag_path)


def _read_bag_rosbag2py(bag_path: str) -> dict[str, list]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    storage_options   = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader            = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topic_types = {
        info.name: info.type
        for info in reader.get_all_topics_and_types()
    }
    messages: dict[str, list] = {t: [] for t in topic_types}

    while reader.has_next():
        topic, data, ts = reader.read_next()
        msg_type = get_message(topic_types[topic])
        msg      = deserialize_message(data, msg_type)
        messages[topic].append((ts, msg))

    return messages


def _read_bag_sqlite(bag_path: str) -> dict[str, list]:
    """Minimal SQLite3 fallback — extracts raw bytes only."""
    import sqlite3, glob
    db_files = glob.glob(os.path.join(bag_path, "*.db3"))
    if not db_files:
        raise FileNotFoundError(f"No .db3 file found in {bag_path}")

    conn = sqlite3.connect(db_files[0])
    cur  = conn.cursor()
    cur.execute("SELECT name, type FROM topics")
    topic_map = {row[0]: row[1] for row in cur.fetchall()}
    messages: dict[str, list] = {t: [] for t in topic_map}

    cur.execute(
        "SELECT topics.name, messages.timestamp, messages.data "
        "FROM messages JOIN topics ON messages.topic_id = topics.id "
        "ORDER BY messages.timestamp"
    )
    for name, ts, data in cur.fetchall():
        messages[name].append((ts, data))

    conn.close()
    return messages


# ============================================================================
# Helpers
# ============================================================================
def decode_compressed_image(data) -> np.ndarray | None:
    """Decode sensor_msgs/CompressedImage.data → numpy (H, W, 3) uint8 BGR."""
    try:
        buf = np.frombuffer(data.data, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def nearest(msgs_list, target_ts):
    """Find message with timestamp nearest to target_ts."""
    if not msgs_list:
        return None
    diffs = [abs(ts - target_ts) for ts, _ in msgs_list]
    return msgs_list[np.argmin(diffs)][1]


def extract_state_32d(js_msg) -> list[float]:
    """
    Extract the 32D observation vector from a JointState message published
    by xarm_sim_node.  The "position" field carries all 32 channels.

    Falls back to zeros for any missing channel.
    """
    if js_msg is None:
        return [0.0] * STATE_DIM

    pos = list(js_msg.position) if hasattr(js_msg, "position") else []

    # Pad or truncate to exactly STATE_DIM
    if len(pos) < STATE_DIM:
        pos = pos + [0.0] * (STATE_DIM - len(pos))
    return pos[:STATE_DIM]


def extract_action_4d(cmd_msg) -> list[float]:
    """
    Extract the 4D action [dx, dy, dz, gripper] from a JointState command
    message (published by teleop/policy node on /sim/joint_command).
    """
    if cmd_msg is None:
        return [0.0] * ACTION_DIM
    pos = list(cmd_msg.position) if hasattr(cmd_msg, "position") else []
    if len(pos) < ACTION_DIM:
        pos = pos + [0.0] * (ACTION_DIM - len(pos))
    return pos[:ACTION_DIM]


# ============================================================================
# Per-episode conversion
# ============================================================================
def bag_to_episode(
    bag_path:    str,
    episode_idx: int,
    output_dir:  str,
    chunk:       int = 0,
) -> dict:
    """
    Convert a single rosbag2 recording to one LeRobot episode.
    Returns episode metadata dict.
    """
    msgs = read_bag(bag_path)

    js_topic    = "/sim/joint_states"                       # 32D state vector
    cmd_topic   = "/sim/joint_command"                      # 4D action
    img_topic   = "/sim/camera/image_compressed"            # top/side camera
    wrist_topic = "/sim/camera/wrist/image_compressed"      # wrist camera

    js_msgs     = msgs.get(js_topic,    [])
    cmd_msgs    = msgs.get(cmd_topic,   [])
    img_msgs    = msgs.get(img_topic,   [])
    wrist_msgs  = msgs.get(wrist_topic, [])

    has_wrist = len(wrist_msgs) > 0

    if not js_msgs:
        print(f"  WARNING: No joint_states in {bag_path}, skipping.")
        return {}

    n_frames = len(js_msgs)

    # ---- Video writers ------------------------------------------------------
    ep_str = f"episode_{episode_idx:06d}"

    # Top camera
    top_video_dir = os.path.join(
        output_dir, "videos", f"chunk-{chunk:03d}", "observation.images.top"
    )
    os.makedirs(top_video_dir, exist_ok=True)
    top_video_path = os.path.join(top_video_dir, f"{ep_str}.mp4")

    sample_img = decode_compressed_image(nearest(img_msgs, js_msgs[0][0]))
    h, w       = (480, 640) if sample_img is None else sample_img.shape[:2]
    top_writer = cv2.VideoWriter(
        top_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h)
    )

    # Wrist camera (128×128)
    wrist_writer = None
    wrist_video_path = None
    if has_wrist:
        wrist_video_dir = os.path.join(
            output_dir, "videos", f"chunk-{chunk:03d}", "observation.images.wrist"
        )
        os.makedirs(wrist_video_dir, exist_ok=True)
        wrist_video_path = os.path.join(wrist_video_dir, f"{ep_str}.mp4")
        wrist_writer = cv2.VideoWriter(
            wrist_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (128, 128)
        )

    # ---- Build rows --------------------------------------------------------
    rows = []
    for frame_idx, (ts, js_msg) in enumerate(js_msgs):
        state  = extract_state_32d(js_msg)
        action = extract_action_4d(nearest(cmd_msgs, ts))

        # Top camera frame
        img_msg = nearest(img_msgs, ts)
        frame   = decode_compressed_image(img_msg) if img_msg else None
        if frame is not None:
            top_writer.write(frame)
        else:
            top_writer.write(np.zeros((h, w, 3), np.uint8))

        # Wrist camera frame
        if has_wrist and wrist_writer is not None:
            w_msg   = nearest(wrist_msgs, ts)
            w_frame = decode_compressed_image(w_msg) if w_msg else None
            if w_frame is not None:
                if w_frame.shape[:2] != (128, 128):
                    w_frame = cv2.resize(w_frame, (128, 128))
                wrist_writer.write(w_frame)
            else:
                wrist_writer.write(np.zeros((128, 128, 3), np.uint8))

        rows.append({
            "episode_index": episode_idx,
            "frame_index":   frame_idx,
            "timestamp":     ts / 1e9,           # nanoseconds → seconds
            "observation.state": state,          # 32D list
            "action":            action,         # 4D list
            "next.done":    (frame_idx == n_frames - 1),
        })

    top_writer.release()
    if wrist_writer is not None:
        wrist_writer.release()

    video_path = top_video_path  # keep for metadata compat

    # ---- Write Parquet -------------------------------------------------------
    parquet_dir = os.path.join(output_dir, "data", f"chunk-{chunk:03d}")
    os.makedirs(parquet_dir, exist_ok=True)

    df = pd.DataFrame(rows)

    # Write observation.state and action as fixed-size float32 list columns.
    # LeRobot (v2.1 and v3.0) expects list<float32>[D] columns — NOT individual
    # per-channel columns.  Writing them directly here avoids any downstream
    # column-merge step that could accidentally zero out the values.
    for col, dim in [("observation.state", STATE_DIM), ("action", ACTION_DIM)]:
        target_type = pa.list_(pa.float32(), dim)
        arr = np.array(df[col].tolist(), dtype=np.float32)   # (N, dim)
        df[col] = pd.Series(arr.tolist())                     # list-of-lists

    # task_index required by lerobot-train
    if "task_index" not in df.columns:
        df["task_index"] = 0

    # Rewrite timestamps as (local_frame_index / fps) — deterministic,
    # no ROS clock jitter, no out-of-bounds video frame indices.
    fps = 30.0
    df["timestamp"] = (df.groupby("episode_index", sort=False).cumcount() / fps)

    table = pa.Table.from_pandas(df, preserve_index=False)

    # Cast list columns to fixed-size list<float32>[D] (LeRobot requirement)
    for col, dim in [("observation.state", STATE_DIM), ("action", ACTION_DIM)]:
        idx = table.schema.get_field_index(col)
        if idx >= 0:
            target_type = pa.list_(pa.float32(), dim)
            if table.schema.field(col).type != target_type:
                data = np.array(table[col].to_pylist(), dtype=np.float32)
                table = table.set_column(idx, col,
                    pa.array(data.tolist(), type=target_type))

    pq.write_table(table, os.path.join(parquet_dir, f"{ep_str}.parquet"))

    cam_str = "top+wrist" if has_wrist else "top-only"
    print(
        f"  ✓ Episode {episode_idx}: {n_frames} frames "
        f"→ {ep_str}.parquet (32D state + 4D action) + .mp4 [{cam_str}]"
    )

    return {
        "episode_index":   episode_idx,
        "tasks":           ["xarm_lift"],
        "length":          n_frames,
        "has_wrist":       has_wrist,
        "video_path":      video_path,
        "wrist_video_path": wrist_video_path,
        "parquet_path":    os.path.join(parquet_dir, f"{ep_str}.parquet"),
    }


# ============================================================================
# Dataset-level metadata
# ============================================================================
def write_dataset_meta(output_dir: str, episodes: list[dict], task_name: str):
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)

    total_frames = sum(e.get("length", 0) for e in episodes)
    has_wrist    = any(e.get("has_wrist", False) for e in episodes)

    info = {
        "codebase_version": "v2.1",
        "robot_type":       "xarm6",
        "total_episodes":   len(episodes),
        "total_frames":     total_frames,
        "total_videos":     len(episodes) * (2 if has_wrist else 1),
        "total_chunks":     1,
        "chunks_size":      1000,
        "fps":              30,
        "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "splits":     {"train": f"0:{total_frames}"},
        "tasks":            [task_name],
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [STATE_DIM],
                "names": STATE_NAMES_32D,
                "description": (
                    "32D proprioceptive state: EEF pos(3)+quat(4)+"
                    "lin_vel(3)+ang_vel(3)+gripper(1)+"
                    "joint_pos(6)+joint_vel(6)+FT_wrench(6)"
                ),
            },
            "action": {
                "dtype": "float32",
                "shape": [ACTION_DIM],
                "names": ["dx", "dy", "dz", "gripper"],
                "description": "4D EEF delta command + gripper target",
            },
            "observation.images.top": {
                "dtype":  "video",
                "shape":  [480, 640, 3],
                "names":  ["height", "width", "channel"],
            },
            **( {
                "observation.images.wrist": {
                    "dtype":  "video",
                    "shape":  [128, 128, 3],
                    "names":  ["height", "width", "channel"],
                }
            } if has_wrist else {} ),
        },
        "created_at":  datetime.utcnow().isoformat() + "Z",
        "description": (
            f"xARM6 {task_name} dataset collected via MYBOTSHOP Robotic Webserver "
            "teleoperation. Recorded with rosbag2, converted to LeRobot format. "
            f"Observation: {STATE_DIM}D proprioceptive (EEF pose+vel+gripper+joints+F/T). "
            "Mirrors Intrinsic AIC UR5e 30D dataset architecture."
        ),
    }

    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)

    with open(os.path.join(meta_dir, "episodes.jsonl"), "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")

    with open(os.path.join(meta_dir, "tasks.jsonl"), "w") as f:
        f.write(json.dumps({"task_index": 0, "task": task_name}) + "\n")

    print(f"\nDataset metadata written to {meta_dir}")
    print(f"  Episodes     : {len(episodes)}")
    print(f"  Frames       : {total_frames}")
    print(f"  State dim    : {STATE_DIM}D  ({len(STATE_NAMES_32D)} channels)")
    print(f"  Action dim   : {ACTION_DIM}D  [dx, dy, dz, gripper]")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Convert rosbag2 recordings to LeRobot format (32D observation)"
    )
    parser.add_argument("--bags_dir",   default="/data/bags",
                        help="Directory containing episode_XXXX_* bags")
    parser.add_argument("--output_dir", default="/data/datasets/xarm_lift_v1",
                        help="Output dataset directory")
    parser.add_argument("--task_name",  default="xarm_lift",
                        help="Task name string")
    args = parser.parse_args()

    bags = sorted(pathlib.Path(args.bags_dir).glob("episode_*"))
    if not bags:
        print(f"No bags found in {args.bags_dir}")
        return

    print(f"Converting {len(bags)} rosbag2 recordings → LeRobot dataset")
    print(f"Observation: {STATE_DIM}D | Action: {ACTION_DIM}D | Output: {args.output_dir}\n")

    episodes = []
    for idx, bag_path in enumerate(bags):
        print(f"[{idx+1}/{len(bags)}] {bag_path.name}")
        ep_meta = bag_to_episode(str(bag_path), idx, args.output_dir)
        if ep_meta:
            episodes.append(ep_meta)

    write_dataset_meta(args.output_dir, episodes, args.task_name)
    print("\nConversion complete.")


if __name__ == "__main__":
    main()
