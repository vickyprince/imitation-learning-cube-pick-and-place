"""
Inspect MYBOTSHOP rosbag2 episodes (SQLite3 backend).

CDR alignment key rule:
  All alignment is computed relative to the CDR body start (blob[4:]),
  NOT the absolute blob offset.  Use body_offset = abs_offset - 4 for
  alignment maths, then add 4 back.
"""

import sqlite3
import struct
import os
import numpy as np

BAGS = {
    "ep0": "/Users/vickyprince/Projects/mybotshop_il_demo/data/bags/episode_0000_20260519_025341/episode_0000_20260519_025341_0.db3",
    "ep1": "/Users/vickyprince/Projects/mybotshop_il_demo/data/bags/episode_0001_20260519_023119/episode_0001_20260519_023119_0.db3",
}

STATE_NAMES = [
    "ee_x","ee_y","ee_z",
    "ee_qx","ee_qy","ee_qz","ee_qw",
    "ee_vx","ee_vy","ee_vz",
    "ee_wx","ee_wy","ee_wz",
    "gripper",
    "j1","j2","j3","j4","j5","j6",
    "dj1","dj2","dj3","dj4","dj5","dj6",
    "ft_fx","ft_fy","ft_fz","ft_tx","ft_ty","ft_tz",
]
ACTION_NAMES = ["ax","ay","az","gripper"]
CDR_HEADER = 4  # bytes before CDR body


def body_align(abs_offset, size):
    """Align abs_offset so that (abs_offset - CDR_HEADER) % size == 0."""
    body = abs_offset - CDR_HEADER
    body_aligned = (body + size - 1) & ~(size - 1)
    return body_aligned + CDR_HEADER


def read_uint32(blob, offset):
    offset = body_align(offset, 4)
    return struct.unpack_from("<I", blob, offset)[0], offset + 4


def read_string(blob, offset):
    offset = body_align(offset, 4)
    (slen,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    s = blob[offset:offset + max(slen - 1, 0)].decode("utf-8", errors="replace")
    return s, offset + slen


def read_float64_seq(blob, offset):
    offset = body_align(offset, 4)
    (count,) = struct.unpack_from("<I", blob, offset)
    offset += 4
    if count == 0:
        return np.array([], dtype=np.float64), offset
    offset = body_align(offset, 8)
    arr = np.array(struct.unpack_from(f"<{count}d", blob, offset), dtype=np.float64)
    return arr, offset + count * 8


def decode_joint_state(blob):
    blob = bytes(blob)
    if len(blob) < 16:
        return None, None
    # blob[0:4] = CDR encapsulation header; body starts at 4
    # stamp: sec(int32) + nanosec(uint32) = 8 bytes at body offset 0 = abs 4
    offset = CDR_HEADER + 8   # skip header + stamp
    try:
        _frame_id, offset = read_string(blob, offset)
        name_count, offset = read_uint32(blob, offset)
        names = []
        for _ in range(name_count):
            s, offset = read_string(blob, offset)
            names.append(s)
        positions, offset = read_float64_seq(blob, offset)
    except Exception:
        return None, None
    return names, positions


def decode_wrench(blob):
    blob = bytes(blob)
    if len(blob) < 56:
        return None
    offset = CDR_HEADER + 8
    try:
        _frame_id, offset = read_string(blob, offset)
        offset = body_align(offset, 8)
        vals = struct.unpack_from("<6d", blob, offset)
        return np.array(vals, dtype=np.float64)
    except Exception:
        return None


# ──────────────────────────────────────────────────────────────────────────────
def read_topic_js(db_path, topic_name, expected_n):
    conn = sqlite3.connect(db_path)
    cur  = conn.cursor()
    cur.execute("SELECT id FROM topics WHERE name=?", (topic_name,))
    row = cur.fetchone()
    if not row:
        conn.close(); return []
    tid = row[0]
    cur.execute("SELECT data FROM messages WHERE topic_id=? ORDER BY timestamp", (tid,))
    rows = cur.fetchall(); conn.close()
    out = []
    for (blob,) in rows:
        _names, pos = decode_joint_state(blob)
        if pos is not None and len(pos) == expected_n:
            out.append(pos.astype(np.float64))
    return out


# ──────────────────────────────────────────────────────────────────────────────
def analyse_episode(ep_name, db_path):
    print(f"\n{'='*64}")
    print(f"  EPISODE: {ep_name}  —  {os.path.basename(os.path.dirname(db_path))}")
    print(f"{'='*64}")

    states  = read_topic_js(db_path, "/sim/joint_states",  32)
    actions = read_topic_js(db_path, "/sim/joint_command",  4)

    print(f"\n  State frames (32D)  : {len(states)}")
    print(f"  Action frames (4D)  : {len(actions)}")

    if not states:
        print("  [ERROR] No 32D state frames decoded.")
        return

    S = np.stack(states,  axis=0)   # (T, 32)  float64
    A = np.stack(actions, axis=0) if actions else None

    finite_mask = np.all(np.isfinite(S), axis=1)
    bad = (~finite_mask).sum()
    if bad:
        print(f"  WARNING: {bad} frames contain non-finite values (dropping)")
    S = S[finite_mask]
    if A is not None:
        A = A[:len(S)]

    T = len(S)
    print(f"  Clean frames        : {T}")
    print(f"  Duration @30Hz      : ~{T/30:.1f} s")

    # ── EEF Position ──────────────────────────────────────────────
    print(f"\n  EEF Position (m):")
    for i, ax in enumerate(["X","Y","Z"]):
        c = S[:, i]
        print(f"    {ax}: [{c.min():+.4f}, {c.max():+.4f}]  "
              f"range={c.max()-c.min():.4f} m  mean={c.mean():+.4f}")

    # ── EEF Quaternion sanity ─────────────────────────────────────
    qnorm = np.linalg.norm(S[:, 3:7], axis=1)
    print(f"\n  Quaternion norm: min={qnorm.min():.4f}  max={qnorm.max():.4f}  "
          f"({'✓ unit' if abs(qnorm.mean()-1.0)<0.01 else '⚠ not unit'})")

    # ── Gripper ───────────────────────────────────────────────────
    grip = S[:, 13]
    n_open = (grip > 0).sum()   # gym: >0 = open tendency
    n_closed = (grip <= 0).sum()
    print(f"\n  Gripper (gym: -1=open, +1=close):")
    print(f"    min={grip.min():+.3f}  max={grip.max():+.3f}  "
          f"mean={grip.mean():+.3f}  open={n_open}  closed={n_closed}")

    # ── Joint Positions ───────────────────────────────────────────
    print(f"\n  Joint positions (rad) [j1..j6]:")
    for k in range(6):
        c = S[:, 14+k]
        print(f"    j{k+1}: [{c.min():+.4f}, {c.max():+.4f}]  std={c.std():.4f}")

    # ── EEF Velocity ──────────────────────────────────────────────
    spd = np.linalg.norm(S[:, 7:10], axis=1)
    print(f"\n  EEF speed (m/s): max={spd.max():.4f}  mean={spd.mean():.4f}  "
          f"p99={np.percentile(spd,99):.4f}")

    # ── F/T Wrench ────────────────────────────────────────────────
    ft = S[:, 26:32]
    fmag = np.linalg.norm(ft[:, :3], axis=1)
    print(f"\n  F/T Wrench (N / N·m):")
    for k, lbl in enumerate(["fx","fy","fz","tx","ty","tz"]):
        c = ft[:, k]
        print(f"    {lbl}: [{c.min():+.3f}, {c.max():+.3f}]  std={c.std():.4f}")
    pct_contact = 100*(fmag > 0.1).mean()
    print(f"    |F|: max={fmag.max():.3f} N  mean={fmag.mean():.3f} N  "
          f"% >0.1 N = {pct_contact:.1f}%")

    # ── Actions ───────────────────────────────────────────────────
    if A is not None and len(A):
        print(f"\n  Actions [ax, ay, az, gripper]:")
        for k, lbl in enumerate(ACTION_NAMES):
            c = A[:, k]
            nonz = 100*(c != 0).mean()
            print(f"    {lbl}: [{c.min():+.3f}, {c.max():+.3f}]  "
                  f"std={c.std():.4f}  nonzero={nonz:.1f}%")

    # ── Temporal consistency ──────────────────────────────────────
    dpos = np.linalg.norm(np.diff(S[:, :3], axis=0), axis=1)
    big  = (dpos > 0.05).sum()
    print(f"\n  Temporal consistency:")
    print(f"    Mean EEF Δ/step : {dpos.mean()*100:.3f} cm")
    print(f"    Max  EEF Δ/step : {dpos.max()*100:.3f} cm")
    print(f"    Steps >5cm      : {big}  ({'⚠ BAD' if big > 5 else '✓ OK'})")

    # ── Image frame count (from metadata) ────────────────────────
    conn2 = sqlite3.connect(db_path)
    c2 = conn2.cursor()
    c2.execute("SELECT id FROM topics WHERE name='/sim/camera/image_compressed'")
    r = c2.fetchone()
    if r:
        c2.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?", (r[0],))
        n_img = c2.fetchone()[0]
        print(f"\n  Camera frames       : {n_img}  (~{n_img/T*100:.0f}% of state steps)")
    conn2.close()

    # ── Verdict ───────────────────────────────────────────────────
    issues = []
    if S[:, :3].std(axis=0).max() < 0.001:
        issues.append("EEF never moved")
    if grip.std() < 0.01:
        issues.append("gripper state never changed")
    if pct_contact < 1.0:
        issues.append("F/T all zero — rebuild Docker image to pick up contact fix")
    if big > 5:
        issues.append(f"{big} large EEF jumps (>5cm/step)")
    if T < 100:
        issues.append("episode too short")
    if bad > T * 0.05:
        issues.append(f">{bad} non-finite state frames")

    print(f"\n  {'✓  Episode looks GOOD' if not issues else '⚠  Issues: ' + '; '.join(issues)}")


# ──────────────────────────────────────────────────────────────────────────────
for ep_name, db_path in BAGS.items():
    if os.path.exists(db_path):
        analyse_episode(ep_name, db_path)
    else:
        print(f"\n[SKIP] {ep_name}: not found")

print("\n\nDone.")
