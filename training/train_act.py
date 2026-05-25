"""
training/train_act.py
=====================
Trains an ACT (Action Chunking Transformer) policy on demonstrations
collected via the MYBOTSHOP webserver teleoperation interface.

Two modes
---------
State-only  (default):
    Input  : 32D proprioceptive vector
    Output : 4D EEF delta command

Vision + State  (--vision flag):
    Input  : ResNet18(camera image) → 128D features  +  32D state
    Output : 4D EEF delta command

    The ResNet18 backbone is initialised with ImageNet weights and
    fine-tuned end-to-end.  Image frames are loaded from the MP4 videos
    that rosbag2_to_lerobot.py writes alongside the Parquet files.

Observation space: 32D proprioceptive
  EEF pos(3) + quaternion(4) + lin_vel(3) + ang_vel(3) + gripper(1)
  + joint_pos(6) + joint_vel(6) + FT_wrench(6) = 32D

Action space: 4D EEF delta command  [dx, dy, dz, gripper]

Usage:
    # State-only (fast, good baseline)
    python3 training/train_act.py \\
        --dataset_dir data/datasets/xarm_lift_v1 \\
        --output_dir  data/checkpoints/xarm_lift_v1 \\
        --epochs 200

    # Vision + state (best accuracy)
    python3 training/train_act.py \\
        --dataset_dir data/datasets/xarm_lift_v1 \\
        --output_dir  data/checkpoints/xarm_lift_v1 \\
        --epochs 200 --vision
"""

import argparse
import json
import os
import pathlib
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# State channel names — must match xarm_sim_node.py and rosbag2_to_lerobot.py
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
EXPECTED_STATE_DIM  = len(STATE_NAMES_32D)   # 32
EXPECTED_ACTION_DIM = 4                       # [dx, dy, dz, gripper]

# ImageNet normalisation constants (used for ResNet18 pre-trained weights)
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG_SIZE = 84    # native gym_xarm render resolution — upsampled to 224 for ResNet


# ============================================================================
# Dataset
# ============================================================================
class XArmLiftDataset(Dataset):
    """
    Loads LeRobot Parquet episodes.

    Each sample is (observation, action):
      observation["state"] : (32,)    float32  proprioceptive vector
      observation["image"] : (3,224,224) float32  camera frame (vision mode only)
      action               : (4,)    float32  [dx, dy, dz, gripper]
    """

    def __init__(self, dataset_dir: str, split: str = "train",
                 train_ratio: float = 0.9, use_vision: bool = False):
        super().__init__()
        parquet_files = sorted(
            pathlib.Path(dataset_dir, "data").glob("**/*.parquet")
        )
        if not parquet_files:
            raise FileNotFoundError(f"No Parquet files found in {dataset_dir}/data/")

        # Episode-level train/eval split — never split mid-episode
        n_train = max(1, int(len(parquet_files) * train_ratio))
        files   = parquet_files[:n_train] if split == "train" else (
            parquet_files[n_train:] or parquet_files[-1:]
        )

        dfs      = [pd.read_parquet(f) for f in files]
        self._df = pd.concat(dfs, ignore_index=True)

        # ---- State columns (32D) ----
        state_cols = [f"observation.state.{n}" for n in STATE_NAMES_32D]
        available  = [c for c in state_cols if c in self._df.columns]
        if not available:
            available = sorted(c for c in self._df.columns
                               if c.startswith("observation.state."))
        self._states = self._df[available].values.astype(np.float32)
        actual_state_dim = self._states.shape[1]

        # ---- Action columns (4D) ----
        action_cols      = sorted(c for c in self._df.columns if c.startswith("action."))
        self._actions    = self._df[action_cols].values.astype(np.float32)

        print(
            f"  [{split}] {len(self._df)} frames, {len(files)} episodes | "
            f"state_dim={actual_state_dim} (expected {EXPECTED_STATE_DIM}), "
            f"action_dim={self._actions.shape[1]}"
        )
        if actual_state_dim != EXPECTED_STATE_DIM:
            print(f"  WARNING: state_dim mismatch — check recording config.")

        # ---- Vision: load video frames ----
        self._use_vision = use_vision
        self._frames: list | None = None
        if use_vision:
            self._frames = self._load_video_frames(dataset_dir, files)

    def _load_video_frames(self, dataset_dir: str, parquet_files) -> np.ndarray:
        """
        Pre-load, resize, and normalise all video frames once into a single
        float32 numpy array of shape (N, 3, 224, 224).

        Doing this once at dataset init means __getitem__ is a zero-copy
        numpy index — no per-epoch resize or normalise overhead.

        Memory estimate: N × 3 × 224 × 224 × 4 bytes
          Train (22 eps, ~27K frames): ~4 GB
          Eval  (3 eps,  ~4K frames):  ~0.6 GB
        This fits comfortably on a 16 GB M1 Mac alongside the MPS model.
        """
        print("  Loading + preprocessing video frames into RAM…")
        raw_frames = []
        missing    = 0

        for pf in parquet_files:
            try:
                ep_num = int(pf.stem.split("_")[1])
            except (IndexError, ValueError):
                ep_num = int(pf.stem.replace("episode_", ""))

            video_path = (
                pathlib.Path(dataset_dir)
                / "videos" / "chunk-000"
                / "observation.images.top"
                / f"episode_{ep_num:06d}.mp4"
            )

            df_ep  = pd.read_parquet(pf)
            n_rows = len(df_ep)

            if video_path.exists():
                cap       = cv2.VideoCapture(str(video_path))
                ep_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    ep_frames.append(frame)   # BGR uint8 (H, W, 3)
                cap.release()

                fi_col = df_ep["frame_index"].values if "frame_index" in df_ep.columns \
                         else np.arange(n_rows)
                for fi in fi_col:
                    idx = int(fi)
                    raw_frames.append(
                        ep_frames[idx] if idx < len(ep_frames)
                        else np.zeros((84, 84, 3), dtype=np.uint8)
                    )
            else:
                missing += 1
                print(f"  WARNING: video not found — {video_path.name}. "
                      "Using blank frames for this episode.")
                raw_frames.extend(
                    [np.zeros((84, 84, 3), dtype=np.uint8)] * n_rows
                )

        if missing:
            print(f"  {missing}/{len(parquet_files)} videos missing — "
                  "run bag conversion first.")

        # Pre-resize to 112×112 uint8 — stored as (N, 112, 112, 3) uint8.
        # uint8 uses 1 byte/value vs 4 bytes for float32 → 4× less RAM.
        # 27K frames × 112×112×3 ≈ 1 GB  (vs 16 GB float32 at 224×224).
        # Normalization happens in __getitem__ as a fast vectorized numpy op.
        TARGET = 112
        n = len(raw_frames)
        print(f"  Pre-resizing {n} frames to {TARGET}×{TARGET} uint8…")
        out = np.empty((n, TARGET, TARGET, 3), dtype=np.uint8)
        for i, f in enumerate(raw_frames):
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            out[i] = cv2.resize(f, (TARGET, TARGET))
            if (i + 1) % 5000 == 0:
                print(f"    {i+1}/{n} frames resized…")

        mb = out.nbytes // 1_000_000
        print(f"  Done — {n} frames, {mb} MB in RAM")
        return out   # (N, 112, 112, 3) uint8

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        obs    = {"state": torch.from_numpy(self._states[idx])}
        action = torch.from_numpy(self._actions[idx])

        if self._use_vision and self._frames is not None:
            # frames[idx] is (112, 112, 3) uint8 — normalize here (fast numpy op)
            f = self._frames[idx].astype(np.float32) / 255.0   # (112,112,3)
            f = (f - IMG_MEAN) / IMG_STD
            obs["image"] = torch.from_numpy(f.transpose(2, 0, 1))  # (3,112,112)

        return obs, action


# ============================================================================
# State-only policy  (fast, good baseline)
# ============================================================================
class ACTMiniPolicy(nn.Module):
    """
    4-layer MLP with residual connections.
    Maps proprioceptive state (32D) → action (4D).
    """

    def __init__(self, state_dim: int = EXPECTED_STATE_DIM,
                 action_dim: int = EXPECTED_ACTION_DIM,
                 hidden: int = 512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, state):
        z = self.encoder(state)
        z = z + self.residual(z)
        return self.head(z)


# ============================================================================
# Vision + State policy  (ResNet18 image encoder + state MLP)
# ============================================================================
class ACTVisionPolicy(nn.Module):
    """
    ResNet18 image encoder fused with proprioceptive state → action.

    Architecture
    ------------
    image (3×224×224)  →  ResNet18 backbone  →  Linear(512→img_dim)  →  img_feat
    state (32D)        ─────────────────────────────────────────────────────────┐
                                                                                ↓
                                              concat([img_feat, state]) (img_dim+32)
                                                                                ↓
                                                    Residual MLP → head → action (4D)

    The ResNet18 backbone is initialised with ImageNet-pretrained weights.
    Both the backbone and the head are fine-tuned end-to-end.
    """

    def __init__(self, state_dim: int = EXPECTED_STATE_DIM,
                 action_dim: int = EXPECTED_ACTION_DIM,
                 img_dim: int = 128,
                 hidden: int = 512):
        super().__init__()
        self.img_dim   = img_dim
        self.state_dim = state_dim

        # ---- Image encoder: ResNet18 with custom projection head ----
        import torchvision.models as tvm
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        # Replace the 1000-class classifier with a projection to img_dim
        resnet.fc = nn.Sequential(
            nn.Linear(512, img_dim),
            nn.LayerNorm(img_dim),
            nn.GELU(),
        )
        self.image_encoder = resnet

        # ---- Fusion MLP: image features + state → action ----
        fused_dim = img_dim + state_dim
        self.encoder = nn.Sequential(
            nn.Linear(fused_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.GELU(),
        )
        self.residual = nn.Sequential(
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, state, image):
        """
        state : (B, 32)         float32
        image : (B, 3, 224, 224) float32  ImageNet-normalised
        """
        img_feat = self.image_encoder(image)              # (B, img_dim)
        fused    = torch.cat([img_feat, state], dim=-1)   # (B, img_dim+32)
        z        = self.encoder(fused)
        z        = z + self.residual(z)
        return self.head(z)                               # (B, 4)


# ============================================================================
# Training loop
# ============================================================================
def train(args):
    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")
    print(f"Mode  : {'vision + state' if args.vision else 'state-only'}")

    # ---- Dataset ----
    print("\nLoading dataset...")
    train_ds = XArmLiftDataset(args.dataset_dir, split="train",
                               use_vision=args.vision)
    eval_ds  = XArmLiftDataset(args.dataset_dir, split="eval",
                               use_vision=args.vision)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size,
                          shuffle=True, num_workers=0)
    eval_dl  = DataLoader(eval_ds,  batch_size=args.batch_size,
                          shuffle=False, num_workers=0)

    state_dim  = train_ds._states.shape[1]
    action_dim = train_ds._actions.shape[1]

    # ---- Model ----
    if args.vision:
        model = ACTVisionPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            img_dim=args.img_dim,
            hidden=512,
        ).to(device)
    else:
        model = ACTMiniPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden=512,
        ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  "
          f"(state_dim={state_dim}, action_dim={action_dim}"
          + (f", img_dim={args.img_dim})" if args.vision else ")"))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.MSELoss()

    # ---- Training ----
    best_eval_loss = float("inf")
    history        = []

    mode_str = "32D state + ResNet18 image → 4D action" if args.vision \
               else "32D state → 4D action"
    print(f"\nTraining for {args.epochs} epochs ({mode_str})...\n")

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_losses = []
        for obs, action in train_dl:
            action = action.to(device)

            if args.vision:
                state  = obs["state"].to(device)
                image  = obs["image"].to(device)
                pred   = model(state, image)
            else:
                state  = obs["state"].to(device)
                pred   = model(state)

            loss = criterion(pred, action)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # -- Eval --
        model.eval()
        eval_losses = []
        with torch.no_grad():
            for obs, action in eval_dl:
                action = action.to(device)
                if args.vision:
                    pred = model(obs["state"].to(device), obs["image"].to(device))
                else:
                    pred = model(obs["state"].to(device))
                eval_losses.append(criterion(pred, action).item())

        scheduler.step()

        train_loss = float(np.mean(train_losses))
        eval_loss  = float(np.mean(eval_losses))
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "eval_loss": eval_loss})

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"train={train_loss:.5f} | eval={eval_loss:.5f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            os.makedirs(args.output_dir, exist_ok=True)
            ckpt_path = os.path.join(args.output_dir, "act_xarm_lift.pt")
            torch.save({
                "epoch":      epoch,
                "state_dict": model.state_dict(),
                "eval_loss":  eval_loss,
                "state_dim":  state_dim,
                "action_dim": action_dim,
                "use_vision": args.vision,
                "img_dim":    args.img_dim,
                "state_names": STATE_NAMES_32D,
            }, ckpt_path)

    print(f"\nTraining complete. Best eval loss: {best_eval_loss:.5f}")
    print(f"Checkpoint: {os.path.join(args.output_dir, 'act_xarm_lift.pt')}")

    hist_path = os.path.join(args.output_dir, "training_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History:    {hist_path}")


# ============================================================================
# Entry point
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Train ACT policy — state-only or vision+state"
    )
    parser.add_argument("--dataset_dir", default="/data/datasets/xarm_lift_v1")
    parser.add_argument("--output_dir",  default="/data/checkpoints/xarm_lift_v1")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--vision",      action="store_true",
                        help="Use ResNet18 camera encoder alongside state")
    parser.add_argument("--img_dim",     type=int,   default=128,
                        help="Image feature dimension from ResNet18 projection head")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
