"""
training/train_act.py
=====================
Trains an ACT (Action Chunking Transformer) policy on demonstrations
collected via the MYBOTSHOP webserver teleoperation interface.

Observation space: 32D proprioceptive
  EEF pos(3) + quaternion(4) + lin_vel(3) + ang_vel(3) + gripper(1)
  + joint_pos(6) + joint_vel(6) + FT_wrench(6) = 32D

Action space: 4D EEF delta command
  [dx, dy, dz, gripper]

State channels (STATE_NAMES_32D):
  ee_x, ee_y, ee_z,
  ee_qx, ee_qy, ee_qz, ee_qw,
  ee_vx, ee_vy, ee_vz,
  ee_wx, ee_wy, ee_wz,
  gripper,
  j1, j2, j3, j4, j5, j6,
  dj1, dj2, dj3, dj4, dj5, dj6,
  ft_fx, ft_fy, ft_fz, ft_tx, ft_ty, ft_tz

Usage:
    python3 training/train_act.py \\
        --dataset_dir /data/datasets/xarm_lift_v1 \\
        --output_dir  /data/checkpoints/xarm_lift_v1 \\
        --epochs      100 \\
        --batch_size  8

The script:
    1. Loads the LeRobot-format Parquet dataset (32D state columns)
    2. Builds train/eval splits at the episode level (not frame level)
    3. Trains ACT with behaviour cloning MSE loss
    4. Saves best checkpoint to output_dir/act_xarm_lift.pt
    5. Logs training metrics to stdout (and optionally Weights & Biases)
"""

import argparse
import json
import os
import pathlib
import time

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


# ============================================================================
# Dataset
# ============================================================================
class XArmLiftDataset(Dataset):
    """
    Loads LeRobot Parquet episodes.  Each sample is an (observation, action) pair.

    observation:
        state : (32,) float32  — full proprioceptive vector
        image : not loaded here (state-only training; swap in image encoder for ACT+vision)
    action:
        (4,) float32   [dx, dy, dz, gripper]
    """

    def __init__(self, dataset_dir: str, split: str = "train", train_ratio: float = 0.9):
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

        # ---- State columns (32D — prefixed "observation.state.<name>") ----
        state_cols = [f"observation.state.{n}" for n in STATE_NAMES_32D]
        available  = [c for c in state_cols if c in self._df.columns]

        if not available:
            # Fallback: any column matching "observation.state.*"
            available = sorted(c for c in self._df.columns
                               if c.startswith("observation.state."))

        self._states = self._df[available].values.astype(np.float32)
        actual_state_dim = self._states.shape[1]

        # ---- Action columns (4D) ------------------------------------------
        action_cols = sorted(c for c in self._df.columns if c.startswith("action."))
        self._actions = self._df[action_cols].values.astype(np.float32)

        print(
            f"  [{split}] {len(self._df)} frames, {len(files)} episodes | "
            f"state_dim={actual_state_dim} (expected {EXPECTED_STATE_DIM}), "
            f"action_dim={self._actions.shape[1]}"
        )

        if actual_state_dim != EXPECTED_STATE_DIM:
            print(
                f"  WARNING: state_dim={actual_state_dim} ≠ {EXPECTED_STATE_DIM}. "
                "Check that the dataset was recorded with the 32D observation node."
            )

    def __len__(self):
        return len(self._df)

    def __getitem__(self, idx):
        obs    = {"state": torch.from_numpy(self._states[idx])}
        action = torch.from_numpy(self._actions[idx])
        return obs, action


# ============================================================================
# ACT-like policy network (state-only MLP with residual connections)
# Replace with full lerobot.policies.act.modeling_act.ACTPolicy for
# vision-based training (add ResNet18 / ViT image encoder).
# ============================================================================
class ACTMiniPolicy(nn.Module):
    """
    4-layer MLP with residual connections, LayerNorm, and GELU.
    Maps proprioceptive state (32D) → action (4D).

    For full ACT with vision:
        - Replace with lerobot.policies.act.modeling_act.ACTPolicy
        - Add image encoder (ResNet18 or ViT)
        - Use action chunking (predict T future actions, execute first K)

    Architecture insight (from AIC RunACT.py):
        - Real Intrinsic robot used ACTPolicy with 26D state + 3 cameras
        - Our 32D state gives richer signal (velocities + F/T) for contact-rich tasks
        - F/T channels are especially useful for insertion / peg-in-hole tasks
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
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden),   nn.LayerNorm(hidden), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, action_dim),
        )

    def forward(self, state):
        z = self.encoder(state)
        z = z + self.residual(z)   # residual connection
        return self.head(z)


# ============================================================================
# Training loop
# ============================================================================
def train(args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"\nDevice: {device}")

    # ---- Dataset ----
    print("\nLoading dataset...")
    train_ds = XArmLiftDataset(args.dataset_dir, split="train")
    eval_ds  = XArmLiftDataset(args.dataset_dir, split="eval")

    # num_workers=0: avoids "Too many open files" on macOS (multiprocessing
    # spawn opens file descriptors per worker; hits ulimit with Parquet files).
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    eval_dl  = DataLoader(eval_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    state_dim  = train_ds._states.shape[1]
    action_dim = train_ds._actions.shape[1]

    # ---- Model ----
    model    = ACTMiniPolicy(state_dim=state_dim, action_dim=action_dim, hidden=512).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}  (state_dim={state_dim}, action_dim={action_dim})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    # ---- Training ----
    best_eval_loss = float("inf")
    history        = []

    print(f"\nTraining for {args.epochs} epochs (32D state → 4D action)...\n")

    for epoch in range(1, args.epochs + 1):
        # -- Train --
        model.train()
        train_losses = []
        for obs, action in train_dl:
            state  = obs["state"].to(device)
            action = action.to(device)

            pred = model(state)
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
                state  = obs["state"].to(device)
                action = action.to(device)
                pred   = model(state)
                eval_losses.append(criterion(pred, action).item())

        scheduler.step()

        train_loss = float(np.mean(train_losses))
        eval_loss  = float(np.mean(eval_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "eval_loss": eval_loss})

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="/data/datasets/xarm_lift_v1")
    parser.add_argument("--output_dir",  default="/data/checkpoints/xarm_lift_v1")
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=1e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
