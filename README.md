
# MYBOTSHOP Imitation Learning Platform — Vision Branch

**End-to-end imitation learning for xARM6 pick-and-place using a full
[LeRobot](https://github.com/huggingface/lerobot) ACT policy — a CVAE-augmented transformer
with a ResNet18 vision backbone, trained on GPU and deployed via a browser-based
teleoperation UI and ROS2.**

> The `main` branch demonstrates the complete pipeline with a lightweight state-only MLP policy.
> This branch upgrades Stage 5–6 to a production-grade ACT transformer: the same browser UI,
> the same ROS2 topics, the same Docker stack — but with a transformer that sees the camera
> and predicts 50 future actions in a single forward pass.

**Policy inference — autonomous pick-and-place after training:**

<video src="https://github.com/user-attachments/assets/6ddb8433-11dd-4262-95a0-f0d665992b32" autoplay loop muted playsinline width="100%"></video>

---

## Policy Architecture

```
Top camera (480×640 RGB)          32D proprioceptive state
         ↓                                  ↓
   ResNet18 encoder             Linear projection (state embedding)
   (ImageNet pretrained)                     ↓
         ↓                                  ↓
         └──────── fused token sequence ────┘
                              ↓
              CVAE encoder  (training only)
              encodes the ground-truth action sequence
              into a latent style vector z ~ N(0,1)
                              ↓
              Transformer decoder
              cross-attends to visual + state tokens
              conditioned on z (or z=0 at inference)
                              ↓
          chunk of 50 future EEF delta actions predicted
          in a single forward pass
              ↓
          execute 10 actions → re-query transformer → repeat
```

| Parameter | Value |
|---|---|
| Vision backbone | ResNet18 (ImageNet pretrained, torchvision) |
| Camera input | Top view 480×640 RGB |
| Proprioceptive input | 32D state vector |
| Action output | 4D EEF delta (dx, dy, dz, gripper) |
| `chunk_size` | 50 actions predicted per forward pass |
| `n_action_steps` | 10 actions executed before re-querying |
| Training framework | LeRobot (HuggingFace) |
| Training steps | 100 000 |
| Final eval loss | 0.034 |
| Checkpoint | Produced by `lerobot-train --output_dir=checkpoints/<name>` |

---

## Pipeline Overview

```
Browser UI (http://localhost:9000)
  ├── Top + wrist camera feed  ← /sim/camera/image_compressed
  │                            ← /sim/camera/wrist/image_compressed
  ├── Joystick                 → /joy → teleop_node → /sim/joint_command
  ├── Record buttons           → /recording/start|stop|discard
  └── Policy buttons           → /policy/run | /policy/stop
                               ← /policy/status | /policy/confidence | /policy/inference_fps

Stage 1  Simulation       xarm_sim_node    gym_xarm XArmLift-v0 (MuJoCo)
Stage 2  Teleoperation    teleop_node      /joy → Cartesian delta → gym action
Stage 3  Recording        recording_manager → rosbag2 bags
Stage 4  Dataset          rosbag2_to_lerobot → LeRobot Parquet + MP4
Stage 5  Training         lerobot-train → ACTPolicy (ResNet18 + CVAE + transformer)
Stage 6  Deployment       policy_node      ROS2 lifecycle node, ACT inference @ 30 Hz
Stage 7  Safety           watchdog_node    confidence + joint limits + manual override
```

---

## Quick Start

```bash
git clone -b feature/vision-act-cluster-training \
    https://github.com/vickyprince/imitation-learning-cube-pick-and-place.git
cd imitation-learning-cube-pick-and-place
```

No checkpoint is needed to start. The full workflow is:

1. Run Docker → collect demonstrations in the browser → dataset is saved under `data/`
2. Train the ACT policy with `lerobot-train` (see Stage 5) → checkpoint is saved under `checkpoints/`
3. Restart Docker → click **Run Policy** → the checkpoint is volume-mounted automatically

The `docker-compose.yml` mounts `../checkpoints` into the container at
`/data/lerobot_checkpoints/`, so any checkpoint trained locally is immediately
available without a rebuild.

---

### Mac M1 / Apple Silicon

**Prerequisites:** Docker Desktop for Apple Silicon (ARM64)

```bash
docker compose -f docker/docker-compose.yml build sim_stack
docker compose -f docker/docker-compose.yml up
```

Open **http://localhost:9000** — no login required. The top-view and wrist camera feeds
appear within a few seconds. Collect demonstrations, train the ACT policy (Stage 5),
then click **▶ Run Policy** to start inference.

---

### Ubuntu

Docker handles all ROS2 and MuJoCo dependencies automatically. NVIDIA GPUs use the
EGL renderer instead of OSMesa.

**Prerequisites:** Docker + [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
# Install NVIDIA Container Toolkit (if not already installed)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

```bash
# Override the MuJoCo GL backend for NVIDIA
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  docker compose -f docker/docker-compose.yml build sim_stack

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  docker compose -f docker/docker-compose.yml up
```

Open **http://localhost:9000** — same UI as Mac.

---

## Stage-by-Stage Guide

### Stage 1 — Simulation

The `xarm_sim_node` runs `gym_xarm/XArmLift-v0` (MuJoCo, ARM64-native) and
publishes observations as ROS2 topics:

| Topic | Type | Description |
|---|---|---|
| `/sim/camera/image_compressed` | `CompressedImage` | Top-view camera @ 30 Hz (ACT policy input) |
| `/sim/camera/wrist/image_compressed` | `CompressedImage` | Wrist-view camera @ 30 Hz (display only) |
| `/sim/joint_states` | `JointState` | 32D proprioceptive state vector |
| `/sim/ee_pose` | `PoseStamped` | End-effector position |
| `/sim/wrench` | `WrenchStamped` | Force/torque at EEF |

The 32D state vector layout:

| Channels | Description |
|---|---|
| [0:3] | EEF position (x, y, z) |
| [3:7] | EEF orientation (quaternion) |
| [7:10] | EEF linear velocity |
| [10:13] | EEF angular velocity |
| [13] | Gripper opening [0, 1] |
| [14:20] | Joint positions j1–j6 (rad) |
| [20:26] | Joint velocities dj1–dj6 (rad/s) |
| [26:32] | F/T wrench (Fx, Fy, Fz, Tx, Ty, Tz) |

---

### Stage 2 — Teleoperation

The browser joystick publishes `sensor_msgs/Joy` on `/joy`:

```
left stick X/Y  →  EEF lateral (±5 mm/step)
right stick Z   →  EEF height  (±5 mm/step)
Button C        →  gripper close
Button O        →  gripper open
```

The `teleop_node` converts these to Cartesian delta targets on `/sim/joint_command`.

---

### Stage 3 — Recording Demonstrations

**Teleoperation data collection via browser joystick:**

<video src="https://github.com/user-attachments/assets/8e97411a-15f9-41af-8d88-9b3c74941abf" autoplay loop muted playsinline width="100%"></video>

1. Click **▶ Start Demo** → calls `/recording/start`
2. Use the joystick to pick the cube and place it on the target
3. Click **⏹ Stop Demo** → calls `/recording/stop`, saves bag to `/data/bags/`
4. Click **🗑 Discard** if the demo was bad (deletes the last bag)

Bags are named `episode_XXXX_YYYYMMDD_HHMMSS/` and contain all five topics at full rate.

Aim for **100+ demonstrations** covering varied cube positions. The ACT transformer
scales better with data than a state-only MLP — more demonstrations directly improve
the policy's spatial generalisation across the workspace.

---

### Stage 4 — Dataset Conversion

Convert all recorded bags to LeRobot Parquet format:

```bash
# Via browser: click ▶ Train — conversion runs automatically before training
# Or manually:
./scripts/convert_dataset.sh
```

Output dataset structure (LeRobot v2.1 format):

```
data/datasets/xarm_lift_v1/
  meta/
    info.json          # dataset metadata, feature shapes, fps
    episodes.jsonl     # per-episode stats
    tasks.jsonl        # task descriptions
  data/chunk-000/
    episode_000000.parquet
    episode_000001.parquet
  videos/chunk-000/
    observation.images.top/
      episode_000000.mp4
```

---

### Stage 5 — Training the ACT Policy

ACT training requires a GPU. The recommended workflow is to train on a GPU cluster
and volume-mount the output checkpoint into Docker.

**Train on a GPU cluster:**

```bash
lerobot-train \
    --dataset.repo_id=local/xarm_lift \
    --dataset.root=data/datasets/xarm_lift_v1 \
    --policy.type=act \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --batch_size=8 \
    --steps=100000 \
    --output_dir=checkpoints/xarm_act
```

The checkpoint directory is volume-mounted into Docker
(`../checkpoints:/data/lerobot_checkpoints`). No rebuild needed after training —
once the checkpoint lands in `checkpoints/`, update `checkpoint_path` in
`sim_bringup.launch.py`, restart the stack, and click **Run Policy**.

The reference checkpoint for this project was trained at H-BRS University on a GPU cluster,
100 000 steps, batch size 8, reaching a final eval loss of **0.034**.

---

### Stage 6 — Policy Deployment

After `lerobot-train` completes, the checkpoint directory is already under `checkpoints/`
and volume-mounted into Docker. Update `checkpoint_path` in `sim_bringup.launch.py`
to match the directory name, then restart Docker:

```bash
docker compose -f docker/docker-compose.yml restart sim_stack
```

Then in the browser:

| Button | Effect |
|---|---|
| **▶ Run Policy** | Loads ACT checkpoint (once) + starts 30 Hz inference |
| **⏹ Stop Policy** | Stops inference, joystick teleop resumes |

Clicking **Run Policy** again after **Reset Env** always works correctly — the node
deactivates the previous session, resets the internal action queue, and starts a
fresh inference thread.

While active, the UI shows:

| Metric | Description |
|---|---|
| Policy FPS | Rolling 30-frame average inference rate |
| Confidence | Mean sigmoid of predicted actions (0–1) |

**Action chunking:** The ACT transformer predicts a chunk of 50 future EEF delta
actions in a single forward pass. LeRobot's `select_action()` manages an internal
action queue — it executes `n_action_steps=10` actions from the current chunk before
calling the transformer again. This means the transformer runs once every 10 control
ticks, dramatically reducing compute overhead while preserving temporal consistency
over multi-step pick-and-place trajectories.

---

### Stage 7 — Safety & Runtime Supervision

The `watchdog_node` monitors three conditions continuously:

| Condition | Threshold | Action |
|---|---|---|
| Policy confidence | < 0.02 for sustained period | Deactivate policy, re-enable teleop |
| Joint position | Outside limits | Emergency stop |
| Inference FPS | < 5 Hz | Warning alert |

**Manual override** (red button in UI):
- Calls `/safety/manual_override` → immediately deactivates policy
- Re-enables joystick teleop
- No confirmation required — designed for emergency use

---

## Policy Node: Dual Checkpoint Support

`policy_node.py` auto-detects the checkpoint format from the path and loads the
appropriate policy — no configuration change needed:

| `checkpoint_path` points to | Policy loaded |
|---|---|
| A directory (LeRobot format) | Full ACT transformer — ResNet18 + CVAE + transformer decoder |
| A `.pt` file | Lightweight residual MLP (state-only, produced by browser Train button) |

The node logs which format was detected and prints `chunk_size`, `n_action_steps`,
and which cameras are active at startup. The same Docker image and browser
**Run Policy** button works for both.

---

## Status

| Component | Status |
|---|---|
| Data collection → LeRobot dataset | ✅ Working |
| Full ACT training — ResNet18 + CVAE + transformer | ✅ Working — final eval loss 0.034 |
| Checkpoint loading in Docker | ✅ Working |
| End-to-end inference in browser | ✅ Working |
| Robot generalisation (varied cube positions) | ⚠️ Needs more data (100–200 demos) |

The model loads and runs inference without errors. Generalisation to new cube positions
requires more demonstrations — 38 episodes validates the full pipeline end-to-end, but
the ACT transformer needs 100–200 diverse episodes to reliably cover the workspace.

---

## Known Limitations & Future Work

**Policy generalisation** — 38 demonstrations is enough to validate the pipeline but
real-world robustness requires 100–200 episodes covering diverse cube positions. The
ResNet18 vision backbone gives the policy spatial awareness from the top camera, but
more data is needed for reliable workspace coverage.

**Simulation only** — The sim-to-real gap is not addressed. Adapting to a physical
xARM6 requires calibrating the action scale, handling camera latency, and domain
randomisation during training.

**Wrist camera not used in training** — The simulation renders a wrist-mounted camera
(`/sim/camera/wrist/image_compressed`) that is displayed in the browser but not yet
included in the training dataset. Adding it as a second image input would give the
transformer a close-up view of the gripper during grasping and improve grasp precision.
The policy node already supports dual-camera checkpoints — it reads
`observation.images.wrist` from config and subscribes to the wrist topic automatically
when present.

**No data augmentation** — Adding random crop, colour jitter, and brightness shifts
to image frames during training would improve vision-policy generalisation with the
existing demonstrations.

**Checkpoint hot-reload** — After retraining, the policy node requires a Docker restart
to load new weights. A `/policy/reload` service that hot-swaps the checkpoint without
restarting would improve the iteration loop.

---

## Repository Structure

```
mybotshop_il_demo/
├── docker/
│   ├── docker-compose.yml          # sim_stack + teleop_ui services
│   └── sim_stack/
│       ├── Dockerfile              # ROS2 Humble ARM64 + MuJoCo + gym_xarm + lerobot
│       └── entrypoint.sh
├── ros2_ws/src/
│   ├── sim_bridge/                 # gym_xarm → ROS2 topics (top + wrist cameras)
│   │   ├── sim_bridge/xarm_sim_node.py
│   │   └── launch/sim_bringup.launch.py
│   ├── teleop_bridge/              # joystick, recording, training manager
│   │   └── teleop_bridge/
│   │       ├── teleop_node.py
│   │       ├── recording_manager.py
│   │       └── training_manager.py   ← browser-triggered convert + train
│   ├── dataset_pipeline/           # rosbag2 → LeRobot Parquet conversion
│   │   └── dataset_pipeline/rosbag2_to_lerobot.py
│   ├── policy_lifecycle_manager/   # LeRobot ACT inference — ROS2 lifecycle node
│   │   └── policy_lifecycle_manager/policy_node.py
│   └── safety_watchdog/            # confidence monitor + emergency stop
│       └── safety_watchdog/watchdog_node.py
├── training/
│   └── train_act.py                # state-only MLP baseline (CPU/MPS fallback)
├── teleop_ui/
│   └── index.html                  # standalone browser UI (no framework)
├── scripts/
│   ├── collect_demos.sh            # instructions for demo collection
│   ├── convert_dataset.sh          # manual bag → dataset conversion
│   └── train.sh                    # manual training script
└── data/                           # gitignored — bags, datasets, checkpoints
    ├── bags/                       # rosbag2 recordings
    ├── datasets/xarm_lift_v1/      # LeRobot Parquet dataset
    └── checkpoints/xarm_lift_v1/   # MLP baseline checkpoint (fallback)
```

---

## Hardware Compatibility

The pipeline is hardware-agnostic. To adapt to a different arm:

1. Replace `gym_xarm/XArmLift-v0` in `xarm_sim_node.py` with your simulator
2. Update `JOINT_NAMES` and action/state space dimensions
3. Adjust joint limits in `watchdog_node.py`
4. Retrain the ACT policy on the new demonstration data

Real robot backends: any ROS2-controlled arm publishing `/joint_states`.

---

## Technologies

| Component | Technology |
|---|---|
| Robot middleware | ROS2 Humble |
| Simulation | gym_xarm 0.1.1 / MuJoCo 2.x |
| WebSocket bridge | rosbridge_server |
| Dataset format | LeRobot v2.1 (Apache Parquet + MP4) |
| Vision backbone | ResNet18 (ImageNet pretrained, torchvision) |
| Policy | LeRobot `ACTPolicy` — CVAE + Transformer decoder + ResNet18 |
| Action chunking | chunk_size=50, n_action_steps=10 |
| Training framework | LeRobot (HuggingFace) |
| Training hardware | GPU cluster (H-BRS University) |
| Container | Docker ARM64 native for Apple Silicon |
| Browser UI | Vanilla HTML/JS + ROSLIB.js |

---

## Author

**Vicky Prince** · Robotics Software Engineer  
vickyprincevictor22@gmail.com
