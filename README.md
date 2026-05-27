

# MYBOTSHOP Imitation Learning Platform

**ROS2-native robotic learning operations platform for xARM6 pick-and-place,
integrated end-to-end with a browser-based teleoperation UI.**

> The full pipeline — demonstration collection, dataset conversion, policy
> training, and deployment — is driven from the browser. No terminal access
> is required during normal operation.

**ACT policy inference — autonomous pick-and-place after training:**

<video src="https://github.com/user-attachments/assets/6ddb8433-11dd-4262-95a0-f0d665992b32" autoplay loop muted playsinline width="100%"></video>

---

## Pipeline Overview

```
Browser UI (http://localhost:9000)
  ├── Camera feed         ← /sim/camera/image_compressed
  ├── Joystick            → /joy → teleop_node → /sim/joint_command
  ├── Record buttons      → /recording/start|stop|discard
  ├── Train button        → /training/config + /training/start
  │                         (auto-converts bags, then trains)
  └── Policy buttons      → /policy/run | /policy/stop
                          ← /policy/status | /policy/confidence | /policy/inference_fps

Stage 1  Simulation       xarm_sim_node    gym_xarm XArmLift-v0 (MuJoCo)
Stage 2  Teleoperation    teleop_node      /joy → Cartesian delta → gym action
Stage 3  Recording        recording_manager → rosbag2 bags
Stage 4  Dataset          rosbag2_to_lerobot → LeRobot Parquet + MP4
Stage 5  Training         training_manager → train_act.py  (ACT behaviour cloning)
Stage 6  Deployment       policy_node      ROS2 lifecycle node, ACT inference @ 30 Hz
Stage 7  Safety           watchdog_node    confidence + joint limits + manual override
```

---

## Quick Start

```bash
git clone https://github.com/vickyprince/imitation-learning-cube-pick-and-place.git
cd imitation-learning-cube-pick-and-place
```

---

### Mac M1 / Apple Silicon

**Prerequisites:** Docker Desktop for Apple Silicon (ARM64)

```bash
docker compose -f docker/docker-compose.yml build sim_stack
docker compose -f docker/docker-compose.yml up
```

Open **http://localhost:9000** — no login required. The camera feed appears within a few seconds.

For policy training, run `train_act.py` directly on the Mac (outside Docker) to use the MPS GPU — ~10× faster than Docker CPU:

```bash
python3 training/train_act.py \
    --dataset_dir data/datasets/xarm_lift_v1 \
    --output_dir  data/checkpoints/xarm_lift_v1 \
    --epochs 200
```

---

### Ubuntu

Docker is the recommended approach on Ubuntu too — it handles all ROS2 and MuJoCo dependencies automatically. The only difference from Mac is that NVIDIA GPUs use the EGL renderer instead of OSMesa.

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

Then build and run with EGL (GPU-accelerated headless rendering):

```bash
# Override the MuJoCo GL backend for NVIDIA
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  docker compose -f docker/docker-compose.yml build sim_stack

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  docker compose -f docker/docker-compose.yml up
```

Open **http://localhost:9000** — same UI as Mac.

Training auto-detects CUDA, so running `train_act.py` on the host will use the GPU automatically:

```bash
python3 training/train_act.py \
    --dataset_dir data/datasets/xarm_lift_v1 \
    --output_dir  data/checkpoints/xarm_lift_v1 \
    --epochs 200
# Device: cuda  ← printed automatically if CUDA is available
```

---

## Stage-by-Stage Guide

### Stage 1 — Simulation

The `xarm_sim_node` runs `gym_xarm/XArmLift-v0` (MuJoCo, ARM64-native) and
publishes observations as ROS2 topics:

| Topic | Type | Description |
|---|---|---|
| `/sim/camera/image_compressed` | `CompressedImage` | Top-view camera @ 30 Hz |
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

<video src="https://github.com/user-attachments/assets/d1f3dc7c-c372-4689-9d5b-4a02cafbce66" autoplay loop muted playsinline width="100%"></video>



1. Click **▶ Start Demo** → calls `/recording/start`
2. Use the joystick to pick the cube and place it on the target
3. Click **⏹ Stop Demo** → calls `/recording/stop`, saves bag to `/data/bags/`
4. Click **🗑 Discard** if the demo was bad (deletes the last bag)

Bags are named `episode_XXXX_YYYYMMDD_HHMMSS/` and contain all five topics at full rate.

Aim for **100+ demonstrations** covering varied cube positions for reliable generalization.

---

### Stage 4 & 5 — Dataset Conversion + Training (Browser)

Click **▶ Train** in the UI. The training manager runs the full pipeline automatically:

1. **Converting** (amber status) — converts all bags in `/data/bags/` to LeRobot
   Parquet format at `/data/datasets/xarm_lift_v1/`
2. **Training** (teal status) — trains ACT on the dataset, streams epoch/loss
   logs and a progress bar to the browser

Tweakable parameters in the UI before clicking Train:

| Parameter | Default | Description |
|---|---|---|
| Epochs | 2 | Training epochs (use 2 for pipeline test, 100–200 for quality) |
| Batch Size | 8 | Samples per gradient step |
| Learning Rate | 1e-4 | Initial LR (cosine annealed to 0) |
| Auto-convert bags | ✓ | Runs conversion before training |

**For full-quality training and complete controll over training:

```bash
python3 training/train_act.py \
    --dataset_dir data/datasets/xarm_lift_v1 \
    --output_dir  data/checkpoints/xarm_lift_v1 \
    --epochs 200 \
    --batch_size 8 \
    --lr 1e-4
```

The best checkpoint (lowest eval loss) is saved to
`data/checkpoints/xarm_lift_v1/act_xarm_lift.pt`.

---

### Stage 4 & 5 — Manual Scripts (alternative to browser)

```bash
# Convert bags → LeRobot dataset
./scripts/convert_dataset.sh

# Train (Docker CPU)
./scripts/train.sh
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

### Stage 6 — Policy Deployment

After training, restart Docker so the policy node loads the new checkpoint:

```bash
docker compose -f docker/docker-compose.yml restart sim_stack
```

Then in the browser:

| Button | Effect |
|---|---|
| **▶ Run Policy** | Loads checkpoint (once) + starts 30 Hz ACT inference |
| **⏹ Stop Policy** | Stops inference, joystick teleop resumes |

Clicking **Run Policy** again after **Reset Env** always works correctly — the
node deactivates the previous session, resets the action buffer, and starts
a fresh inference thread.

While active, the UI shows:

| Metric | Description |
|---|---|
| Policy FPS | Rolling 30-frame average inference rate |
| Confidence | Mean sigmoid of predicted actions (0–1) |

**Action chunking:** The policy predicts 4 actions at once (`action_horizon=4`)
and executes them before re-querying — this prevents compounding errors from
noisy single-step predictions.

**Success rate:** The UI tracks picks per session — each "Run Policy" click is
one attempt; a successful cube lift increments the counter automatically.

---

### Policy Architecture

**State-only (fast baseline):**
```
32D proprioceptive state
       ↓
  Residual MLP (512 hidden, LayerNorm, GELU)
       ↓
  4D EEF delta action  [dx, dy, dz, gripper]
```

Train with:
```bash
python3 training/train_act.py \
    --dataset_dir data/datasets/xarm_lift_v1 \
    --output_dir  data/checkpoints/xarm_lift_v1 \
    --epochs 200
```

**Vision + State** (ResNet18 camera encoder fused with proprioceptive state) is available in the `feature/vision-act-cluster-training` branch.

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

## Known Limitations & Future Work

This project demonstrates a complete end-to-end IL pipeline. Known gaps and planned improvements:

**Policy generalisation** — 25 demonstrations is enough to validate the pipeline but real-world robustness requires 100+ episodes covering diverse cube positions. The vision policy (ResNet18 encoder) already helps here by giving the model spatial awareness from the camera.

**Simulation only** — The sim-to-real gap is not addressed. Adapting to a physical xARM6 requires calibrating the action scale, handling camera latency, and domain randomisation during training.

**Action prediction** — The current model predicts a single action per forward pass, repeated for the chunk horizon. True ACT predicts a sequence of T future actions in one shot using a CVAE prior — this would improve temporal consistency over longer horizons.

**No data augmentation** — Adding random crop, colour jitter, and brightness shifts to the image frames during training would improve vision-policy generalisation with the existing 25 demos.

**Checkpoint hot-reload** — After retraining, the policy node requires a Docker restart to load the new weights. A `/policy/reload` service that hot-swaps the checkpoint without restarting would improve the iteration loop.

---

## Branch: Vision + Full ACT Training (`feature/vision-act-cluster-training`)

This branch extends the main pipeline with a full [LeRobot](https://github.com/huggingface/lerobot) ACT policy — a transformer-based action chunking model with a ResNet18 vision backbone — trained on a GPU server.

### What was added

**Full LeRobot ACT training pipeline**

The dataset collected via the browser pipeline is converted to LeRobot v3.0 format and used to train a proper ACT policy using `lerobot-train`. The trained checkpoint is then loaded by the policy node in Docker for inference.

Training configuration:
- Policy: ACT with ResNet18 vision backbone
- Input: top camera (480×640) + 32D proprioceptive state
- Output: 4D EEF delta action (dx, dy, dz, gripper)
- chunk_size: 50, n_action_steps: 10
- Training: 100K steps, final loss 0.034

**Dual-format policy node**

The policy node was extended to support both checkpoint formats transparently:
- A local `.pt` file (produced by the browser Train button) → lightweight MLP
- A LeRobot pretrained directory (produced by `lerobot-train`) → full ACT transformer

The same Docker image and browser **Run Policy** button works for both — no configuration change required.

---

### Getting started

Collect demonstrations using the browser pipeline (same as main branch), then train:

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

Place the output directory under `checkpoints/`, update the `checkpoint_path` in `ros2_ws/src/sim_bridge/launch/sim_bringup.launch.py`, and rebuild Docker. The policy node detects the directory format automatically and loads the LeRobot checkpoint without any further changes.

---

### Status

| Component | Status |
|---|---|
| Data collection → LeRobot dataset conversion | ✅ Working |
| Full ACT training (100K steps, ResNet18 + transformer) | ✅ Working — final loss 0.034 |
| Checkpoint loading in Docker | ✅ Working |
| Policy inference — robot motion | ⚠️ Incomplete |

### Known issue — inference produces near-zero motion

The ACT model loads and runs without errors but outputs near-zero EEF delta actions, so the robot barely moves. The state+vision MLP from the browser Train button does move the robot correctly.

**Root cause:** The full ACT transformer is a high-capacity model that needs substantially more data than the ~38 episodes used here. With limited demonstrations it converges to predicting near-mean actions rather than generalising to new observations.

**Fix:** Collect 100–200 demonstrations and retrain. This scale of data is the standard recommendation for ACT with a vision backbone.

---

### Planned extensions (not yet implemented)

- **Wrist camera in data collection** — the simulation already renders a wrist camera (`/sim/camera/wrist/image_compressed`) but it is not yet recorded during demonstrations or used in training. Adding it as a second input would give the policy a close-up view of the gripper and object, which significantly improves grasp precision.

- **Train mode selection in browser** — the Train button currently always runs the lightweight MLP. It should offer a choice: state-only (fast, works with few demos) or full ACT with vision (higher accuracy, needs 100+ demos). The correct pipeline would launch automatically based on the selection.

- **Dual checkpoint inference** — if both a lightweight `.pt` checkpoint and a LeRobot ACT directory are present, the Run Policy button should let the user choose which to run rather than always loading whichever path is hardcoded in the launch file.

---

## Repository Structure

```
mybotshop_il_demo/
├── docker/
│   ├── docker-compose.yml          # sim_stack + teleop_ui services
│   └── sim_stack/
│       ├── Dockerfile              # ROS2 Humble ARM64 + MuJoCo + gym_xarm
│       └── entrypoint.sh
├── ros2_ws/src/
│   ├── sim_bridge/                 # gym_xarm → ROS2 topics
│   │   ├── sim_bridge/xarm_sim_node.py
│   │   └── launch/sim_bringup.launch.py
│   ├── teleop_bridge/              # joystick, recording, training manager
│   │   └── teleop_bridge/
│   │       ├── teleop_node.py
│   │       ├── recording_manager.py
│   │       └── training_manager.py   ← browser-triggered convert + train
│   ├── dataset_pipeline/           # rosbag2 → LeRobot Parquet conversion
│   │   └── dataset_pipeline/rosbag2_to_lerobot.py
│   ├── policy_lifecycle_manager/   # ACT inference ROS2 lifecycle node
│   │   └── policy_lifecycle_manager/policy_node.py
│   └── safety_watchdog/            # confidence monitor + emergency stop
│       └── safety_watchdog/watchdog_node.py
├── training/
│   └── train_act.py                # ACT behaviour cloning trainer (MPS/CPU)
├── teleop_ui/
│   └── index.html                  # standalone browser UI (no framework)
├── scripts/
│   ├── collect_demos.sh            # instructions for demo collection
│   ├── convert_dataset.sh          # manual bag → dataset conversion
│   └── train.sh                    # manual training via Docker
└── data/                           # gitignored — bags, datasets, checkpoints
    ├── bags/                       # rosbag2 recordings
    ├── datasets/xarm_lift_v1/      # LeRobot Parquet dataset
    └── checkpoints/xarm_lift_v1/   # trained ACT checkpoint
```

---

## Hardware Compatibility

The pipeline is hardware-agnostic. To adapt to a different arm:

1. Replace `gym_xarm/XArmLift-v0` in `xarm_sim_node.py` with your simulator
2. Update `JOINT_NAMES` and action/state space dimensions
3. Adjust joint limits in `watchdog_node.py`

Tested simulation backends: gym_xarm (MuJoCo), gym_aloha (MuJoCo), gym_pusht.
Real robot backends: any ROS2-controlled arm publishing `/joint_states`.

---

## Technologies

| Component | Technology |
|---|---|
| Robot middleware | ROS2 Humble |
| Simulation | gym_xarm 0.1.1 / MuJoCo 2.x |
| WebSocket bridge | rosbridge_server |
| Dataset format | LeRobot v2.1 (Apache Parquet + MP4) |
| Policy | ACT — Action Chunking Transformer |
| Training device | MPS (Apple M1) or CPU |
| Container | Docker ARM64 native for Apple Silicon |
| Browser UI | Vanilla HTML/JS + ROSLIB.js |

---

## Author

**Vicky Prince** · Robotics Software Engineer  
vickyprincevictor22@gmail.com
