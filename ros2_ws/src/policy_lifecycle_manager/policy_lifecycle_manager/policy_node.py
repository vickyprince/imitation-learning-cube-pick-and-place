"""
policy_lifecycle_manager/policy_node.py
========================================
ROS2 Lifecycle Node — loads a LeRobot ACT checkpoint and runs inference.

Lifecycle states
----------------
  unconfigured  No model loaded. No commands published.
  configured    Model loaded in memory. No commands published.
  active        Inference running. Publishes /sim/joint_command at policy FPS.
  deactivated   Inference paused. Falls back to teleop.
  shutdown      Clean teardown.

Browser UI buttons call Trigger services (avoids lifecycle_msgs/ChangeState
which rosbridge cannot resolve):
  "Run Policy"  → /policy/run   (configure if needed, then activate)
  "Stop Policy" → /policy/stop  (deactivate)

Published topics
----------------
/sim/joint_command        sensor_msgs/JointState      inference output
/policy/status            std_msgs/String             state label
/policy/inference_fps     std_msgs/Float32            rolling avg FPS
/policy/confidence        std_msgs/Float32            mean |action| score

Subscribed topics
-----------------
/sim/camera/image_compressed        sensor_msgs/CompressedImage  top camera
/sim/camera/wrist/image_compressed  sensor_msgs/CompressedImage  wrist camera
/sim/joint_states                   sensor_msgs/JointState       32D proprioception
"""

import time
import threading
import collections

import cv2
import numpy as np
import rclpy
from rclpy.lifecycle import Node as LifecycleNode
from rclpy.lifecycle import State, TransitionCallbackReturn

from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# ImageNet normalisation constants (match train_act.py)
_IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMG_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Local checkpoint model classes — mirror of train_act.py
# (used when checkpoint_path points to a .pt file, not a LeRobot directory)
# ---------------------------------------------------------------------------

class _ACTMiniPolicy(nn.Module):
    """4-layer MLP: proprioceptive state (32D) → action (4D)."""

    def __init__(self, state_dim: int = 32, action_dim: int = 4, hidden: int = 512):
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


class _ACTVisionPolicy(nn.Module):
    """ResNet18 image encoder + proprioceptive state → action (4D)."""

    def __init__(self, state_dim: int = 32, action_dim: int = 4,
                 img_dim: int = 128, hidden: int = 512):
        super().__init__()
        self.img_dim = img_dim
        import torchvision.models as tvm
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        resnet.fc = nn.Sequential(
            nn.Linear(512, img_dim), nn.LayerNorm(img_dim), nn.GELU(),
        )
        self.image_encoder = resnet
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
        img_feat = self.image_encoder(image)
        fused    = torch.cat([img_feat, state], dim=-1)
        z        = self.encoder(fused)
        z        = z + self.residual(z)
        return self.head(z)


class _LocalACTWrapper:
    """
    Wraps a locally-trained .pt checkpoint (train_act.py output) with the
    same select_action() / reset() interface used by LeRobot ACTPolicy,
    so the rest of policy_node.py needs no changes.

    obs dict keys consumed:
      "observation.state"      : (1, 32)  float32 tensor
      "observation.images.top" : (1, 3, H, W) float32 tensor  (vision mode only)
    """

    def __init__(self, model, device, use_vision: bool):
        self._model      = model
        self._device     = device
        self._use_vision = use_vision

    def reset(self):
        pass  # no action queue to clear

    def select_action(self, obs: dict):
        state = obs["observation.state"].to(self._device)   # (1, 32)
        with torch.no_grad():
            if self._use_vision:
                img = obs["observation.images.top"].to(self._device)  # (1,3,H,W)
                # Resize to 112×112 and ImageNet-normalise to match training
                B, C, H, W = img.shape
                if H != 112 or W != 112:
                    # Use bilinear interpolate via torch (no OpenCV dep here)
                    img = torch.nn.functional.interpolate(
                        img, size=(112, 112), mode="bilinear", align_corners=False
                    )
                mean = torch.tensor(_IMG_MEAN, device=self._device).view(1, 3, 1, 1)
                std  = torch.tensor(_IMG_STD,  device=self._device).view(1, 3, 1, 1)
                img  = (img - mean) / std
                action = self._model(state, img)   # (1, 4)
            else:
                action = self._model(state)        # (1, 4)
        return action.squeeze(0)  # (4,)


class PolicyLifecycleManager(LifecycleNode):
    """
    LeRobot ACTPolicy inference node following the ROS2 managed lifecycle.

    Checkpoint: /data/lerobot_checkpoints/xarm_act_<job_id>/
      Produced by lerobot-train on the H-BRS cluster.
      Supports single-camera (observation.images.top) and dual-camera
      (observation.images.top + observation.images.wrist) checkpoints.
      The presence of "observation.images.wrist" in config.json is
      detected automatically at load time.

    Action chunking is handled internally by LeRobot's select_action()
    via _action_queue (n_action_steps from config). Call once per step.
    """

    # Top camera trained resolution (policy config: shape [3, 480, 640])
    TOP_H, TOP_W       = 480, 640
    # Wrist camera trained resolution (policy config: shape [3, 128, 128])
    WRIST_H, WRIST_W   = 128, 128

    def __init__(self):
        super().__init__("policy_node")

        self.declare_parameter(
            "checkpoint_path", "/data/checkpoints/xarm_lift_v2/act_xarm_lift.pt"
        )
        self.declare_parameter("inference_fps",          30.0)
        self.declare_parameter("confidence_threshold",   0.02)

        self._policy     = None
        self._use_wrist  = False   # set True when checkpoint has wrist camera
        self._running    = False
        self._lock       = threading.Lock()

        # Latest observations
        self._latest_image       = None   # top camera (BGR, any resolution)
        self._latest_wrist_image = None   # wrist camera (BGR, 128×128 from sim)
        self._latest_joints      = None   # 32D state vector

        # Metrics
        self._fps_window = collections.deque(maxlen=30)
        self._confidence = 0.0

        # Trigger services for browser UI
        self.create_service(Trigger, "/policy/run",  self._srv_run)
        self.create_service(Trigger, "/policy/stop", self._srv_stop)

        # Pre-load 2 s after startup so "Run Policy" click is instant
        self._pre_configure_done = False
        self.create_timer(2.0, self._pre_configure_once)

        self.get_logger().info("PolicyLifecycleManager created — awaiting configure.")

    # ================================================================== #
    # Startup pre-load
    # ================================================================== #
    def _pre_configure_once(self):
        if self._pre_configure_done:
            return
        self._pre_configure_done = True
        try:
            ckpt = self.get_parameter("checkpoint_path").value
            self._policy    = self._load_policy(ckpt)
            self._setup_pubsub()
            self.get_logger().info("Policy pre-loaded — Run Policy click will be instant.")
        except Exception as e:
            self.get_logger().warn(f"Pre-configure failed: {e}")

    def _setup_pubsub(self):
        """Create publishers and subscribers (idempotent)."""
        if hasattr(self, "_pub_cmd"):
            return  # already done
        self._pub_cmd  = self.create_publisher(JointState, "/sim/joint_command",    10)
        self._pub_stat = self.create_publisher(String,     "/policy/status",         1)
        self._pub_fps  = self.create_publisher(Float32,    "/policy/inference_fps",  10)
        self._pub_conf = self.create_publisher(Float32,    "/policy/confidence",     10)

        self._sub_img = self.create_subscription(
            CompressedImage, "/sim/camera/image_compressed",
            self._cb_image, 10
        )
        self._sub_wrist = self.create_subscription(
            CompressedImage, "/sim/camera/wrist/image_compressed",
            self._cb_wrist_image, 10
        )
        self._sub_js = self.create_subscription(
            JointState, "/sim/joint_states", self._cb_joints, 10
        )

    # ================================================================== #
    # Lifecycle callbacks
    # ================================================================== #
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        if self._pre_configure_done and self._policy is not None:
            self.get_logger().info("Already pre-loaded — skipping configure.")
            self._publish_status("configured")
            return TransitionCallbackReturn.SUCCESS

        self.get_logger().info("Configuring: loading LeRobot ACT checkpoint…")
        ckpt = self.get_parameter("checkpoint_path").value
        try:
            self._policy = self._load_policy(ckpt)
        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint: {e}")
            return TransitionCallbackReturn.FAILURE

        self._setup_pubsub()
        self._publish_status("configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating: starting inference loop.")
        with self._lock:
            self._latest_image       = None
            self._latest_wrist_image = None
            self._latest_joints      = None
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()
        self._publish_status("active")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating: stopping inference.")
        self._running = False
        if hasattr(self, "_inference_thread"):
            self._inference_thread.join(timeout=2.0)
        self._publish_status("inactive")
        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: State) -> TransitionCallbackReturn:
        self._policy = None
        self._publish_status("unconfigured")
        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: State) -> TransitionCallbackReturn:
        self._running = False
        return TransitionCallbackReturn.SUCCESS

    # ================================================================== #
    # Observation callbacks
    # ================================================================== #
    def _cb_image(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._lock:
                self._latest_image = frame

    def _cb_wrist_image(self, msg: CompressedImage):
        buf   = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._lock:
                self._latest_wrist_image = frame

    def _cb_joints(self, msg: JointState):
        with self._lock:
            self._latest_joints = list(msg.position)

    # ================================================================== #
    # Inference loop
    # ================================================================== #
    def _inference_loop(self):
        fps_target = self.get_parameter("inference_fps").value
        dt   = 1.0 / fps_target
        step = 0

        while self._running:
            t0 = time.time()

            with self._lock:
                image       = self._latest_image.copy()       if self._latest_image       is not None else None
                wrist_image = self._latest_wrist_image.copy() if self._latest_wrist_image is not None else None
                joints      = list(self._latest_joints)       if self._latest_joints      is not None else None

            # Wait until we have top camera + joints.
            # Wrist camera is optional: if checkpoint was trained without it,
            # _use_wrist is False and wrist_image is ignored.
            if image is None or joints is None:
                time.sleep(dt)
                continue
            if self._use_wrist and wrist_image is None:
                time.sleep(dt)
                continue

            obs    = self._build_obs(image, wrist_image, joints)
            action = self._predict(obs)
            self._publish_command(action)

            # Debug log every 30 steps
            step += 1
            if step % 30 == 1:
                j = joints[:32]
                self.get_logger().info(
                    f"[step {step}] EEF ({j[0]:.3f},{j[1]:.3f},{j[2]:.3f}) "
                    f"grip={j[13]:.3f} | "
                    f"act dx={action[0]:.4f} dy={action[1]:.4f} "
                    f"dz={action[2]:.4f} g={action[3]:.4f} "
                    f"conf={self._confidence:.3f}"
                )

            elapsed = time.time() - t0
            self._fps_window.append(1.0 / max(elapsed, 1e-6))
            self._pub_fps.publish(Float32(data=float(np.mean(self._fps_window))))
            self._pub_conf.publish(Float32(data=float(self._confidence)))

            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ================================================================== #
    # Trigger service handlers (browser UI)
    # ================================================================== #
    def _srv_run(self, _req, response):
        """/policy/run — always does a clean restart."""
        try:
            if self._running:
                self.on_deactivate(None)

            if self._policy is None:
                result = self.on_configure(None)
                if result != TransitionCallbackReturn.SUCCESS:
                    response.success = False
                    response.message = "Configure failed — check checkpoint path."
                    return response

            self.on_activate(None)
            cam_str = "top+wrist" if self._use_wrist else "top-only"
            response.success = True
            response.message = f"ACT inference active ({cam_str})."
        except Exception as e:
            self._running = False
            response.success = False
            response.message = f"Policy run error: {e}"
        return response

    def _srv_stop(self, _req, response):
        """/policy/stop — deactivate and restore teleop."""
        try:
            self.on_deactivate(None)
            response.success = True
            response.message = "Policy deactivated — teleop restored."
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    # ================================================================== #
    # Model helpers
    # ================================================================== #
    def _load_policy(self, ckpt_path: str):
        """
        Load ACT policy checkpoint.  Supports two formats:

        1. Local .pt file (train_act.py output) — detected when ckpt_path ends
           with '.pt' or is a regular file.
           Keys: state_dict, state_dim, action_dim, use_vision, img_dim, …

        2. LeRobot pretrained directory — detected when ckpt_path is a directory.
           Loaded via ACTPolicy.from_pretrained().
        """
        import os
        if not _TORCH_AVAILABLE:
            self.get_logger().warn("torch not available — random policy.")
            return None

        if not os.path.exists(ckpt_path):
            self.get_logger().warn(f"Checkpoint not found at {ckpt_path}. Random policy.")
            return None

        # ---- Local .pt checkpoint (browser training via train_act.py) ----
        if os.path.isfile(ckpt_path):
            return self._load_local_pt_policy(ckpt_path)

        # ---- LeRobot pretrained directory (cluster lerobot-train) ----
        return self._load_lerobot_policy(ckpt_path)

    def _load_local_pt_policy(self, pt_path: str):
        """Load a .pt checkpoint saved by train_act.py."""
        try:
            ckpt = torch.load(pt_path, map_location="cpu", weights_only=False)
            state_dim  = int(ckpt.get("state_dim",  32))
            action_dim = int(ckpt.get("action_dim", 4))
            use_vision = bool(ckpt.get("use_vision", False))
            img_dim    = int(ckpt.get("img_dim",    128))

            if use_vision:
                model = _ACTVisionPolicy(
                    state_dim=state_dim, action_dim=action_dim,
                    img_dim=img_dim, hidden=512,
                )
                self._use_wrist = False   # local vision model uses top camera only
            else:
                model = _ACTMiniPolicy(
                    state_dim=state_dim, action_dim=action_dim, hidden=512,
                )
                self._use_wrist = False

            model.load_state_dict(ckpt["state_dict"])
            model.eval()

            # Run on CPU (Docker container may not have CUDA; fast enough for 4D output)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            self.get_logger().info(
                f"Local ACT checkpoint loaded — "
                f"state_dim={state_dim}, action_dim={action_dim}, "
                f"mode={'vision+state' if use_vision else 'state-only'}, "
                f"epoch={ckpt.get('epoch', '?')}, "
                f"eval_loss={ckpt.get('eval_loss', float('nan')):.5f}"
            )
            return _LocalACTWrapper(model, device, use_vision)

        except Exception as e:
            self.get_logger().warn(f"Could not load local .pt checkpoint: {e}. Random policy.")
            return None

    def _load_lerobot_policy(self, ckpt_path: str):
        """Load a LeRobot pretrained_model directory (cluster training output)."""
        try:
            from lerobot.policies.act.modeling_act import ACTPolicy
            policy = ACTPolicy.from_pretrained(ckpt_path)
            policy.eval()

            # Detect wrist camera from config
            cfg = policy.config
            input_keys = list(getattr(cfg, "input_features", {}).keys())
            self._use_wrist = "observation.images.wrist" in input_keys

            self.get_logger().info(
                f"LeRobot ACTPolicy loaded — "
                f"chunk_size={cfg.chunk_size}, "
                f"n_action_steps={cfg.n_action_steps}, "
                f"cameras={'top+wrist' if self._use_wrist else 'top-only'}"
            )
            return policy

        except Exception as e:
            self.get_logger().warn(f"Could not load LeRobot checkpoint: {e}. Random policy.")
            return None

    def _build_obs(self, image: np.ndarray,
                   wrist_image: np.ndarray | None,
                   joints: list) -> dict:
        """
        Build observation batch for LeRobot ACTPolicy.select_action().

          observation.state         : (1, 32)  float32
          observation.images.top    : (1, 3, 480, 640)  float32  [0,1]
          observation.images.wrist  : (1, 3, 128, 128)  float32  [0,1]  (if used)
        """
        state = joints[:32]
        if len(state) < 32:
            state = state + [0.0] * (32 - len(state))

        obs = {
            "observation.state": torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0),   # (1, 32)
        }

        # Top camera — Docker headless renders at 320×240; resize to training res
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (self.TOP_H, self.TOP_W):
            img = cv2.resize(img, (self.TOP_W, self.TOP_H),
                             interpolation=cv2.INTER_LINEAR)
        obs["observation.images.top"] = torch.from_numpy(
            (img.astype(np.float32) / 255.0).transpose(2, 0, 1)
        ).unsqueeze(0)   # (1, 3, 480, 640)

        # Wrist camera (only included when checkpoint was trained with it)
        if self._use_wrist and wrist_image is not None:
            w = cv2.cvtColor(wrist_image, cv2.COLOR_BGR2RGB)
            if w.shape[:2] != (self.WRIST_H, self.WRIST_W):
                w = cv2.resize(w, (self.WRIST_W, self.WRIST_H),
                               interpolation=cv2.INTER_LINEAR)
            obs["observation.images.wrist"] = torch.from_numpy(
                (w.astype(np.float32) / 255.0).transpose(2, 0, 1)
            ).unsqueeze(0)   # (1, 3, 128, 128)

        return obs

    def _predict(self, obs: dict) -> list:
        """
        One inference step.  LeRobot's select_action() manages the internal
        _action_queue — it re-predicts automatically every n_action_steps.
        Returns a single [dx, dy, dz, gripper] action.
        """
        if self._policy is None:
            self._confidence = 0.0
            return np.random.uniform(-0.02, 0.02, size=4).tolist()

        try:
            with torch.no_grad():
                action = self._policy.select_action(obs)   # (4,) or (1,4)

            action_np = action.squeeze().cpu().numpy()     # (4,)
            action_np[:3] = np.clip(action_np[:3], -0.4, 0.4)
            action_np[3]  = np.clip(action_np[3],  -1.0, 1.0)
            self._confidence = float(np.abs(action_np).mean())
            return action_np.tolist()

        except Exception as e:
            self.get_logger().warn(f"Inference error: {e}")
            self._confidence = 0.0
            return [0.0, 0.0, 0.0, 0.0]

    def _publish_command(self, action: list):
        cmd              = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name         = ["eef_x", "eef_y", "eef_z", "gripper"]
        cmd.position     = [float(v) for v in action[:4]]
        self._pub_cmd.publish(cmd)

    def _publish_status(self, status: str):
        if hasattr(self, "_pub_stat"):
            self._pub_stat.publish(String(data=status))


def main(args=None):
    rclpy.init(args=args)
    node = PolicyLifecycleManager()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
