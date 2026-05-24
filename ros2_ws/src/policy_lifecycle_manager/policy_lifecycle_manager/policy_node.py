"""
policy_lifecycle_manager/policy_node.py
========================================
ROS2 Lifecycle Node that loads an ACT checkpoint and runs inference.

Lifecycle states
----------------
  unconfigured  No model loaded. No commands published.
  configured    Model loaded in memory. No commands published.
  active        Inference running. Publishes /sim/joint_command at policy FPS.
  deactivated   Inference paused. Falls back to teleop.
  shutdown      Clean teardown.

The MYBOTSHOP webserver action buttons call ROS2 lifecycle transitions:
  "Configure Policy"   → ros2 lifecycle set /policy_node configure
  "Activate Policy"    → ros2 lifecycle set /policy_node activate
  "Deactivate Policy"  → ros2 lifecycle set /policy_node deactivate

Published topics
----------------
/sim/joint_command        sensor_msgs/JointState      inference output
/policy/status            std_msgs/String             state label
/policy/inference_fps     std_msgs/Float32            rolling avg FPS
/policy/confidence        std_msgs/Float32            action confidence score

Subscribed topics
-----------------
/sim/camera/image_compressed  sensor_msgs/CompressedImage  visual observation
/sim/joint_states             sensor_msgs/JointState       proprioceptive obs
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

# ---------------------------------------------------------------------------
# ACTMiniPolicy — must match training/train_act.py exactly
# (copied here so the inference container doesn't need the training package)
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn

    class ACTMiniPolicy(nn.Module):
        """4-layer MLP with residual connections. Maps 32D state → 4D action."""
        def __init__(self, state_dim: int = 32, action_dim: int = 4, hidden: int = 512):
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
            z = z + self.residual(z)
            return self.head(z)

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class PolicyLifecycleManager(LifecycleNode):
    """
    ACT policy inference node following the ROS2 managed lifecycle.
    On configure: loads checkpoint from /data/checkpoints/act_xarm_lift.pt
    On activate:  starts inference loop at ~30 Hz
    """

    def __init__(self):
        super().__init__("policy_node")

        self.declare_parameter("checkpoint_path", "/data/checkpoints/xarm_lift_v1/act_xarm_lift.pt")
        self.declare_parameter("inference_fps",   30.0)
        self.declare_parameter("action_horizon",  8)    # ACT chunking horizon
        self.declare_parameter("confidence_threshold", 0.3)

        self._policy    = None
        self._device    = "cpu"   # M1 via Docker: use CPU (MPS not available in container)
        self._lock      = threading.Lock()

        # Rolling observations (latest)
        self._latest_image  = None
        self._latest_joints = None

        # Action chunk buffer (ACT predicts a horizon of actions at once)
        self._action_chunk: list = []
        self._chunk_idx = 0

        # Metrics
        self._fps_window = collections.deque(maxlen=30)
        self._confidence = 0.0

        # Simple Trigger services for browser UI (avoids lifecycle_msgs/ChangeState
        # which rosbridge can't resolve without extra type introspection).
        # /policy/run  → configure (if needed) + activate
        # /policy/stop → deactivate
        self.create_service(Trigger, "/policy/run",  self._srv_run)
        self.create_service(Trigger, "/policy/stop", self._srv_stop)

        self.get_logger().info("PolicyLifecycleManager created — awaiting configure.")

    # ================================================================== #
    # Lifecycle callbacks
    # ================================================================== #
    def on_configure(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Configuring: loading ACT checkpoint…")

        ckpt = self.get_parameter("checkpoint_path").value
        try:
            self._policy = self._load_policy(ckpt)
            self.get_logger().info(f"Checkpoint loaded: {ckpt}")
        except Exception as e:
            self.get_logger().error(f"Failed to load checkpoint: {e}")
            return TransitionCallbackReturn.FAILURE

        # Publishers (created on configure, not in __init__)
        self._pub_cmd   = self.create_publisher(JointState,  "/sim/joint_command",    10)
        self._pub_stat  = self.create_publisher(String,      "/policy/status",         1)
        self._pub_fps   = self.create_publisher(Float32,     "/policy/inference_fps",  10)
        self._pub_conf  = self.create_publisher(Float32,     "/policy/confidence",     10)

        # Subscribers
        self._sub_img = self.create_subscription(
            CompressedImage, "/sim/camera/image_compressed", self._cb_image, 10
        )
        self._sub_js = self.create_subscription(
            JointState, "/sim/joint_states", self._cb_joints, 10
        )

        self._publish_status("configured")
        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Activating: starting inference loop.")
        # Clear any stale observations and action chunks from a previous run
        self._action_chunk = []
        self._chunk_idx    = 0
        with self._lock:
            self._latest_image  = None
            self._latest_joints = None
        self._running = True
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()
        self._publish_status("active")
        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: State) -> TransitionCallbackReturn:
        self.get_logger().info("Deactivating: stopping inference, restoring teleop.")
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
        self.get_logger().info("PolicyLifecycleManager shutting down.")
        return TransitionCallbackReturn.SUCCESS

    # ================================================================== #
    # Observation callbacks
    # ================================================================== #
    def _cb_image(self, msg: CompressedImage):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is not None:
            with self._lock:
                self._latest_image = frame

    def _cb_joints(self, msg: JointState):
        with self._lock:
            self._latest_joints = list(msg.position)

    # ================================================================== #
    # Inference loop (runs in background thread while active)
    # ================================================================== #
    def _inference_loop(self):
        fps_target = self.get_parameter("inference_fps").value
        dt = 1.0 / fps_target
        horizon = self.get_parameter("action_horizon").value

        while self._running:
            t0 = time.time()

            with self._lock:
                image  = self._latest_image.copy()  if self._latest_image  is not None else None
                joints = list(self._latest_joints)  if self._latest_joints is not None else None

            if image is None or joints is None:
                time.sleep(dt)
                continue

            # -- Action chunking: re-predict every 'horizon' steps --------- #
            if self._chunk_idx >= len(self._action_chunk):
                obs = self._build_obs(image, joints)
                self._action_chunk = self._predict(obs, horizon)
                self._chunk_idx = 0

            action = self._action_chunk[self._chunk_idx]
            self._chunk_idx += 1

            self._publish_command(action)

            # -- Metrics --------------------------------------------------- #
            elapsed = time.time() - t0
            self._fps_window.append(1.0 / max(elapsed, 1e-6))
            avg_fps = float(np.mean(self._fps_window))

            self._pub_fps.publish(Float32(data=avg_fps))
            self._pub_conf.publish(Float32(data=float(self._confidence)))

            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ================================================================== #
    # Model helpers
    # ================================================================== #
    # ================================================================== #
    # Simple browser-facing Trigger services
    # ================================================================== #
    def _srv_run(self, _request, response):
        """
        /policy/run — called by 'Run Policy' button in the webserver UI.

        Always does a clean restart:
          1. Stop the current inference thread if running (deactivate).
          2. Load the checkpoint if not already loaded (configure).
          3. Reset action chunk and start a fresh inference thread (activate).

        This means clicking Run Policy a second time after Reset Env always
        works correctly, without needing to rebuild Docker.
        """
        try:
            # Step 1: stop any running inference thread first
            if self._running:
                self.get_logger().info("Re-run requested — deactivating current session.")
                self.on_deactivate(None)

            # Step 2: load checkpoint (only once per container lifetime)
            if not hasattr(self, '_policy') or self._policy is None:
                result = self.on_configure(None)
                if result != TransitionCallbackReturn.SUCCESS:
                    response.success = False
                    response.message = "Configure failed — check checkpoint path."
                    return response

            # Step 3: fresh activate — resets action chunk, starts new thread
            self.on_activate(None)
            response.success = True
            response.message = "Policy active — ACT inference running."

        except Exception as e:
            self._running = False
            response.success = False
            response.message = f"Policy run error: {e}"

        return response

    def _srv_stop(self, _request, response):
        """/policy/stop — called by 'Stop Policy' button in the webserver UI."""
        try:
            self.on_deactivate(None)
            response.success = True
            response.message = "Policy deactivated — teleop restored."
        except Exception as e:
            response.success = False
            response.message = str(e)
        return response

    def _load_policy(self, ckpt_path: str):
        """
        Load ACTMiniPolicy from a checkpoint saved by training/train_act.py.
        Falls back to random policy if checkpoint not found (demo mode).
        """
        import os
        if not _TORCH_AVAILABLE:
            self.get_logger().warn("torch not available — using random policy.")
            return None

        if not os.path.exists(ckpt_path):
            self.get_logger().warn(
                f"Checkpoint not found at {ckpt_path}. Using random policy."
            )
            return None

        try:
            import torch
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dim  = ckpt.get("state_dim",  32)
            action_dim = ckpt.get("action_dim",  4)
            policy = ACTMiniPolicy(state_dim=state_dim, action_dim=action_dim, hidden=512)
            policy.load_state_dict(ckpt["state_dict"])
            policy.eval()
            self.get_logger().info(
                f"ACTMiniPolicy loaded — state_dim={state_dim}, "
                f"action_dim={action_dim}, eval_loss={ckpt.get('eval_loss', '?'):.5f}"
            )
            # Store dims for obs building
            self._state_dim  = state_dim
            self._action_dim = action_dim
            return policy
        except Exception as e:
            self.get_logger().warn(f"Could not load checkpoint: {e}. Using random policy.")
            return None

    def _build_obs(self, image: np.ndarray, joints: list) -> dict:
        """
        Package the full 32D proprioceptive state into a tensor.
        joints is the full list from /sim/joint_states position field (32 values).
        """
        import torch
        state_dim = getattr(self, "_state_dim", 32)
        # Pad or truncate to state_dim
        state = joints[:state_dim]
        if len(state) < state_dim:
            state = state + [0.0] * (state_dim - len(state))
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, 32)
        return {"observation.state": state_t}

    def _predict(self, obs: dict, horizon: int) -> list:
        """
        Run ACTMiniPolicy inference and return a list of `horizon` identical
        actions (state-only MLP predicts one action per forward pass; we
        repeat it for the full chunk so the action-chunking buffer works).
        """
        if self._policy is None:
            self._confidence = 0.0
            return [
                np.random.uniform(-0.02, 0.02, size=4).tolist()
                for _ in range(horizon)
            ]

        import torch
        with torch.no_grad():
            action = self._policy(obs["observation.state"])   # (1, 4)

        action_np = action.squeeze(0).cpu().numpy()   # (4,)

        # Clamp XYZ deltas to SENSITIVITY range used during teleoperation (±0.4).
        # Gripper stays in full [-1, 1] range.
        action_np[:3] = np.clip(action_np[:3], -0.4, 0.4)
        action_np[3]  = np.clip(action_np[3],  -1.0, 1.0)

        self._confidence = float(np.abs(action_np).mean())
        # Repeat for full chunk — policy re-predicts every `horizon` steps
        return [action_np.tolist() for _ in range(horizon)]

    def _publish_command(self, action: list):
        """Publish one action step as a JointState command."""
        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name     = ["eef_x", "eef_y", "eef_z", "gripper"]
        cmd.position = [float(v) for v in action[:4]]
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
