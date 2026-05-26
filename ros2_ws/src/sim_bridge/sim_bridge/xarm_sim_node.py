"""
sim_bridge/xarm_sim_node.py
===========================
Bridges the gym_xarm MuJoCo simulation into ROS2.

Production-grade multi-modal observation space (32D proprioceptive):
--------------------------------------------------------------------
  [0:3]   EEF Position      ee_x, ee_y, ee_z                 (m)
  [3:7]   EEF Orientation   ee_qx, ee_qy, ee_qz, ee_qw       (quaternion)
  [7:10]  EEF Linear Vel    ee_vx, ee_vy, ee_vz              (m/s)
  [10:13] EEF Angular Vel   ee_wx, ee_wy, ee_wz              (rad/s)
  [13]    Gripper           gripper_opening                   [0=closed, 1=open]
  [14:20] Joint Positions   j1..j6                            (rad)
  [20:26] Joint Velocities  dj1..dj6                         (rad/s)
  [26:32] F/T Wrench        ft_fx, ft_fy, ft_fz, ft_tx, ft_ty, ft_tz
                                                              (N, N·m)
TOTAL: 32D

Design rationale (mirrors AIC UR5e 30D dataset):
  - Real Intrinsic AIC robot used:  TCP pos+quat+vel + joints + (partial FT)
  - MYBOTSHOP adds full 6D F/T wrench and explicit gripper channel
  - Enables sim-to-real transfer to any Cartesian-controlled 6-DOF arm
  - Safety watchdog can enforce joint, velocity, and force limits

Published topics
----------------
/sim/camera/image_compressed   sensor_msgs/CompressedImage   @ 30 Hz
/sim/joint_states              sensor_msgs/JointState        @ 30 Hz  (32D state as "position")
/sim/ee_pose                   geometry_msgs/PoseStamped     @ 30 Hz
/sim/wrench                    geometry_msgs/WrenchStamped   @ 30 Hz
/sim/task_complete             std_msgs/Bool                 latched

Subscribed topics
-----------------
/sim/joint_command             sensor_msgs/JointState        EEF action targets
/sim/reset                     std_msgs/Empty                reset the episode

Services
--------
/sim/set_render_mode           std_srvs/SetBool              reserved for future camera switching
"""

import os
import platform
import threading
import time

# MuJoCo backend: glfw on macOS, osmesa inside Docker/Linux (headless).
# Must be set before any MuJoCo/gym import.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "glfw" if platform.system() == "Darwin" else "osmesa"

import cv2
import gymnasium as gym
import gym_xarm  # noqa: F401  registers gym_xarm envs
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from geometry_msgs.msg import PoseStamped, WrenchStamped, Vector3
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Bool, Empty
from std_srvs.srv import Empty as EmptySrv, SetBool

# gym_xarm uses end-effector (Cartesian) control.
# agent_pos = [ee_x, ee_y, ee_z, gripper_opening] → 4 values from the env.
# We extend this by reading MuJoCo internal state for the full 32D observation.

# State channel layout — index ranges into the 32D vector published on /sim/joint_states
STATE_NAMES = [
    # EEF Pose (7)
    "ee_x", "ee_y", "ee_z",
    "ee_qx", "ee_qy", "ee_qz", "ee_qw",
    # EEF Velocity (6)
    "ee_vx", "ee_vy", "ee_vz",
    "ee_wx", "ee_wy", "ee_wz",
    # Gripper (1)
    "gripper",
    # Joint Positions — xARM6 (6)
    "j1", "j2", "j3", "j4", "j5", "j6",
    # Joint Velocities (6)
    "dj1", "dj2", "dj3", "dj4", "dj5", "dj6",
    # F/T Wrench (6)
    "ft_fx", "ft_fy", "ft_fz",
    "ft_tx", "ft_ty", "ft_tz",
]
STATE_DIM = len(STATE_NAMES)  # 32

# xARM6 has 6 revolute joints (joint1–joint6) + 1 gripper joint (drive_joint).
# The gym_xarm MJCF model typically places the object freejoint first in qpos.
XARM_N_JOINTS = 6  # arm joints only (excludes gripper)

# gym_xarm task — XArmLift: lift a red cube off the table
GYM_ENV_ID = "gym_xarm/XarmLift-v0"

# Render resolution — reduced for headless Docker (OSMesa software renderer).
# OSMesa on ARM64 can sustain ~9 Hz at 640×480; 320×240 brings it to ~25 Hz.
# For real deployment with a GPU or display, restore to 640×480.
_headless = (os.environ.get("MUJOCO_GL", "") == "osmesa")
RENDER_HEIGHT = 240 if _headless else 480
RENDER_WIDTH  = 320 if _headless else 640
TARGET_FPS    = 30

# Wrist camera: rendered at 128×128 — small enough for low overhead,
# large enough for the policy to see cube alignment under the gripper.
WRIST_HEIGHT = 128
WRIST_WIDTH  = 128

# Publish camera image every N sim steps (decouples image rate from state rate).
# State (32D) publishes every step; image publishes every IMAGE_EVERY steps.
# At 30 Hz sim, IMAGE_EVERY=3 → image at 10 Hz — enough for behaviour cloning.
IMAGE_EVERY = 3 if _headless else 1
DEFAULT_CAMERA = "side"


# ---------------------------------------------------------------------------
# Maths helpers
# ---------------------------------------------------------------------------
def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    Convert a 3×3 rotation matrix to a unit quaternion [x, y, z, w].
    Uses Shepperd's method — numerically stable for all rotations.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def _get_robot_joint_indices(model) -> tuple[list[int], list[int]]:
    """
    Introspect the MuJoCo model to find qpos/qvel slices for the xARM6 arm joints.

    Returns (qpos_indices, qvel_indices) — lists of ints into data.qpos / data.qvel.

    Strategy: look for joints named 'joint1'–'joint6'; fall back to the last
    XARM_N_JOINTS revolute joints in the model if names aren't found.
    """
    try:
        import mujoco

        qpos_idx, qvel_idx = [], []
        target_names = {f"joint{i}" for i in range(1, 7)}

        for jid in range(model.njnt):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            jtype = model.jnt_type[jid]
            # mjtJoint: 0=free, 1=ball, 3=hinge (revolute), 4=slide
            if jtype == 3 and name in target_names:  # hinge joint with expected name
                qpos_idx.append(int(model.jnt_qposadr[jid]))
                qvel_idx.append(int(model.jnt_dofadr[jid]))

        if len(qpos_idx) == XARM_N_JOINTS:
            return sorted(qpos_idx), sorted(qvel_idx)

        # Fallback: last XARM_N_JOINTS hinge joints by qpos address
        hinge_joints = [
            (int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid]))
            for jid in range(model.njnt)
            if model.jnt_type[jid] == 3  # hinge
        ]
        if len(hinge_joints) >= XARM_N_JOINTS:
            hinge_joints.sort()
            arm = hinge_joints[:XARM_N_JOINTS]  # first 6 hinge = arm joints
            return [a[0] for a in arm], [a[1] for a in arm]

    except Exception:
        pass

    # Last-resort fallback: assume object freejoint (7 qpos) + arm (6) layout
    return list(range(7, 7 + XARM_N_JOINTS)), list(range(6, 6 + XARM_N_JOINTS))


def _get_eef_site_id(model) -> int:
    """
    Find the MuJoCo site ID for the end-effector (TCP).
    Tries common names used in gym_xarm MJCF files.
    gym_xarm XarmLift uses "grasp"; other envs may differ.
    """
    try:
        import mujoco
        for candidate in ["grasp", "end_effector", "eef_site", "tcp",
                          "grasp_site", "grip_site", "attachment_site"]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, candidate)
            if sid >= 0:
                return sid
        # Fallback: first site in the model
        if model.nsite > 0:
            return 0
    except Exception:
        pass
    return -1


def _site_velocity(model, data, site_id: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute translational and rotational velocity of a MuJoCo site.

    Uses the Jacobian method (mj_jacSite) which is version-stable across all
    MuJoCo Python bindings.  Avoids relying on data.site_xvelp / site_xvelr
    which are not present in some mujoco-py / mujoco<3 versions.

    Returns:
        lin_vel : (3,) float32  translational velocity (m/s)
        ang_vel : (3,) float32  angular velocity (rad/s)
    """
    try:
        import mujoco
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        qvel     = np.array(data.qvel, dtype=np.float64)
        lin_vel  = (jacp @ qvel).astype(np.float32)
        ang_vel  = (jacr @ qvel).astype(np.float32)
        return lin_vel, ang_vel
    except Exception:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)


class XArmSimNode(Node):
    """
    Runs the gym_xarm simulation in a background thread and exposes
    its observations as ROS2 topics.

    Publishes the full 32D production observation vector on /sim/joint_states
    (using the "position" field to carry all 32 channels for ROS2 compatibility).
    """

    def __init__(self):
        super().__init__("xarm_sim_node")

        # ------------------------------------------------------------------ #
        # Gym environment
        # ------------------------------------------------------------------ #
        self.get_logger().info(f"Loading simulation: {GYM_ENV_ID}")
        self._env = gym.make(
            GYM_ENV_ID,
            obs_type="pixels_agent_pos",
            render_mode="rgb_array",
            visualization_width=RENDER_WIDTH,
            visualization_height=RENDER_HEIGHT,
            # Disable the gym TimeLimit wrapper — the recording_manager controls
            # episode boundaries via Start/Stop buttons, not a step counter.
            max_episode_steps=None,
        )
        self._obs, _ = self._env.reset()
        self._action = np.zeros(self._env.action_space.shape, dtype=np.float32)
        self._done   = False
        self._lock   = threading.Lock()

        # Pre-compute joint indices and gripper geom IDs from model introspection
        try:
            import mujoco as _mj
            model = self._env.unwrapped.model
            self._qpos_idx, self._qvel_idx = _get_robot_joint_indices(model)
            self._eef_site_id = _get_eef_site_id(model)
            self.get_logger().info(
                f"MuJoCo model introspection: "
                f"arm qpos_idx={self._qpos_idx}, "
                f"eef_site_id={self._eef_site_id}"
            )

            # Build set of geom IDs belonging to gripper/wrist bodies.
            # Used by the contact-force F/T computation.
            # Names confirmed from gym_xarm XarmLift-v0 model introspection.
            _gripper_body_names = [
                "link7",                                          # wrist
                "left_outer_knuckle", "left_inner_knuckle", "left_finger", "left_hand",
                "right_outer_knuckle", "right_inner_knuckle", "right_finger", "right_hand",
            ]
            gripper_body_ids: set[int] = set()
            for bname in _gripper_body_names:
                bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, bname)
                if bid >= 0:
                    gripper_body_ids.add(bid)

            # Map body IDs → geom IDs
            self._gripper_geom_ids: set[int] = {
                gid for gid in range(model.ngeom)
                if model.geom_bodyid[gid] in gripper_body_ids
            }
            self.get_logger().info(
                f"F/T sensor: tracking {len(gripper_body_ids)} gripper bodies, "
                f"{len(self._gripper_geom_ids)} geoms"
            )

        except Exception as e:
            self.get_logger().warn(f"Model introspection failed ({e}), using defaults.")
            self._qpos_idx       = list(range(7, 7 + XARM_N_JOINTS))
            self._qvel_idx       = list(range(6, 6 + XARM_N_JOINTS))
            self._eef_site_id    = -1
            self._gripper_geom_ids: set[int] = set()

        # ------------------------------------------------------------------ #
        # Wrist camera — free camera that follows the EEF site each frame.
        #
        # Design notes:
        #   • Uses mjCAMERA_FREE (not TRACKING): tracking cameras in mujoco
        #     Python 2.x do not update lookat via update_scene(), so the view
        #     defaults to world origin [0,0,0] while the robot is at X≈-1.8m
        #     → black image.
        #   • Dedicated mujoco.Renderer (separate from gym's renderer).
        #     gym_xarm's Lift class does NOT expose mujoco_renderer, so we
        #     cannot reuse gym's renderer. A dedicated renderer is fine here:
        #     the lookat bug (not dual-context) was the original cause of the
        #     black image, and that is fixed by using mjCAMERA_FREE.
        #   • lookat is updated to data.site_xpos[eef_site_id] each step.
        #   • elevation = -75: in MuJoCo convention this puts the camera
        #     above the lookat looking mostly downward (confirmed by default
        #     viewer using elevation = -20 for an above-scene view).
        # ------------------------------------------------------------------ #
        self._wrist_cam      = None
        self._wrist_renderer = None
        try:
            import mujoco as _mj
            model = self._env.unwrapped.model

            # Confirm body exists (for logging only)
            wrist_body_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, "link7")
            if wrist_body_id < 0:
                for candidate in ["link_tcp", "link6", "wrist", "gripper_base"]:
                    wrist_body_id = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, candidate)
                    if wrist_body_id >= 0:
                        break

            cam = _mj.MjvCamera()
            cam.type      = int(_mj.mjtCamera.mjCAMERA_FREE)
            cam.distance  = 0.30   # 30 cm from lookat
            cam.elevation = -75.0  # camera above, looking mostly down (MuJoCo: negative = above)
            cam.azimuth   = 90.0   # face same direction as default viewer
            # lookat will be set to EEF site position on every render frame
            self._wrist_cam = cam

            # gym_xarm model XML sets offscreen framebuffer to 84×84 by default.
            # gymnasium lazily creates its own renderer (on first render() call),
            # so it hasn't bumped the offscreen size yet when we get here.
            # Manually raise offwidth/offheight to fit both renderers now.
            model.vis.global_.offwidth  = max(
                int(model.vis.global_.offwidth),  RENDER_WIDTH,  WRIST_WIDTH
            )
            model.vis.global_.offheight = max(
                int(model.vis.global_.offheight), RENDER_HEIGHT, WRIST_HEIGHT
            )

            # Dedicated renderer — renders at wrist resolution directly
            self._wrist_renderer = _mj.Renderer(model, WRIST_HEIGHT, WRIST_WIDTH)

            self.get_logger().info(
                f"Wrist camera ready (FREE cam, body id={wrist_body_id}). "
                f"Rendering at {WRIST_WIDTH}×{WRIST_HEIGHT} via dedicated mujoco.Renderer."
            )
        except Exception as e:
            self.get_logger().warn(f"Wrist camera init failed: {e}. Wrist feed disabled.")

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self._bridge = CvBridge()

        self._pub_img       = self.create_publisher(
            CompressedImage,  "/sim/camera/image_compressed", 10)
        self._pub_wrist_img = self.create_publisher(
            CompressedImage,  "/sim/camera/wrist/image_compressed", 10)
        self._pub_js     = self.create_publisher(
            JointState,       "/sim/joint_states", 10)
        self._pub_ee     = self.create_publisher(
            PoseStamped,      "/sim/ee_pose", 10)
        self._pub_wrench = self.create_publisher(
            WrenchStamped,    "/sim/wrench", 10)
        self._pub_done   = self.create_publisher(
            Bool,             "/sim/task_complete", 1)

        # ------------------------------------------------------------------ #
        # Subscribers
        # ------------------------------------------------------------------ #
        self.create_subscription(
            JointState, "/sim/joint_command", self._cb_joint_cmd, 10)
        self.create_subscription(
            Empty, "/sim/reset", self._cb_reset, 1)

        # ------------------------------------------------------------------ #
        # Services
        # ------------------------------------------------------------------ #
        # /sim/reset as a SERVICE (called by webserver Reset button via rosbridge)
        # Also keep the topic subscription for CLI use: ros2 topic pub /sim/reset
        self.create_service(EmptySrv, "/sim/reset", self._srv_reset)
        self.create_service(SetBool, "/sim/set_render_mode", self._srv_render)

        # ------------------------------------------------------------------ #
        # Simulation loop timer
        # ------------------------------------------------------------------ #
        self._step_count = 0
        self._timer = self.create_timer(1.0 / TARGET_FPS, self._sim_step)

        self.get_logger().info(
            f"XArmSimNode ready — 32D multi-modal observation, {TARGET_FPS} Hz "
            f"| render: {RENDER_WIDTH}×{RENDER_HEIGHT} every {IMAGE_EVERY} steps "
            f"({'headless OSMesa' if _headless else 'display'})."
        )

    # ---------------------------------------------------------------------- #
    # Simulation step
    # ---------------------------------------------------------------------- #
    def _sim_step(self):
        with self._lock:
            if not self._done:
                obs, _reward, terminated, _truncated, _info = self._env.step(self._action)
                self._obs = obs
                # Only stop on TASK SUCCESS (terminated=True).
                # Ignore truncated (gym TimeLimit at 500 steps) — the operator
                # controls episode length via ▶ Start / ⏹ Stop buttons.
                if terminated:
                    self._done = True
                    self._pub_done.publish(Bool(data=True))
                    self.get_logger().info(
                        "Task complete (cube lifted)! "
                        "Click 'Reset Env' for the next episode."
                    )
            # Whether running or paused after success: keep publishing so the
            # camera feed and telemetry stay live.

        self._publish_obs()

    # ---------------------------------------------------------------------- #
    # Build 32D state vector from MuJoCo internal state
    # ---------------------------------------------------------------------- #
    def _extract_state_32d(self) -> np.ndarray:
        """
        Extract the production 32D observation vector from the running MuJoCo
        simulation.  Falls back gracefully to zeros for any unavailable channel.

        Layout matches STATE_NAMES (see module header).
        """
        with self._lock:
            agent_pos = np.array(
                self._obs.get("agent_pos", np.zeros(4)), dtype=float
            )

        ee_pos  = agent_pos[:3].astype(np.float32)   # [ee_x, ee_y, ee_z]
        gripper = agent_pos[3:4].astype(np.float32)  # [gripper_opening]

        try:
            data  = self._env.unwrapped.data
            model = self._env.unwrapped.model
        except Exception:
            data  = None
            model = None

        # ---- EEF Orientation (quaternion from TCP site rotation matrix) ----
        quat = np.array([0., 0., 0., 1.], dtype=np.float32)  # identity fallback
        if data is not None and self._eef_site_id >= 0:
            try:
                mat  = np.array(data.site_xmat[self._eef_site_id]).reshape(3, 3)
                quat = _rotmat_to_quat(mat)
            except Exception:
                pass

        # ---- EEF Linear and Angular Velocity (Jacobian method) -------------
        # Uses mj_jacSite which is stable across all MuJoCo Python versions.
        # Avoids site_xvelp / site_xvelr which are absent in some builds.
        ee_vel_lin = np.zeros(3, dtype=np.float32)
        ee_vel_ang = np.zeros(3, dtype=np.float32)
        if data is not None and model is not None and self._eef_site_id >= 0:
            ee_vel_lin, ee_vel_ang = _site_velocity(model, data, self._eef_site_id)

        # ---- Joint Positions and Velocities --------------------------------
        joint_pos = np.zeros(XARM_N_JOINTS, dtype=np.float32)
        joint_vel = np.zeros(XARM_N_JOINTS, dtype=np.float32)
        if data is not None:
            try:
                joint_pos = np.array(
                    [data.qpos[i] for i in self._qpos_idx], dtype=np.float32
                )
                joint_vel = np.array(
                    [data.qvel[i] for i in self._qvel_idx], dtype=np.float32
                )
            except Exception:
                pass

        # ---- F/T Wrench (contact-force method) --------------------------------
        # data.cfrc_ext is the "externally applied" wrench and stays near-zero
        # in position-controlled simulations.  Instead we iterate over MuJoCo
        # contacts and sum forces on gripper geoms using mj_contactForce, which
        # correctly reports constraint forces exchanged between gripper and object.
        #
        # self._gripper_geom_ids is pre-cached in __init__ from confirmed
        # gym_xarm body names (link7 + all gripper finger bodies).
        wrench = np.zeros(6, dtype=np.float32)
        if data is not None and model is not None and self._gripper_geom_ids:
            try:
                import mujoco
                cf = np.zeros(6, dtype=np.float64)
                for i in range(data.ncon):
                    contact = data.contact[i]
                    g1_in = int(contact.geom1) in self._gripper_geom_ids
                    g2_in = int(contact.geom2) in self._gripper_geom_ids
                    if not (g1_in or g2_in):
                        continue
                    mujoco.mj_contactForce(model, data, i, cf)
                    # contact.frame is the 3×3 contact frame (row-major).
                    # cf[:3] = force in contact frame; cf[3:] = torque in contact frame.
                    frame  = np.array(contact.frame, dtype=np.float64).reshape(3, 3)
                    f_world = (frame.T @ cf[:3]).astype(np.float32)
                    t_world = (frame.T @ cf[3:]).astype(np.float32)
                    # Convention: positive = force ON the gripper.
                    # geom1 receives +force; geom2 receives -force (Newton 3rd law).
                    sign = 1.0 if g1_in else -1.0
                    wrench[:3] += sign * f_world
                    wrench[3:] += sign * t_world
            except Exception:
                pass

        # ---- Concatenate 32D vector ----------------------------------------
        state = np.concatenate([
            ee_pos,      # [0:3]   EEF position (m)
            quat,        # [3:7]   EEF quaternion (x, y, z, w)
            ee_vel_lin,  # [7:10]  EEF linear velocity (m/s)
            ee_vel_ang,  # [10:13] EEF angular velocity (rad/s)
            gripper,     # [13]    gripper opening [0, 1]
            joint_pos,   # [14:20] joint positions (rad)
            joint_vel,   # [20:26] joint velocities (rad/s)
            wrench,      # [26:32] F/T wrench (N, N·m)
        ]).astype(np.float32)

        assert state.shape == (STATE_DIM,), \
            f"State dim mismatch: got {state.shape}, expected ({STATE_DIM},)"
        return state

    # ---------------------------------------------------------------------- #
    # Publish current observation as ROS2 messages
    # ---------------------------------------------------------------------- #
    def _publish_obs(self):
        now = self.get_clock().now().to_msg()
        self._step_count += 1

        # --- Camera image (published every IMAGE_EVERY steps) -------------- #
        # Decouples expensive OSMesa render from the fast state publish rate.
        # State (32D) always publishes at TARGET_FPS; image publishes at
        # TARGET_FPS / IMAGE_EVERY (e.g. 30/3 = 10 Hz in headless mode).
        if self._step_count % IMAGE_EVERY == 0:
            # --- Top/side camera (via gym render) -------------------------- #
            try:
                pixels = self._env.render()  # (H, W, 3) uint8 RGB
                if pixels is not None:
                    bgr = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
                    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    img_msg                 = CompressedImage()
                    img_msg.header.stamp    = now
                    img_msg.header.frame_id = "xarm_camera"
                    img_msg.format          = "jpeg"
                    img_msg.data            = buf.tobytes()
                    self._pub_img.publish(img_msg)
            except Exception:
                pass  # rendering failure should never crash the state loop

            # --- Wrist camera (FREE cam, dedicated mujoco.Renderer) --------- #
            # gym_xarm's Lift class does not expose mujoco_renderer, so we
            # use a dedicated mujoco.Renderer initialised in __init__.
            # The lookat bug (mjCAMERA_TRACKING + mujoco 2.x) is avoided by
            # using mjCAMERA_FREE with an explicit lookat update each frame.
            if self._wrist_cam is not None and self._wrist_renderer is not None:
                try:
                    mj_data = self._env.unwrapped.data

                    # Snap lookat to the EEF site position each frame
                    if self._eef_site_id >= 0:
                        self._wrist_cam.lookat[:] = np.array(
                            mj_data.site_xpos[self._eef_site_id], dtype=np.float64
                        )

                    self._wrist_renderer.update_scene(mj_data, camera=self._wrist_cam)
                    wrist_pixels = self._wrist_renderer.render()  # (WRIST_H, WRIST_W, 3) RGB

                    bgr_w = cv2.cvtColor(wrist_pixels, cv2.COLOR_RGB2BGR)
                    # Rotate + flip to align wrist-cam axes with main (side) camera.
                    # 90° CW fixes X/Y swap; vertical flip corrects inverted Y.
                    bgr_w = cv2.rotate(bgr_w, cv2.ROTATE_90_CLOCKWISE)
                    bgr_w = cv2.flip(bgr_w, 0)   # 0 = flip vertically (Y axis)
                    _, wbuf = cv2.imencode(".jpg", bgr_w, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    w_msg                 = CompressedImage()
                    w_msg.header.stamp    = now
                    w_msg.header.frame_id = "wrist_camera"
                    w_msg.format          = "jpeg"
                    w_msg.data            = wbuf.tobytes()
                    self._pub_wrist_img.publish(w_msg)
                except Exception as _wrist_err:
                    # Log once, then suppress — rendering failure must not crash state loop
                    if not getattr(self, "_wrist_err_logged", False):
                        self.get_logger().warn(f"Wrist cam render error: {_wrist_err}")
                        self._wrist_err_logged = True

        # --- Full 32D state vector ----------------------------------------- #
        state = self._extract_state_32d()

        js              = JointState()
        js.header.stamp = now
        js.name         = STATE_NAMES
        js.position     = state.tolist()   # all 32 channels in "position" field
        self._pub_js.publish(js)

        # --- End-effector pose --------------------------------------------- #
        ee              = PoseStamped()
        ee.header.stamp = now
        ee.header.frame_id       = "world"
        ee.pose.position.x       = float(state[0])   # ee_x
        ee.pose.position.y       = float(state[1])   # ee_y
        ee.pose.position.z       = float(state[2])   # ee_z
        ee.pose.orientation.x    = float(state[3])   # qx
        ee.pose.orientation.y    = float(state[4])   # qy
        ee.pose.orientation.z    = float(state[5])   # qz
        ee.pose.orientation.w    = float(state[6])   # qw
        self._pub_ee.publish(ee)

        # --- F/T Wrench ---------------------------------------------------- #
        w_msg                    = WrenchStamped()
        w_msg.header.stamp       = now
        w_msg.header.frame_id    = "wrist_link"
        w_msg.wrench.force.x     = float(state[26])
        w_msg.wrench.force.y     = float(state[27])
        w_msg.wrench.force.z     = float(state[28])
        w_msg.wrench.torque.x    = float(state[29])
        w_msg.wrench.torque.y    = float(state[30])
        w_msg.wrench.torque.z    = float(state[31])
        self._pub_wrench.publish(w_msg)

    # ---------------------------------------------------------------------- #
    # Callbacks
    # ---------------------------------------------------------------------- #
    def _cb_joint_cmd(self, msg: JointState):
        """Accept EEF action targets (4D: dx, dy, dz, gripper)."""
        action_dim = self._env.action_space.shape[0]
        if len(msg.position) >= action_dim:
            with self._lock:
                self._action = np.array(
                    msg.position[:action_dim], dtype=np.float32
                )

    def _cb_reset(self, _msg):
        """Topic callback — kept for CLI use."""
        with self._lock:
            self._obs, _ = self._env.reset()
            self._action  = np.zeros(self._env.action_space.shape, dtype=np.float32)
            self._done    = False
        self.get_logger().info("Episode reset (topic).")

    def _srv_reset(self, _request, response):
        """Service handler — called by webserver Reset button via rosbridge."""
        with self._lock:
            self._obs, _ = self._env.reset()
            self._action  = np.zeros(self._env.action_space.shape, dtype=np.float32)
            self._done    = False
        self.get_logger().info("Episode reset (service).")
        return response

    def _srv_render(self, request, response):
        response.success = True
        response.message = "Render mode switching not implemented."
        return response

    def destroy_node(self):
        if self._wrist_renderer is not None:
            try:
                self._wrist_renderer.close()
            except Exception:
                pass
        self._env.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = XArmSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
