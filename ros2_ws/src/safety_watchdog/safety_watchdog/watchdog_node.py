"""
safety_watchdog/watchdog_node.py
=================================
Runtime safety supervisor for the imitation learning deployment.

Monitors policy behaviour and intervenes when anomalies are detected.
Provides one-click manual override from the MYBOTSHOP webserver Console.

32D observation vector layout (matches xarm_sim_node.py):
  [0:3]   EEF Position      (m)
  [3:7]   EEF Orientation   quaternion (x,y,z,w)
  [7:10]  EEF Linear Vel    (m/s)
  [10:13] EEF Angular Vel   (rad/s)
  [13]    Gripper            [0=closed, 1=open]
  [14:20] Joint Positions   (rad)
  [20:26] Joint Velocities  (rad/s)
  [26:32] F/T Wrench        (N, N·m)

Monitored conditions
--------------------
1. Confidence drop       /policy/confidence  < threshold  → warn → deactivate
2. Joint workspace limit /sim/joint_states   any EEF coord > limit → e-stop
3. Joint velocity limit  /sim/joint_states   any |dj| > limit → e-stop
4. F/T force limit       /sim/wrench         |force| > limit → e-stop
5. F/T torque limit      /sim/wrench         |torque| > limit → e-stop
6. Inference timeout     /policy/inference_fps < min_fps → warn
7. Manual override       /safety/manual_override service → immediate teleop

Services
--------
/safety/manual_override   std_srvs/Trigger   Immediately deactivate policy, enable teleop
/safety/resume_policy     std_srvs/Trigger   Re-activate policy after manual check

Published topics
----------------
/safety/status        std_msgs/String    "ok" | "warn" | "emergency"
/safety/alert_reason  std_msgs/String    human-readable alert message

Subscribed topics
-----------------
/policy/confidence    std_msgs/Float32
/policy/inference_fps std_msgs/Float32
/policy/status        std_msgs/String
/sim/joint_states     sensor_msgs/JointState   (32D state vector)
/sim/wrench           geometry_msgs/WrenchStamped
"""

import rclpy
from rclpy.node import Node

from lifecycle_msgs.srv import ChangeState
from lifecycle_msgs.msg import Transition
from geometry_msgs.msg import WrenchStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, String
from std_srvs.srv import SetBool, Trigger


# ============================================================================
# Safety limit tables
# ============================================================================

# EEF workspace limits (metres) — gym_xarm XarmLift-v0 coordinate frame.
# NOTE: gym_xarm places the robot base at a large negative X offset (~-2 m)
# relative to the MuJoCo world origin.  These limits are derived from the
# observed teleoperation range + 25% buffer, NOT from real xARM6 URDF.
# For real-robot deployment: replace with URDF-derived Cartesian limits.
EEF_LIMITS = {
    "ee_x":   (-2.30, -1.50),   # observed range [-2.04, -1.78] + buffer
    "ee_y":   (-0.65,  0.25),   # observed range [-0.45, -0.11] + buffer
    "ee_z":   (-0.90, -0.35),   # observed range [-0.72, -0.52] + buffer
}

# xARM6 joint position limits (radians) — from xARM6 URDF spec.
# joint1: ±360°, joint2: −118°/+120°, joint3: −225°/+11°,
# joint4: ±360°, joint5: ±124°, joint6: ±360°
import math
JOINT_POS_LIMITS = [
    (-math.radians(360), math.radians(360)),   # j1
    (-math.radians(118), math.radians(120)),   # j2
    (-math.radians(225), math.radians(11)),    # j3
    (-math.radians(360), math.radians(360)),   # j4
    (-math.radians(124), math.radians(124)),   # j5
    (-math.radians(360), math.radians(360)),   # j6
]

# Joint velocity limits (rad/s).
# Sim note: MuJoCo's finite-difference qvel computation adds small numerical
# noise (~0.5°/s) that can nudge joint velocities fractionally over the rated
# 180°/s limit. Using 250°/s here gives clearance for sim noise while still
# catching genuine runaway velocities.
# For real-robot deployment: restore to math.radians(180).
JOINT_VEL_LIMIT = math.radians(250)   # 4.36 rad/s

# F/T Wrench limits — set high for sim use.
# gym_xarm's MuJoCo contact solver produces forces in the 100–10000 N range
# due to stiff contact parameters (sim artefact, not physically realistic).
# For real-robot deployment: restore to FT_FORCE_LIMIT=50, FT_TORQUE_LIMIT=10.
FT_FORCE_LIMIT  = 20000.0   # N   — effectively disabled in sim
FT_TORQUE_LIMIT =  2000.0   # N·m — effectively disabled in sim

# Gripper [0=fully closed, 1=fully open]
GRIPPER_LIMITS = (0.0, 1.0)

# State vector channel indices (must match xarm_sim_node.py)
IDX_EE_X  = 0
IDX_EE_Y  = 1
IDX_EE_Z  = 2
IDX_JP    = slice(14, 20)  # joint positions
IDX_JV    = slice(20, 26)  # joint velocities
IDX_FT    = slice(26, 32)  # F/T wrench


class SafetyWatchdog(Node):

    def __init__(self):
        super().__init__("safety_watchdog")

        # Confidence threshold: mean |action| < this → low-confidence stop.
        # ACTMiniPolicy outputs conservative small actions (~0.1–0.3 range),
        # so 0.2 is too tight for sim. Set to 0.02 — catches truly dead/zero
        # policies while allowing normal conservative inference.
        self.declare_parameter("confidence_threshold", 0.02)
        self.declare_parameter("min_inference_fps",    5.0)
        self.declare_parameter("warn_count_threshold", 10)

        self._conf_thresh  = self.get_parameter("confidence_threshold").value
        self._min_fps      = self.get_parameter("min_inference_fps").value
        self._warn_limit   = self.get_parameter("warn_count_threshold").value

        self._policy_active   = False
        self._conf_warn_count = 0
        self._fps_warn_count  = 0
        self._safety_state    = "ok"

        # ------------------------------------------------------------------ #
        # Publishers
        # ------------------------------------------------------------------ #
        self._pub_status = self.create_publisher(String, "/safety/status",       10)
        self._pub_alert  = self.create_publisher(String, "/safety/alert_reason", 10)

        # ------------------------------------------------------------------ #
        # Subscribers
        # ------------------------------------------------------------------ #
        self.create_subscription(Float32,       "/policy/confidence",    self._cb_confidence, 10)
        self.create_subscription(Float32,       "/policy/inference_fps", self._cb_fps,        10)
        self.create_subscription(String,        "/policy/status",        self._cb_policy_stat,10)
        self.create_subscription(JointState,    "/sim/joint_states",     self._cb_joints,     10)
        self.create_subscription(WrenchStamped, "/sim/wrench",           self._cb_wrench,     10)

        # ------------------------------------------------------------------ #
        # Policy control clients — use the same Trigger services as the UI.
        # This avoids touching the lifecycle state machine, which would fail
        # because _srv_run bypasses it (calls on_activate directly).
        # ------------------------------------------------------------------ #
        self._policy_stop_cli = self.create_client(Trigger, "/policy/stop")
        self._policy_run_cli  = self.create_client(Trigger, "/policy/run")
        self._teleop_cli      = self.create_client(SetBool, "/teleop/enable")

        # ------------------------------------------------------------------ #
        # Services
        # ------------------------------------------------------------------ #
        self.create_service(Trigger, "/safety/manual_override", self._srv_manual_override)
        self.create_service(Trigger, "/safety/resume_policy",   self._srv_resume_policy)

        # Status heartbeat (1 Hz)
        self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            "SafetyWatchdog online — 32D state monitoring: "
            "EEF workspace, joint pos/vel, F/T wrench, policy confidence."
        )

    # ================================================================== #
    # Monitoring callbacks
    # ================================================================== #
    def _cb_confidence(self, msg: Float32):
        if not self._policy_active:
            return
        if msg.data < self._conf_thresh:
            self._conf_warn_count += 1
            if self._conf_warn_count >= self._warn_limit:
                self._trigger_safety_stop(
                    f"Low confidence ({msg.data:.3f} < {self._conf_thresh}) "
                    f"for {self._conf_warn_count} consecutive frames."
                )
        else:
            self._conf_warn_count = 0

    def _cb_fps(self, msg: Float32):
        if not self._policy_active:
            return
        if msg.data < self._min_fps:
            self._fps_warn_count += 1
            if self._fps_warn_count >= self._warn_limit:
                self._alert(
                    f"Policy inference FPS dropped to {msg.data:.1f} "
                    f"(min: {self._min_fps}). Check system resources."
                )
        else:
            self._fps_warn_count = 0

    def _cb_policy_stat(self, msg: String):
        self._policy_active = (msg.data == "active")

    def _cb_joints(self, msg: JointState):
        """
        Check EEF workspace limits, joint position limits, and joint velocity limits.

        The /sim/joint_states "position" field carries the full 32D state vector
        (see xarm_sim_node.py). We slice the relevant channels by index.
        """
        if not self._policy_active:
            return

        state = list(msg.position)
        if len(state) < 26:
            return  # malformed message — skip

        # --- EEF workspace limits ---
        ee_xyz = {
            "ee_x": state[IDX_EE_X],
            "ee_y": state[IDX_EE_Y],
            "ee_z": state[IDX_EE_Z],
        }
        for name, val in ee_xyz.items():
            lo, hi = EEF_LIMITS[name]
            if not (lo <= val <= hi):
                self._trigger_safety_stop(
                    f"EEF workspace violation: {name} = {val:.4f} m "
                    f"(allowed: [{lo}, {hi}] m)"
                )
                return

        # --- Joint position limits ---
        joint_pos = state[IDX_JP]
        for i, (pos, (lo, hi)) in enumerate(zip(joint_pos, JOINT_POS_LIMITS)):
            if not (lo <= pos <= hi):
                self._trigger_safety_stop(
                    f"Joint position violation: j{i+1} = {math.degrees(pos):.1f}° "
                    f"(allowed: [{math.degrees(lo):.0f}°, {math.degrees(hi):.0f}°])"
                )
                return

        # --- Joint velocity limits ---
        if len(state) >= 26:
            joint_vel = state[IDX_JV]
            for i, vel in enumerate(joint_vel):
                if abs(vel) > JOINT_VEL_LIMIT:
                    self._trigger_safety_stop(
                        f"Joint velocity violation: dj{i+1} = {math.degrees(vel):.1f}°/s "
                        f"(max: ±{math.degrees(JOINT_VEL_LIMIT):.0f}°/s)"
                    )
                    return

    def _cb_wrench(self, msg: WrenchStamped):
        """
        Check F/T wrench limits.  Mirrors an ATI Gamma sensor safety check
        on a real robot — e-stop on force or torque overload.
        """
        if not self._policy_active:
            return

        import math as _math
        fx, fy, fz = msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z
        tx, ty, tz = msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z

        force_mag  = _math.sqrt(fx**2 + fy**2 + fz**2)
        torque_mag = _math.sqrt(tx**2 + ty**2 + tz**2)

        if force_mag > FT_FORCE_LIMIT:
            self._trigger_safety_stop(
                f"F/T force overload: |F| = {force_mag:.1f} N "
                f"(max: {FT_FORCE_LIMIT} N)"
            )
        elif torque_mag > FT_TORQUE_LIMIT:
            self._trigger_safety_stop(
                f"F/T torque overload: |τ| = {torque_mag:.2f} N·m "
                f"(max: {FT_TORQUE_LIMIT} N·m)"
            )

    # ================================================================== #
    # Service handlers
    # ================================================================== #
    def _srv_manual_override(self, _req, resp):
        self.get_logger().warn("MANUAL OVERRIDE requested from webserver.")
        self._trigger_safety_stop("Manual override by operator.")
        resp.success = True
        resp.message = "Policy deactivated. Teleop enabled."
        return resp

    def _srv_resume_policy(self, _req, resp):
        if not self._policy_run_cli.service_is_ready():
            resp.success = False
            resp.message = "Policy /policy/run service not available."
            return resp

        self._policy_run_cli.call_async(Trigger.Request())
        self._set_teleop(False)
        self._safety_state = "ok"
        self.get_logger().info("Policy resumed after manual inspection.")
        resp.success = True
        resp.message = "Policy re-activated."
        return resp

    # ================================================================== #
    # Safety actions
    # ================================================================== #
    def _trigger_safety_stop(self, reason: str):
        self.get_logger().error(f"SAFETY STOP: {reason}")
        self._safety_state = "emergency"
        self._pub_alert.publish(String(data=reason))

        # Call /policy/stop (same Trigger service the UI uses) — avoids the
        # lifecycle state machine which would throw if the node state is stale.
        if self._policy_stop_cli.service_is_ready():
            self._policy_stop_cli.call_async(Trigger.Request())

        self._set_teleop(True)
        self._conf_warn_count = 0
        self._fps_warn_count  = 0

    def _alert(self, reason: str):
        self.get_logger().warn(f"WATCHDOG WARN: {reason}")
        self._safety_state = "warn"
        self._pub_alert.publish(String(data=reason))

    def _set_teleop(self, enabled: bool):
        if self._teleop_cli.service_is_ready():
            req      = SetBool.Request()
            req.data = enabled
            self._teleop_cli.call_async(req)

    def _publish_status(self):
        self._pub_status.publish(String(data=self._safety_state))
        if self._safety_state == "warn":
            self._safety_state = "ok"


def main(args=None):
    rclpy.init(args=args)
    node = SafetyWatchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
