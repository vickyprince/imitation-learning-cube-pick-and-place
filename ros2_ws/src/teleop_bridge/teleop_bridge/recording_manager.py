"""
teleop_bridge/recording_manager.py
===================================
Manages rosbag2 recording sessions, triggered by MYBOTSHOP webserver
action buttons via ROS2 services.

The webserver's Console panel has configurable action buttons that call
ROS2 services. We wire two buttons:
  "▶ Start Demo"  → calls /recording/start  (std_srvs/Trigger)
  "⏹ Stop Demo"   → calls /recording/stop   (std_srvs/Trigger)
  "🗑 Discard"     → calls /recording/discard (std_srvs/Trigger)

Each recording is saved to:
  /data/bags/episode_<TIMESTAMP>/

Published topics
----------------
/recording/status   std_msgs/String    "idle" | "recording" | "saved"
/recording/episode  std_msgs/Int32     current episode index
"""

import datetime
import os
import subprocess
import signal

import rclpy
from rclpy.node import Node

from std_msgs.msg import Int32, String
from std_srvs.srv import Trigger


class RecordingManager(Node):

    def __init__(self):
        super().__init__("recording_manager")

        self.declare_parameter("bag_output_dir", "/data/bags")
        self.declare_parameter("topics", [
            "/sim/camera/image_compressed",
            "/sim/joint_states",       # 32D state vector
            "/sim/ee_pose",            # EEF pose (PoseStamped)
            "/sim/wrench",             # F/T wrench (WrenchStamped)
            "/sim/joint_command",      # action commands sent to sim
        ])

        self._bag_dir   = self.get_parameter("bag_output_dir").value
        self._topics    = self.get_parameter("topics").value
        self._episode   = 0
        self._bag_proc  = None      # subprocess running ros2 bag record
        self._current_bag = None    # path of the bag being recorded
        self._status    = "idle"

        os.makedirs(self._bag_dir, exist_ok=True)

        # Publishers
        self._pub_status  = self.create_publisher(String, "/recording/status",  10)
        self._pub_episode = self.create_publisher(Int32,  "/recording/episode", 10)

        # Services  (wired to webserver action buttons in robot_webserver.yaml)
        self.create_service(Trigger, "/recording/start",   self._srv_start)
        self.create_service(Trigger, "/recording/stop",    self._srv_stop)
        self.create_service(Trigger, "/recording/discard", self._srv_discard)

        # Status heartbeat @ 1 Hz so webserver Telemetry panel stays current
        self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            f"RecordingManager ready. Saving bags to: {self._bag_dir}"
        )

    # ---------------------------------------------------------------------- #
    # Service handlers (called by webserver action buttons)
    # ---------------------------------------------------------------------- #
    def _srv_start(self, _req, resp):
        if self._status == "recording":
            resp.success = False
            resp.message = "Already recording."
            return resp

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_path = os.path.join(self._bag_dir, f"episode_{self._episode:04d}_{ts}")
        self._current_bag = bag_path

        topic_args = " ".join(self._topics)
        cmd = (
            f"source /opt/ros/humble/setup.bash && "
            f"ros2 bag record -o {bag_path} {topic_args}"
        )
        self._bag_proc = subprocess.Popen(
            cmd, shell=True, executable="/bin/bash",
            preexec_fn=os.setsid  # allows killing the whole process group
        )

        self._status = "recording"
        self.get_logger().info(f"Recording started → {bag_path}")
        resp.success = True
        resp.message = f"Recording episode {self._episode} to {bag_path}"
        return resp

    def _srv_stop(self, _req, resp):
        if self._status != "recording" or self._bag_proc is None:
            resp.success = False
            resp.message = "No active recording."
            return resp

        os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGINT)
        self._bag_proc.wait()
        self._bag_proc = None
        self._status = "saved"
        self._episode += 1

        self.get_logger().info(
            f"Recording stopped. Episode {self._episode - 1} saved to {self._current_bag}"
        )
        resp.success = True
        resp.message = f"Episode {self._episode - 1} saved."
        return resp

    def _srv_discard(self, _req, resp):
        """Stop and delete the current bag (bad demo)."""
        if self._status == "recording":
            self._srv_stop(None, type("R", (), {"success": True, "message": ""})())

        import shutil
        if self._current_bag and os.path.exists(self._current_bag):
            shutil.rmtree(self._current_bag)
            self._episode = max(0, self._episode - 1)  # rollback counter
            self.get_logger().info(f"Discarded bag: {self._current_bag}")

        self._status = "idle"
        resp.success = True
        resp.message = "Demo discarded."
        return resp

    def _publish_status(self):
        self._pub_status.publish(String(data=self._status))
        self._pub_episode.publish(Int32(data=self._episode))

    def destroy_node(self):
        if self._bag_proc is not None:
            os.killpg(os.getpgid(self._bag_proc.pid), signal.SIGINT)
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecordingManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
