"""
teleop_bridge/teleop_node.py
============================
Maps MYBOTSHOP webserver joystick → Cartesian velocity → xARM joint command.

The MYBOTSHOP webserver publishes joystick inputs as sensor_msgs/Joy on /joy.
This node converts those axes to end-effector delta movements and sends them
to the simulation as JointState commands on /sim/joint_command.

Axis mapping (standard gamepad, configurable via ROS2 params):
  axes[0]  left stick X  → EEF delta-Y  (lateral)
  axes[1]  left stick Y  → EEF delta-X  (forward/back)
  axes[3]  right stick Y → EEF delta-Z  (up/down)
  buttons[0] A / cross   → gripper close
  buttons[1] B / circle  → gripper open

Published topics
----------------
/sim/joint_command   sensor_msgs/JointState   Cartesian delta as position targets

Subscribed topics
-----------------
/joy                 sensor_msgs/Joy           Webserver joystick input

Services
--------
/teleop/enable       std_srvs/SetBool          Enable/disable teleop (safety)
"""

import numpy as np
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState, Joy
from std_srvs.srv import SetBool


# Joystick sensitivity — fraction of the gym action range [-1, 1] per full deflection.
# gym_xarm applies action * 0.05 m internally, so SENSITIVITY=0.4 → 2 cm/step max.
# Increase if the arm feels sluggish, decrease if too twitchy.
SENSITIVITY   = 0.4

# gym_xarm XarmLift-v0 gripper convention (verified empirically):
#   action[3] >= 0  → fingers OPEN
#   action[3] <  0  → fingers CLOSE
# Note: this is the OPPOSITE of the intuitive naming, hence the explicit comment.
GRIPPER_OPEN  = -1.0   # negative → gym opens gripper
GRIPPER_CLOSE =  1.0   # positive → gym closes gripper


class TeleopNode(Node):

    def __init__(self):
        super().__init__("teleop_node")

        # Parameters
        self.declare_parameter("axis_left_x",  0)
        self.declare_parameter("axis_left_y",  1)
        self.declare_parameter("axis_right_y", 3)
        self.declare_parameter("btn_close",    0)
        self.declare_parameter("btn_open",     1)

        self._ax_lx  = self.get_parameter("axis_left_x").value
        self._ax_ly  = self.get_parameter("axis_left_y").value
        self._ax_ry  = self.get_parameter("axis_right_y").value
        self._btn_cl = self.get_parameter("btn_close").value
        self._btn_op = self.get_parameter("btn_open").value

        # Gripper state — start CLOSED to match gym_xarm default initial state
        self._gripper_closed = True
        self._enabled = True           # start enabled; watchdog or webserver can disable

        # Publisher
        self._pub = self.create_publisher(JointState, "/sim/joint_command", 10)

        # Subscriber
        self.create_subscription(Joy, "/joy", self._cb_joy, 10)

        # Service: /teleop/enable
        self.create_service(SetBool, "/teleop/enable", self._srv_enable)

        self.get_logger().info("TeleopNode ready — joystick active.")

    # ---------------------------------------------------------------------- #
    # Joy callback
    # ---------------------------------------------------------------------- #
    def _cb_joy(self, msg: Joy):
        if not self._enabled:
            return

        axes    = msg.axes
        buttons = msg.buttons

        def dead(v, thresh=0.05):
            return v if abs(v) > thresh else 0.0

        # Map joystick deflection directly to gym delta action [-1, 1].
        # gym_xarm applies action * 0.05 m per step, so SENSITIVITY=0.4
        # gives up to 0.4 * 0.05 = 2 cm/step at full deflection.
        # Zero joystick → zero action → arm holds position (no drift).
        ax = dead(axes[self._ax_ly]) * SENSITIVITY   # forward / back
        ay = dead(axes[self._ax_lx]) * SENSITIVITY   # lateral
        az = dead(axes[self._ax_ry]) * SENSITIVITY   # up / down

        # Gripper: toggle on button press (state persists between messages)
        if len(buttons) > self._btn_cl and buttons[self._btn_cl]:
            self._gripper_closed = True
        if len(buttons) > self._btn_op and buttons[self._btn_op]:
            self._gripper_closed = False

        # gym_xarm threshold: action[3] < 0 → close, ≥ 0 → open
        gripper_action = GRIPPER_CLOSE if self._gripper_closed else GRIPPER_OPEN

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name     = ["ax", "ay", "az", "gripper"]
        cmd.position = [float(ax), float(ay), float(az), float(gripper_action)]
        self._pub.publish(cmd)

    def _srv_enable(self, request, response):
        self._enabled = request.data
        state = "enabled" if self._enabled else "disabled"
        self.get_logger().info(f"Teleop {state} by service call.")
        response.success = True
        response.message = f"Teleop {state}."
        return response


def main(args=None):
    rclpy.init(args=args)
    node = TeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
