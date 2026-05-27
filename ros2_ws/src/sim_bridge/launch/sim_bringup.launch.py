"""
sim_bringup.launch.py
Launches the full simulation stack:
  - xarm_sim_node       (gym_xarm → ROS2 topics)
  - teleop_node         (joy → CartesianVel → sim action)
  - recording_manager   (rosbag2 lifecycle, triggered by webserver)
  - rosbridge_websocket (WebSocket bridge for MYBOTSHOP webserver on port 9090)
  - safety_watchdog     (joint/EEF/FT limit enforcement)
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ------------------------------------------------------------------ #
        # Simulation core
        # ------------------------------------------------------------------ #
        Node(
            package="sim_bridge",
            executable="xarm_sim_node",
            name="xarm_sim_node",
            output="screen",
            parameters=[{"use_sim_time": False}],
        ),

        # ------------------------------------------------------------------ #
        # Teleoperation
        # ------------------------------------------------------------------ #
        Node(
            package="teleop_bridge",
            executable="teleop_node",
            name="teleop_node",
            output="screen",
        ),

        # ------------------------------------------------------------------ #
        # Recording manager  (triggered by webserver ▶/⏹/🗑 buttons)
        # ------------------------------------------------------------------ #
        Node(
            package="teleop_bridge",
            executable="recording_manager",
            name="recording_manager",
            output="screen",
            parameters=[{
                "bag_output_dir": "/data/bags",
                "topics": [
                    "/sim/camera/image_compressed",
                    "/sim/joint_states",
                    "/sim/ee_pose",
                    "/sim/wrench",
                    "/sim/joint_command",
                ],
            }],
        ),

        # ------------------------------------------------------------------ #
        # Training manager — runs train_act.py on demand from the browser UI.
        # Publishes progress to /training/status, /training/log, /training/progress.
        # ------------------------------------------------------------------ #
        Node(
            package="teleop_bridge",
            executable="training_manager",
            name="training_manager",
            output="screen",
        ),

        # ------------------------------------------------------------------ #
        # ACT Policy inference node (ROS2 Lifecycle)
        # Loads trained checkpoint and publishes /sim/joint_command.
        # checkpoint_path can be:
        #   - A LeRobot pretrained directory  → full ACT transformer (ResNet18 + CVAE)
        #   - A local .pt file                → lightweight MLP (browser-trained)
        # Activate via: ros2 lifecycle set /policy_node configure
        #               ros2 lifecycle set /policy_node activate
        # ------------------------------------------------------------------ #
        Node(
            package="policy_lifecycle_manager",
            executable="policy_node",
            name="policy_node",
            output="screen",
            parameters=[{
                "checkpoint_path": "/data/lerobot_checkpoints/xarm_act_142952",
                "inference_fps":   30.0,
            }],
        ),

        # ------------------------------------------------------------------ #
        # Safety watchdog  (EEF workspace + joint + F/T limits)
        # ------------------------------------------------------------------ #
        Node(
            package="safety_watchdog",
            executable="watchdog_node",
            name="safety_watchdog",
            output="screen",
            parameters=[{
                # Sim-appropriate thresholds (see watchdog_node.py comments)
                "confidence_threshold": 0.02,
            }],
        ),

        # ------------------------------------------------------------------ #
        # rosbridge WebSocket server
        # Bridges the MYBOTSHOP webserver (browser) ↔ ROS2 topics/services.
        # Port 9090 must be exposed in the Docker run command:
        #   docker run -p 9090:9090 -p 9000:9000 ...
        # ------------------------------------------------------------------ #
        Node(
            package="rosbridge_server",
            executable="rosbridge_websocket",
            name="rosbridge_websocket",
            output="screen",
            parameters=[{
                "port": 9090,
                # Allow unregistered publishers (browser → /joy without prior
                # advertise handshake from the Python side)
                "unregister_timeout": 10.0,
                # Larger fragment size for compressed image messages
                "fragment_timeout": 600,
                "delay_between_messages": 0.0,
                "max_message_size": 10000000,
                "send_action_goals_in_new_thread": False,
            }],
        ),

    ])
