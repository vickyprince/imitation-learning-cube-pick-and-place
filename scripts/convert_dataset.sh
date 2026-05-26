#!/bin/bash
# Convert rosbag2 recordings to LeRobot format.
# Runs inside the sim_stack container so rosbag2_py and ROS2 are available.
docker compose -f docker/docker-compose.yml run --rm sim_stack \
  bash -c "source /opt/ros/humble/setup.bash && \
           source /ros2_ws/install/setup.bash && \
           ros2 run dataset_pipeline rosbag2_to_lerobot \
             --bags_dir /data/bags \
             --output_dir /data/datasets/xarm_lift_v2_fresh \
             --task_name xarm_lift"
