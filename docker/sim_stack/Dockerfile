# =============================================================================
# MYBOTSHOP Imitation Learning Demo — Simulation Stack
# Base: ROS2 Humble on Ubuntu 22.04 Jammy (ARM64 native for Apple M1)
# =============================================================================
FROM ros:humble-ros-base-jammy

# --------------------------------------------------------------------------- #
# System deps
# --------------------------------------------------------------------------- #
ENV DEBIAN_FRONTEND=noninteractive
# MuJoCo headless rendering — OSMesa software renderer (no display needed in Docker)
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    # ROS2 packages
    ros-humble-rclpy \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-std-srvs \
    ros-humble-lifecycle-msgs \
    ros-humble-joy \
    ros-humble-cv-bridge \
    ros-humble-image-transport \
    ros-humble-compressed-image-transport \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-py \
    ros-humble-rosbag2-storage-default-plugins \
    ros-humble-ros2bag \
    # rosbridge: WebSocket bridge for MYBOTSHOP webserver ↔ ROS2
    ros-humble-rosbridge-suite \
    # Utilities
    wget curl git ffmpeg \
    # OpenGL: libosmesa6 provides the headless OSMesa renderer used by MuJoCo.
    # libgl1-mesa-glx provides GLX for display rendering (not used in Docker
    # but needed by some Python OpenGL bindings at import time).
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    libosmesa6 libgl1-mesa-dev \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------------------------------- #
# Python: MuJoCo + gym_xarm + CV + dataset tools
# MuJoCo 2.3.x ships pre-built wheels for linux-aarch64 (M1 ARM64 in Docker)
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Heavy ML deps — separate layer so Docker cache skips re-downloading
# torch/torchvision/lerobot (~1.5 GB) when only source code changes.
# --------------------------------------------------------------------------- #
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    lerobot

# Pre-cache ResNet18 ImageNet weights so first inference doesn't stall.
RUN python3 -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1'); print('ResNet18 weights cached.')"

# --------------------------------------------------------------------------- #
# Simulation / ROS2 Python deps
# --------------------------------------------------------------------------- #
RUN pip3 install --no-cache-dir \
    "mujoco>=2.3.7,<3.0.0" \
    "gymnasium==0.29.1" \
    "gymnasium-robotics==1.2.4" \
    "gym-xarm==0.1.1" \
    opencv-python-headless \
    # NumPy MUST be <2 — ROS2 Humble's cv_bridge C++ extension was compiled
    # against NumPy 1.x ABI. NumPy 2.x breaks _ARRAY_API at import time.
    "numpy<2" \
    pyarrow \
    pandas \
    datasets \
    huggingface_hub \
    safetensors \
    einops \
    matplotlib \
    scipy

# --------------------------------------------------------------------------- #
# Copy and build ROS2 workspace
# CACHE_BUST: increment to force Docker to re-copy source and rerun colcon.
# --------------------------------------------------------------------------- #
ARG CACHE_BUST=1
WORKDIR /ros2_ws
COPY ros2_ws/src ./src

RUN /bin/bash -c "source /opt/ros/humble/setup.bash && \
    colcon build \
      --symlink-install \
      --packages-select sim_bridge teleop_bridge dataset_pipeline \
                        policy_lifecycle_manager safety_watchdog"

# --------------------------------------------------------------------------- #
# Copy training scripts and webserver config
# --------------------------------------------------------------------------- #
COPY training /training
COPY webserver_config /webserver_config

# --------------------------------------------------------------------------- #
# Entrypoint: source ROS2 + workspace, then run whatever CMD is passed
# --------------------------------------------------------------------------- #
COPY docker/sim_stack/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "sim_bridge", "sim_bringup.launch.py"]
