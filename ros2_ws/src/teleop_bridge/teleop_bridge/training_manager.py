"""
teleop_bridge/training_manager.py
==================================
ROS2 node that runs the full training pipeline from the browser UI:
  1. [optional] Convert rosbag2 bags → LeRobot dataset  (auto_convert=True)
  2. Run train_act.py and stream progress back via rosbridge topics.

Training parameters are received as a JSON string on /training/config
before calling the /training/start service.

Published topics
----------------
/training/status    std_msgs/String   "idle"|"converting"|"running"|"done"|"error"
/training/log       std_msgs/String   one line per message (epoch, loss, etc.)
/training/progress  std_msgs/Float32  0.0 – 100.0 percent complete

Services
--------
/training/start     std_srvs/Trigger  start pipeline with last received config
/training/stop      std_srvs/Trigger  kill current subprocess

Default config (overridden by /training/config topic):
  bags_dir     = /data/bags
  dataset_dir  = /data/datasets/xarm_lift_v1
  output_dir   = /data/checkpoints/xarm_lift_v1
  task_name    = xarm_lift
  auto_convert = true   (run rosbag2_to_lerobot before training)
  epochs       = 2
  batch_size   = 8
  lr           = 1e-4
"""

import json
import os
import re
import subprocess
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger


TRAIN_SCRIPT   = "/training/train_act.py"
CONVERT_SCRIPT = "/ros2_ws/install/dataset_pipeline/lib/dataset_pipeline/rosbag2_to_lerobot"

DEFAULT_CONFIG = {
    "bags_dir":     "/data/bags",
    "dataset_dir":  "/data/datasets/xarm_lift_v2_fresh",
    "output_dir":   "/data/checkpoints/xarm_lift_v2",
    "task_name":    "xarm_lift",
    "auto_convert": True,
    "epochs":       2,
    "batch_size":   8,
    "lr":           1e-4,
}


class TrainingManager(Node):

    def __init__(self):
        super().__init__("training_manager")

        self._config  = dict(DEFAULT_CONFIG)
        self._proc    = None
        self._thread  = None
        self._status  = "idle"

        # Publishers
        self._pub_status   = self.create_publisher(String,  "/training/status",   1)
        self._pub_log      = self.create_publisher(String,  "/training/log",      10)
        self._pub_progress = self.create_publisher(Float32, "/training/progress", 10)

        # Config subscriber — browser publishes JSON before calling /training/start
        self.create_subscription(String, "/training/config", self._cb_config, 1)

        # Services
        self.create_service(Trigger, "/training/start", self._srv_start)
        self.create_service(Trigger, "/training/stop",  self._srv_stop)

        # Heartbeat: re-publish status every 2 s so browser always knows state
        self.create_timer(2.0, self._heartbeat)

        self.get_logger().info("TrainingManager ready — awaiting /training/start.")

    # ------------------------------------------------------------------ #
    # Config
    # ------------------------------------------------------------------ #
    def _cb_config(self, msg: String):
        try:
            cfg = json.loads(msg.data)
            self._config.update(cfg)
            self.get_logger().info(f"Training config updated: {self._config}")
        except Exception as e:
            self.get_logger().warn(f"Bad training config JSON: {e}")

    # ------------------------------------------------------------------ #
    # Services
    # ------------------------------------------------------------------ #
    def _srv_start(self, _req, resp):
        if self._status in ("converting", "running"):
            resp.success = False
            resp.message = f"Pipeline already active (status: {self._status})."
            return resp

        if not os.path.exists(TRAIN_SCRIPT):
            resp.success = False
            resp.message = f"Training script not found: {TRAIN_SCRIPT}"
            return resp

        auto_convert = self._config.get("auto_convert", DEFAULT_CONFIG["auto_convert"])
        bags_dir     = self._config.get("bags_dir",    DEFAULT_CONFIG["bags_dir"])
        dataset_dir  = self._config.get("dataset_dir", DEFAULT_CONFIG["dataset_dir"])

        if auto_convert:
            # Bags dir must exist (we'll warn inside if empty)
            if not os.path.isdir(bags_dir):
                resp.success = False
                resp.message = f"Bags directory not found: {bags_dir}"
                return resp
        else:
            # Without conversion the dataset must already exist
            if not os.path.isdir(dataset_dir):
                resp.success = False
                resp.message = (
                    f"Dataset directory not found: {dataset_dir}. "
                    "Enable auto_convert=true or run conversion manually."
                )
                return resp

        self._thread = threading.Thread(target=self._run_training, daemon=True)
        self._thread.start()

        mode = "convert + train" if auto_convert else "train only"
        resp.success = True
        resp.message = (
            f"Pipeline started ({mode}) — "
            f"{self._config['epochs']} epochs."
        )
        return resp

    def _srv_stop(self, _req, resp):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
            self._set_status("idle")
            self._log("Training stopped by user.")
            resp.success = True
            resp.message = "Training stopped."
        else:
            resp.success = False
            resp.message = "No training process running."
        return resp

    # ------------------------------------------------------------------ #
    # Full pipeline: [convert →] train
    # ------------------------------------------------------------------ #
    def _run_training(self):
        cfg = self._config

        # ---- Step 1: Convert bags → LeRobot dataset (optional) ----------
        auto_convert = cfg.get("auto_convert", DEFAULT_CONFIG["auto_convert"])
        if auto_convert:
            bags_dir    = str(cfg.get("bags_dir",    DEFAULT_CONFIG["bags_dir"]))
            dataset_dir = str(cfg.get("dataset_dir", DEFAULT_CONFIG["dataset_dir"]))
            task_name   = str(cfg.get("task_name",   DEFAULT_CONFIG["task_name"]))

            # Count bags so user knows what's being converted
            import glob as _glob
            n_bags = len(_glob.glob(os.path.join(bags_dir, "episode_*")))
            if n_bags == 0:
                self._log(f"WARNING: No bags found in {bags_dir} — skipping conversion.")
            else:
                self._log(
                    f"[1/2] Converting {n_bags} bag(s) → LeRobot dataset at {dataset_dir} ..."
                )
                self._set_status("converting")
                self._pub_progress.publish(Float32(data=0.0))

                conv_cmd = [
                    CONVERT_SCRIPT,
                    "--bags_dir",   bags_dir,
                    "--output_dir", dataset_dir,
                    "--task_name",  task_name,
                ]
                self._log(f"Running: {' '.join(conv_cmd)}")

                try:
                    self._proc = subprocess.Popen(
                        conv_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                    )
                    for line in self._proc.stdout:
                        line = line.rstrip()
                        if line:
                            self._log(line)
                    self._proc.wait()
                    rc = self._proc.returncode
                    if rc != 0:
                        self._set_status("error")
                        self._log(f"Conversion failed (exit code {rc}). Training aborted.")
                        return
                    self._log(f"[1/2] Conversion done — {dataset_dir} is up to date.")
                except Exception as e:
                    self._set_status("error")
                    self._log(f"Conversion error: {e}. Training aborted.")
                    return

        # ---- Step 2: Train -----------------------------------------------
        dataset_dir = str(cfg.get("dataset_dir", DEFAULT_CONFIG["dataset_dir"]))
        if not os.path.isdir(dataset_dir):
            self._set_status("error")
            self._log(f"Dataset directory not found: {dataset_dir}")
            return

        step_label = "[2/2]" if auto_convert else "[1/1]"
        train_cmd = [
            "python3", TRAIN_SCRIPT,
            "--dataset_dir", dataset_dir,
            "--output_dir",  str(cfg.get("output_dir",  DEFAULT_CONFIG["output_dir"])),
            "--epochs",      str(int(cfg.get("epochs",     DEFAULT_CONFIG["epochs"]))),
            "--batch_size",  str(int(cfg.get("batch_size", DEFAULT_CONFIG["batch_size"]))),
            "--lr",          str(float(cfg.get("lr",       DEFAULT_CONFIG["lr"]))),
        ]

        self._log(f"{step_label} Starting training: {' '.join(train_cmd)}")
        self._set_status("running")
        self._pub_progress.publish(Float32(data=0.0))

        total_epochs = int(cfg.get("epochs", DEFAULT_CONFIG["epochs"]))

        try:
            self._proc = subprocess.Popen(
                train_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in self._proc.stdout:
                line = line.rstrip()
                if not line:
                    continue

                self._log(line)

                # Parse "Epoch   2/100 | train=0.03124 | eval=0.02891 | lr=..."
                m = re.match(r"Epoch\s+(\d+)/(\d+)", line)
                if m:
                    epoch = int(m.group(1))
                    total = int(m.group(2))
                    pct   = round(epoch / total * 100.0, 1)
                    self._pub_progress.publish(Float32(data=pct))

            self._proc.wait()
            rc = self._proc.returncode

            if rc == 0:
                self._set_status("done")
                self._log(
                    "Training complete. "
                    f"Checkpoint saved to: {cfg.get('output_dir')}/act_xarm_lift.pt"
                )
                self._pub_progress.publish(Float32(data=100.0))
            else:
                self._set_status("error")
                self._log(f"Training process exited with code {rc}.")

        except Exception as e:
            self._set_status("error")
            self._log(f"Training error: {e}")

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _set_status(self, status: str):
        self._status = status
        self._pub_status.publish(String(data=status))

    def _log(self, msg: str):
        self.get_logger().info(msg)
        self._pub_log.publish(String(data=msg))

    def _heartbeat(self):
        self._pub_status.publish(String(data=self._status))


def main(args=None):
    rclpy.init(args=args)
    node = TrainingManager()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node._proc and node._proc.poll() is None:
            node._proc.terminate()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
