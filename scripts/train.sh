#!/bin/bash
# Train ACT policy on collected dataset (CPU, inside Docker).
# For faster training on Apple Silicon, run train_act.py directly on Mac
# (uses MPS GPU): python3 training/train_act.py --dataset_dir ...
docker compose -f docker/docker-compose.yml run --rm sim_stack \
  python3 /training/train_act.py \
    --dataset_dir /data/datasets/xarm_lift_v2_fresh \
    --output_dir  /data/checkpoints/xarm_lift_v2 \
    --epochs 100 \
    --batch_size 8
