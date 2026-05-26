#!/bin/bash
#SBATCH --job-name=xarm_lerobot_act
#SBATCH --partition=gpu4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-04:00:00
#SBATCH --output=/work/vvicto2s/aic/logs/xarm_act_%j.out
#SBATCH --error=/work/vvicto2s/aic/logs/xarm_act_%j.err

set -eo pipefail

module load cuda

export CONDA_ROOT=/usr/local/fb03-fem/python/anaconda3
export PS1="${PS1:-}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate aic_lerobot_051

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_HOME=/work/vvicto2s/aic/.cache/torch
export HF_HOME=/work/vvicto2s/aic/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/work/vvicto2s/aic/.cache/huggingface/hub

export WORK_ROOT=/work/vvicto2s/aic
export DATASET_DIR=/work/vvicto2s/aic/data/datasets/xarm_lift_v2_fresh
export OUTPUT_DIR=/work/vvicto2s/aic/outputs_main/xarm_lerobot_act_${SLURM_JOB_ID}

mkdir -p /work/vvicto2s/aic/logs
mkdir -p "$HF_HOME"

echo "=== Job ${SLURM_JOB_ID} started: $(date) ==="
echo "=== GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader) ==="

cd "$WORK_ROOT"

# Step 1 — Determine which dataset directory to use for training.
#
# Three cases:
#   A) xarm_lift_v2_v30 already exists (conversion ran successfully before) → use it
#   B) xarm_lift_v2 is already v3.0 (info.json was patched in-place)         → use it directly
#   C) xarm_lift_v2 is still v2.1                                             → convert first
#
TRAIN_DATASET_DIR=""

# Determine the dataset version.
# The convert_dataset_v21_to_v30 script converts IN-PLACE:
#   - backs up original as {DATASET_DIR}_old
#   - rewrites {DATASET_DIR} as v3.0
# So after conversion, TRAIN_DATASET_DIR = DATASET_DIR.

SRC_VERSION=$(python3 -c "
import json
with open('$DATASET_DIR/meta/info.json') as f:
    print(json.load(f).get('codebase_version', 'unknown'))
" 2>/dev/null || echo "unknown")
echo "=== Source dataset version: $SRC_VERSION ==="

if [ "$SRC_VERSION" = "v3.0" ]; then
    echo "=== Dataset already v3.0 — skipping conversion ==="
    TRAIN_DATASET_DIR="$DATASET_DIR"
else
    echo "=== Dataset is $SRC_VERSION — running Parquet fix + v2.1→v3.0 conversion ==="

    echo "=== Fixing Parquet column format ==="
    python3 /work/vvicto2s/aic/scripts/fix_lerobot_parquet.py \
        --dataset_dir "$DATASET_DIR"

    echo "=== Generating episodes_stats.jsonl (required by v3.0 conversion) ==="
    python3 /work/vvicto2s/aic/scripts/fix_lerobot_episodes_stats.py \
        --dataset_dir "$DATASET_DIR"

    # convert_dataset_v21_to_v30 converts in-place and backs up to {dir}_old.
    # "local/..." is not a real HF repo → final hub-tag call returns 401.
    # We ignore that exit-code; the converted dataset is complete by then.
    echo "=== Converting dataset v2.1 → v3.0 (in-place) ==="
    python -m lerobot.scripts.convert_dataset_v21_to_v30 \
        --repo-id=local/xarm_lift_v2_fresh \
        --root="$DATASET_DIR" || true

    # Verify conversion succeeded (info.json must now say v3.0)
    NEW_VERSION=$(python3 -c "
import json
with open('$DATASET_DIR/meta/info.json') as f:
    print(json.load(f).get('codebase_version', 'unknown'))
" 2>/dev/null || echo "unknown")
    if [ "$NEW_VERSION" != "v3.0" ]; then
        echo "ERROR: conversion failed — info.json still says $NEW_VERSION" >&2
        exit 1
    fi
    echo "=== Conversion verified: dataset is now v3.0 ==="
    TRAIN_DATASET_DIR="$DATASET_DIR"
fi

# Step 1c — Fix episodes from_timestamp / to_timestamp (video frame index must stay in bounds)
echo "=== Fixing episodes from_timestamp in $TRAIN_DATASET_DIR ==="
python3 /work/vvicto2s/aic/scripts/fix_lerobot_episodes.py \
    --dataset_dir "$TRAIN_DATASET_DIR"

# Step 1d — Add global 'index' column to parquet files (required by LeRobot v3.0 dataset_reader)
echo "=== Adding global index column to parquet files ==="
python3 /work/vvicto2s/aic/scripts/fix_lerobot_add_index.py \
    --dataset_dir "$TRAIN_DATASET_DIR"

# Step 1e — Recompute stats.json (normalize_processor requires correct mean/std shapes)
echo "=== Recomputing dataset stats ==="
python3 /work/vvicto2s/aic/scripts/fix_lerobot_stats.py \
    --dataset_dir "$TRAIN_DATASET_DIR" \
    --image_sample_episodes 8

# Step 2 — Train ACT with LeRobot
echo "=== Starting lerobot-train ACT on $TRAIN_DATASET_DIR ==="
lerobot-train \
    --dataset.repo_id=local/xarm_lift_v2 \
    --dataset.root="$TRAIN_DATASET_DIR" \
    --policy.type=act \
    --policy.chunk_size=50 \
    --policy.n_action_steps=10 \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --num_workers=4 \
    --batch_size=8 \
    --steps=100000 \
    --save_freq=10000 \
    --tolerance_s=0.04 \
    --output_dir="$OUTPUT_DIR" \
    --job_name="xarm_lerobot_act_${SLURM_JOB_ID}"

echo "=== Job finished: $(date) ==="
echo "=== Checkpoint: $OUTPUT_DIR/checkpoints/last/pretrained_model ==="
