#!/usr/bin/env bash
# Train UVA on UMI multi-task data without VAE (RGB patch tokens).
# Validation logs: val_action_l2_distances, val_eef_traj_error, val_final_state_dist (see unified_video_action/eval/eval.py).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# For cloth_folding_0 only, add: task=umi_multi_cloth_folding
accelerate launch --num_processes="${NUM_GPUS:-8}" train.py \
    --config-name=uva_umi_multi_no_vae.yaml \
    task.dataset.dataset_root_dir="${DATASET_ROOT:?Set DATASET_ROOT to your processed UMI multi zarr root}" \
    hydra.run.dir="${HYDRA_RUN_DIR:-checkpoints/uva_umi_multi_no_vae}"
