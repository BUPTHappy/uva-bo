#!/bin/sh

# One-time per machine (recommended before first multi-GPU run):
#   sh scripts/setup_jepa_teacher.sh

# Warm-start from video ckpt, align student latent to JEPA latent (policy stage).
accelerate launch --num_processes=3 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_jepa_policy.yaml \
  training.warm_start_checkpoint="checkpoints/libero10_video.ckpt" \
  training.resume=false \
  logging.project=uva-repa-jepa \
  hydra.run.dir="checkpoints/uva_libero10_jepa_latent_try1"
