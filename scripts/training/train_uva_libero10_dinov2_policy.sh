#!/bin/sh

# Continue from video checkpoint with DINOv2 teacher alignment (policy stage).
accelerate launch --num_processes=3 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_dinov2_policy.yaml \
  logging.project=uva-repa-dino \
  hydra.run.dir="checkpoints/uva_libero10_dino_try1"
