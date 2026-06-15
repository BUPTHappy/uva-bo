#!/bin/sh

# Load video-stage weights from warm_start_checkpoint, train policy from epoch 0.
accelerate launch --num_processes=3 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_dinov2_policy.yaml \
  training.warm_start_checkpoint="checkpoints/libero10_video.ckpt" \
  training.resume=false \
  logging.project=uva-repa-dino \
  hydra.run.dir="checkpoints/uva_libero10_dino_latent_try1"
