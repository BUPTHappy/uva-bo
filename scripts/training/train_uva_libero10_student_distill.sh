#!/bin/sh
# Phase 1: vanilla KL-VAE -> small student encoder distillation on Libero10.
# Usage:
#   ./scripts/training/train_uva_libero10_student_distill.sh
#   NUM_PROCESSES=3 ./scripts/training/train_uva_libero10_student_distill.sh

NUM_PROCESSES="${NUM_PROCESSES:-3}"

accelerate launch --num_processes="${NUM_PROCESSES}" train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_student_distill.yaml \
  hydra.run.dir="checkpoints/uva_libero10_student_distill_small"
