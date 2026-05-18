#!/bin/sh
# Phase 2: frozen distilled student + train MAR/action on Libero10.
# Usage:
#   STUDENT_CKPT=checkpoints/uva_libero10_student_distill_small/checkpoints/latest.ckpt \
#     ./scripts/training/train_uva_libero10_student_frozen.sh

if [ -z "${STUDENT_CKPT}" ]; then
  echo "Set STUDENT_CKPT to the phase-1 checkpoint (e.g. checkpoints/.../latest.ckpt)."
  exit 1
fi

NUM_PROCESSES="${NUM_PROCESSES:-3}"

accelerate launch --num_processes="${NUM_PROCESSES}" train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_student_frozen.yaml \
  "model.policy.student_tokenizer_pretrained_path=${STUDENT_CKPT}" \
  hydra.run.dir="checkpoints/uva_libero10_student_frozen_small"
