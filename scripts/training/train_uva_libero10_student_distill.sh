#!/bin/sh
# Phase 1: vanilla KL-VAE -> small student encoder distillation on Libero10.
# Usage:
#   ./scripts/training/train_uva_libero10_student_distill.sh
#   NUM_PROCESSES=2 ./scripts/training/train_uva_libero10_student_distill.sh
#   CUDA_VISIBLE_DEVICES=0 ./scripts/training/train_uva_libero10_student_distill.sh

_detect_num_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    # e.g. "0" -> 1 GPU, "0,1,2" -> 3 GPUs
    echo "${CUDA_VISIBLE_DEVICES}" | awk -F, '{print NF}'
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | wc -l | tr -d ' '
    return
  fi
  echo 1
}

if [ -z "${NUM_PROCESSES}" ]; then
  NUM_PROCESSES="$(_detect_num_gpus)"
fi
if [ "${NUM_PROCESSES}" -lt 1 ] 2>/dev/null; then
  NUM_PROCESSES=1
fi

echo "Launching distill training with NUM_PROCESSES=${NUM_PROCESSES}"

accelerate launch --num_processes="${NUM_PROCESSES}" train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_student_distill.yaml \
  hydra.run.dir="checkpoints/uva_libero10_student_distill_small"
