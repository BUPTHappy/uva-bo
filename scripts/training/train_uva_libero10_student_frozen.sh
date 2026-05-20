#!/bin/sh
# Phase 2: frozen distilled student + train MAR/action on Libero10.
# Usage:
#   ./scripts/training/train_uva_libero10_student_frozen.sh
#   NUM_PROCESSES=3 ./scripts/training/train_uva_libero10_student_frozen.sh
#   STUDENT_CKPT=checkpoints/other.ckpt ./scripts/training/train_uva_libero10_student_frozen.sh

STUDENT_CKPT="${STUDENT_CKPT:-checkpoints/libero10_distill.ckpt}"

if [ ! -f "${STUDENT_CKPT}" ]; then
  echo "Student checkpoint not found: ${STUDENT_CKPT}"
  exit 1
fi

echo "Using frozen student checkpoint: ${STUDENT_CKPT}"

_detect_num_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
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

echo "Launching frozen-student policy training with NUM_PROCESSES=${NUM_PROCESSES}"

accelerate launch --num_processes="${NUM_PROCESSES}" train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_student_frozen.yaml \
  "model.policy.student_tokenizer_pretrained_path=${STUDENT_CKPT}" \
  hydra.run.dir="checkpoints/uva_libero10_student_frozen_phase2"
