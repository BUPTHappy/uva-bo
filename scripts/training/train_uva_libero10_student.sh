#!/bin/sh
# Train libero10 policy with a student tokenizer preset.
# Usage:
#   SIZE=small ./scripts/training/train_uva_libero10_student.sh
#   SIZE=equal ./scripts/training/train_uva_libero10_student.sh
#   SIZE=large ./scripts/training/train_uva_libero10_student.sh

# Default aligns with historic manual CLI sizing (~Enc+quant); use SIZE=small for compact.
SIZE="${SIZE:-equal}"

accelerate launch --num_processes=3 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_student.yaml \
  "model/student_tokenizer=${SIZE}" \
  hydra.run.dir="checkpoints/uva_libero10_student_${SIZE}"
