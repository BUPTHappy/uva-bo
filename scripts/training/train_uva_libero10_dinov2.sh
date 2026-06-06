#!/bin/sh

task_name='libero_10'

## Stage 1: train video generation with student tokenizer + DINOv2 teacher alignment
accelerate launch --num_processes=8 train.py \
    --config-dir=. \
    --config-name=uva_libero10_dinov2.yaml \
    model.policy.action_model_params.predict_action=False \
    model.policy.selected_training_mode=video_model \
    logging.project=uva_${task_name}_dinov2 \
    hydra.run.dir="checkpoints/libero10_dinov2_video"


# ## Stage 2: joint video + action fine-tuning
# accelerate launch --num_processes=8 train.py \
#     --config-dir=. \
#     --config-name=uva_libero10_dinov2.yaml \
#     model.policy.autoregressive_model_params.pretrained_model_path="checkpoints/libero10_dinov2_video/checkpoints/latest.ckpt" \
#     model.policy.action_model_params.predict_action=True \
#     logging.project=uva_${task_name}_dinov2 \
#     hydra.run.dir="checkpoints/libero10_dinov2_video_action"
