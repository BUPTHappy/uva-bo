#!/bin/sh
set -e

# One-time setup on each training machine.
mkdir -p pretrained_models/jepa

if [ ! -f pretrained_models/jepa/vitl.pt ]; then
  echo "Downloading V-JEPA2 ViT-L checkpoint..."
  wget https://dl.fbaipublicfiles.com/vjepa2/vitl.pt -O pretrained_models/jepa/vitl.pt
else
  echo "Checkpoint already exists: pretrained_models/jepa/vitl.pt"
fi

python - <<'PY'
import torch
from unified_video_action.model.common.jepa_teacher import JEPATeacher

teacher = JEPATeacher(
    model_name="vjepa2_vit_large",
    img_size=256,
    checkpoint_path="pretrained_models/jepa/vitl.pt",
    loader="torch_hub",
)
x = torch.rand(1, 3, 2, 256, 256)
y = teacher.extract_tokens(x)
print("JEPA teacher OK, tokens:", tuple(y.shape))
PY
