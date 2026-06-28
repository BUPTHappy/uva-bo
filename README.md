# Unified Video Action — Student Tokenizer + Teacher Alignment (Libero10)

This branch (`uva-repa-dinov2`) extends [UVA](https://github.com/...) with a **student latent tokenizer** and **frozen teacher alignment** on the Libero10 benchmark. The MAR backbone still operates on 16-dimensional VAE-style latents; teachers (VAE, DINOv2, or V-JEPA 2) are used only for representation alignment during training.

**Supported teachers**

| Teacher | Config key | Alignment target | Notes |
|---------|------------|------------------|-------|
| VAE (KL-16) | `teacher_type: vae` | `token_feat` or `latent` | Default baseline; VAE also decodes video for MAR |
| DINOv2 ViT-S/14 | `teacher_type: dinov2` | `token_feat` or `latent` | Loaded via `timm` |
| V-JEPA 2 | `teacher_type: jepa` | `latent` (recommended) | Requires local checkpoint on each machine |

---

## Quick start checklist

1. Clone this repo and check out `uva-repa-dinov2`
2. Download base pretrained weights (`python unified_video_action/utils/download.py`)
3. Create the conda environment (`repa_environment.yml`)
4. Install MuJoCo + LIBERO dependencies
5. Download Libero10 data and clone LIBERO
6. Download `checkpoints/libero10_video.ckpt` and start policy training

---

## 1. Clone the repository

```bash
git clone https://github.com/BUPTHappy/uva-bo.git
cd uva-bo
git checkout uva-repa-dinov2
```

Download VAE and MAR-B weights used by the default UVA config:

```bash
python unified_video_action/utils/download.py
```

This saves:

- `pretrained_models/vae/kl16.ckpt`
- `pretrained_models/mar/mar_base/checkpoint-last.pth`

---

## 2. Environment setup

Install mamba (recommended) and create the environment:

```bash
conda install mamba -c conda-forge
mamba env create -f repa_environment.yml
eval "$(mamba shell hook --shell bash)"
source ~/.bashrc
conda activate repa
```

The environment name is **`repa`** (Python 3.9).

---

## 3. MuJoCo and LIBERO (simulation rollouts)

Libero10 rollouts require `mujoco-py` and the LIBERO package.

### 3.1 OpenGL / MuJoCo dependencies

```bash
conda install -y -c conda-forge \
  glfw glew patchelf mesalib libglvnd libstdcxx-ng \
  xorg-libx11 xorg-libxrandr xorg-libxinerama xorg-libxi xorg-libxcursor
```

Install MuJoCo 2.1 (adjust `$HOME` if needed):

```bash
mkdir -p ~/.mujoco && cd ~/.mujoco
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
tar -xzf mujoco210-linux-x86_64.tar.gz
```

Set environment variables (add these to `~/.bashrc` for persistence):

```bash
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CFLAGS="-I$CONDA_PREFIX/include"
export LDFLAGS="-L$CONDA_PREFIX/lib"

# Some conda builds only ship libGL.so.1 — create an unversioned symlink if missing
[ -f "$CONDA_PREFIX/lib/libGL.so" ] || ln -s $CONDA_PREFIX/lib/libGL.so.1 $CONDA_PREFIX/lib/libGL.so

export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210   # <-- change to your path
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MUJOCO_PY_MUJOCO_PATH/bin
```

Verify:

```bash
python -c "import mujoco_py; print('mujoco-py OK')"
```

If numba-related errors appear during rollouts:

```bash
conda install -y -c conda-forge numba==0.56.4 llvmlite==0.39.1
rm -rf ~/.numba
export NUMBA_DISABLE_JIT=1
```

### 3.2 LIBERO package

Clone LIBERO next to (or anywhere on `PYTHONPATH`) this repo:

```bash
cd ..   # parent of uva-bo
git clone https://github.com/Lifelong-robot-learning/LIBERO.git
```

Follow LIBERO's own install instructions if import errors occur.

---

## 4. Download Libero10 dataset

From the repo root (`uva-bo/`):

```bash
mkdir -p data && cd data
gdown https://drive.google.com/uc?id=1_6Kc7e-s30MblbX8YjpxSofe9ZRPk3xv
gdown https://drive.google.com/uc?id=1cPU2RVAvtukyapcWly8zP1y-dlOEF2ko
unzip libero_10.zip
cd ..
```

The config expects data at `data/libero_10` (see `unified_video_action/config/task/libero10.yaml`).

---

## 5. V-JEPA 2 teacher checkpoints

JEPA teachers load **local** weight files. On multi-GPU or multi-node runs, the checkpoint must exist on **every machine** (torch.hub code is shared, weights are not).

### 5.1 One-time setup (ViT-L, default)

```bash
sh scripts/setup_jepa_teacher.sh
```

This downloads `pretrained_models/jepa/vitl.pt` and runs a smoke test.

### 5.2 Other supported sizes

| Model | `model_name` | Checkpoint file | Download URL |
|-------|--------------|-----------------|--------------|
| ViT-L | `vjepa2_vit_large` | `pretrained_models/jepa/vitl.pt` | https://dl.fbaipublicfiles.com/vjepa2/vitl.pt |
| ViT-H | `vjepa2_vit_huge` | `pretrained_models/jepa/vith.pt` | https://dl.fbaipublicfiles.com/vjepa2/vith.pt |
| ViT-g | `vjepa2_vit_giant` | `pretrained_models/jepa/vitg.pt` | https://dl.fbaipublicfiles.com/vjepa2/vitg.pt |
| ViT-g-384 | `vjepa2_vit_giant_384` | `pretrained_models/jepa/vitg-384.pt` | https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt |

Example for ViT-H:

```bash
mkdir -p pretrained_models/jepa
wget https://dl.fbaipublicfiles.com/vjepa2/vith.pt -O pretrained_models/jepa/vith.pt
```

### 5.3 Verify a checkpoint

```bash
python - <<'PY'
import os, torch
from unified_video_action.model.common.jepa_teacher import JEPATeacher

ckpt = "pretrained_models/jepa/vitl.pt"
assert os.path.isfile(ckpt), f"Missing {ckpt}"
print("File size (GB):", os.path.getsize(ckpt) / 1e9)

teacher = JEPATeacher(
    model_name="vjepa2_vit_large",
    img_size=256,
    checkpoint_path=ckpt,
    loader="torch_hub",
)
x = torch.rand(1, 3, 2, 256, 256)
y = teacher.extract_tokens(x)
print("JEPA teacher OK, tokens:", tuple(y.shape))  # e.g. (1, 2, 256, 1024)
PY
```

To switch JEPA size, update both `model.policy.jepa_teacher_params.model_name` and `checkpoint_path` in the config (or via Hydra overrides).

---

## 6. Training

Policy training warm-starts from a pretrained Libero10 video checkpoint. Download it first (no need to train the video model yourself):

```bash
mkdir -p checkpoints
gdown 1XxWHwUDehiEh1eGscyVBafLU_3V3nqxw -O checkpoints/libero10_video.ckpt
```

Then pick a teacher and run the matching script. Adjust `--num_processes` inside each script to match your GPU count.

**VAE teacher**

```bash
accelerate launch --num_processes=8 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10.yaml \
  model.policy.use_student_tokenizer=true \
  model.policy.teacher_type=vae \
  model.policy.align_params.enable=true \
  model.policy.align_params.coeff=0.05 \
  model.policy.action_model_params.predict_action=true \
  model.policy.selected_training_mode=policy_model \
  dataloader.batch_size=16 \
  training.gradient_accumulate_every=4 \
  training.warm_start_checkpoint=checkpoints/libero10_video.ckpt \
  training.resume=false \
  hydra.run.dir=checkpoints/uva_libero10_vae_policy
```

**DINOv2 teacher**

```bash
sh scripts/training/train_uva_libero10_dinov2_policy.sh
```

**JEPA teacher** (run `setup_jepa_teacher.sh` once per machine first — see Section 5)

```bash
sh scripts/training/train_uva_libero10_jepa_policy.sh
```

By default this uses **ViT-L** (`vjepa2_vit_large` + `pretrained_models/jepa/vitl.pt`), as set in `unified_video_action/config/uva_libero10_jepa_policy.yaml`. Downloading other sizes does not change which teacher runs — only `model_name` and `checkpoint_path` in the config matter.

To use a different size, override both on the command line, e.g. ViT-H:

```bash
accelerate launch --num_processes=3 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_jepa_policy.yaml \
  model.policy.jepa_teacher_params.model_name=vjepa2_vit_huge \
  model.policy.jepa_teacher_params.checkpoint_path=pretrained_models/jepa/vith.pt \
  hydra.run.dir=checkpoints/uva_libero10_jepa_vith_try1
```

| Size | `model_name` | `checkpoint_path` |
|------|--------------|-------------------|
| ViT-L (default) | `vjepa2_vit_large` | `pretrained_models/jepa/vitl.pt` |
| ViT-H | `vjepa2_vit_huge` | `pretrained_models/jepa/vith.pt` |
| ViT-g | `vjepa2_vit_giant` | `pretrained_models/jepa/vitg.pt` |
| ViT-g-384 | `vjepa2_vit_giant_384` | `pretrained_models/jepa/vitg-384.pt` |

---

## 7. Weights & Biases (W&B)

Training logs to W&B when `logging.mode: online` (default in policy configs). On first run you will be prompted for an API key:

1. Create an account at https://wandb.ai
2. Copy your API key from https://wandb.ai/authorize
3. Paste it when prompted (or set `WANDB_API_KEY` in the environment)

Override project/name via Hydra, e.g. `logging.project=uva-repa-jepa`.

To disable W&B: `logging.mode=offline` or `logging.mode=disabled`.

---

## 8. Tips for long-running jobs

### Background sessions with GNU Screen

If you disconnect from SSH, use a terminal multiplexer so training keeps running:

```bash
wget https://ftp.gnu.org/gnu/screen/screen-4.9.1.tar.gz
tar -xzf screen-4.9.1.tar.gz
cd screen-4.9.1
./configure --prefix=$HOME/.local
make -j4 && make install
echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

screen -S uva_train
# run your training command inside the screen session
# detach: Ctrl+A then D
# reattach: screen -r uva_train
```

Alternatives: `tmux`, `nohup`, or your cluster's job scheduler (Slurm, etc.).

### Multi-node / shared filesystem

- JEPA checkpoints must be present on **each compute node**, not only the login node.
- torch.hub downloads V-JEPA **architecture code** once; weights are loaded from your local `checkpoint_path`.
- If multiple processes race on first hub download, the code uses a file lock — still prefer running `setup_jepa_teacher.sh` once before launching many GPUs.

---
