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
6. (Optional) Download V-JEPA 2 checkpoints for JEPA teacher experiments
7. Train the video stage, then warm-start policy training with your chosen teacher

---

## 1. Clone the repository

```bash
git clone <your-repo-url> uva-bo
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

Optional: uncomment calls in `download.py` to also fetch MAR-L, MAR-H, or cache DINOv2 via timm.

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
cd -
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

Training uses [Hydra](https://hydra.cc/) configs under `unified_video_action/config/` and [Accelerate](https://huggingface.co/docs/accelerate) for multi-GPU launch.

### 6.1 Stage 0 — Video model (required warm-start source)

Train the autoregressive video model on Libero10. Output is written to `checkpoints/libero10_video/`.

```bash
sh scripts/training/train_uva_libero10.sh
```

After training, point policy stages at your best checkpoint, e.g. copy or symlink:

```bash
cp checkpoints/libero10_video/checkpoints/epoch=XXXX-test_mean_score=X.XXX.ckpt \
   checkpoints/libero10_video.ckpt
```

Policy configs default to `training.warm_start_checkpoint: checkpoints/libero10_video.ckpt`.

### 6.2 Warm-start vs resume

| Option | Meaning |
|--------|---------|
| `training.warm_start_checkpoint=...` | Load model weights only; **epoch counter starts at 0**; optimizer state is reset |
| `training.resume=true` | Full resume from a prior run directory (epoch, optimizer, etc.) |

Policy training with a frozen teacher typically uses **warm-start**, not resume:

```bash
training.warm_start_checkpoint="checkpoints/libero10_video.ckpt" \
training.resume=false
```

### 6.3 Stage 1 — Policy + VAE teacher (baseline)

Train action policy with student tokenizer aligned to the frozen VAE teacher:

```bash
accelerate launch --num_processes=8 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10.yaml \
  model.policy.use_student_tokenizer=true \
  model.policy.teacher_type=vae \
  model.policy.align_params.enable=true \
  model.policy.align_params.coeff=0.05 \
  model.policy.align_params.align_on=token_feat \
  model.policy.action_model_params.predict_action=true \
  model.policy.selected_training_mode=policy_model \
  dataloader.batch_size=16 \
  training.gradient_accumulate_every=4 \
  training.warm_start_checkpoint="checkpoints/libero10_video.ckpt" \
  training.resume=false \
  hydra.run.dir="checkpoints/uva_libero10_vae_policy"
```

Adjust `align_on` to `latent` if you prefer latent-space alignment.

### 6.4 Stage 2 — Policy + DINOv2 teacher

```bash
sh scripts/training/train_uva_libero10_dinov2_policy.sh
```

Or explicitly:

```bash
accelerate launch --num_processes=6 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_dinov2_policy.yaml \
  training.warm_start_checkpoint="checkpoints/libero10_video.ckpt" \
  training.resume=false \
  logging.project=uva-repa-dino \
  hydra.run.dir="checkpoints/uva_libero10_dino_latent_try1"
```

Config: `unified_video_action/config/uva_libero10_dinov2_policy.yaml`  
(`teacher_type: dinov2`, `align_on: latent`)

Optional video-only stage with DINOv2 alignment (before policy):

```bash
sh scripts/training/train_uva_libero10_dinov2.sh
```

### 6.5 Stage 3 — Policy + JEPA teacher

**Run once per machine before multi-GPU training:**

```bash
sh scripts/setup_jepa_teacher.sh
```

Then:

```bash
sh scripts/training/train_uva_libero10_jepa_policy.sh
```

Or explicitly:

```bash
accelerate launch --num_processes=3 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_jepa_policy.yaml \
  training.warm_start_checkpoint="checkpoints/libero10_video.ckpt" \
  training.resume=false \
  logging.project=uva-repa-jepa \
  hydra.run.dir="checkpoints/uva_libero10_jepa_latent_try1"
```

Config: `unified_video_action/config/uva_libero10_jepa_policy.yaml`  
(`teacher_type: jepa`, `align_on: latent`, default model `vjepa2_vit_large`)

**JEPA ViT-H example** (after downloading `vith.pt`):

```bash
accelerate launch --num_processes=3 train.py \
  --config-dir=unified_video_action/config \
  --config-name=uva_libero10_jepa_policy.yaml \
  model.policy.jepa_teacher_params.model_name=vjepa2_vit_huge \
  model.policy.jepa_teacher_params.checkpoint_path=pretrained_models/jepa/vith.pt \
  training.warm_start_checkpoint="checkpoints/libero10_video.ckpt" \
  training.resume=false \
  hydra.run.dir="checkpoints/uva_libero10_jepa_vith_try1"
```

Change `--num_processes` to match available GPUs on your node.

### 6.6 Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python eval_sim.py \
  --checkpoint checkpoints/uva_libero10_jepa_latent_try1/checkpoints/latest.ckpt \
  --output_dir checkpoints/uva_libero10_jepa_latent_try1/eval
```

See `scripts/eval/eval_sim.sh` for additional examples.

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

## 9. Architecture summary

```
Observations ──► StudentLatentTokenizer ──► latent [16×16×16] ──► MAR (video + action)
                         │                         ▲
                         └── token_feat [384]        │
                                                     │
              Frozen teacher (VAE / DINOv2 / JEPA) ──┘ alignment loss
```

- **MAR** and **VAE decode** still use the original 16-dim VAE latent space.
- **Teachers** are frozen; only the student tokenizer (and policy heads) receive alignment gradients.
- **JEPA** alignment is on student **latent** by default; a learned projector maps JEPA tokens (1024-d) to 16-d before the loss.

Key files:

- `unified_video_action/policy/unified_video_action_policy.py` — teacher switching, alignment
- `unified_video_action/model/common/jepa_teacher.py` — V-JEPA 2 wrapper
- `unified_video_action/model/common/dinov2_teacher.py` — DINOv2 wrapper
- `unified_video_action/workspace/train_unified_video_action_workspace.py` — `warm_start_checkpoint` logic

---

## 10. Troubleshooting

| Issue | Fix |
|-------|-----|
| `JEPA checkpoint not found` | Run `sh scripts/setup_jepa_teacher.sh` on that machine |
| `Missing horizon in shape_meta` | Use configs on this branch (`task/libero10.yaml` includes `horizon`) |
| DINOv2 hub errors on Python 3.9 | DINOv2 uses `timm` loader (`loader: timm` in config) |
| MuJoCo / GL errors | Check `MUJOCO_PY_MUJOCO_PATH`, `LD_LIBRARY_PATH`, and `libGL.so` symlink |
| Hydra struct / missing keys | Policy configs use inline `model.policy.*` overrides; keep `_self_` last in `defaults` |

---

## License

See the upstream UVA repository license. Third-party weights (VAE, MAR, DINOv2, V-JEPA 2) are subject to their respective terms.
