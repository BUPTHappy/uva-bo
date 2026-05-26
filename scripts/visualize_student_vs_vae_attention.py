#!/usr/bin/env python3
"""
Visualize where a trained student tokenizer and the original VAE encoder respond
on LIBERO image observations.

The script produces two complementary maps:
  1. latent_norm: per-patch latent L2 energy, closest to a token/patch activation map.
  2. input_grad: input-gradient saliency of latent energy, closer to "what pixels
     affect the encoder output".

Example:
  python scripts/visualize_student_vs_vae_attention.py \
      --checkpoint checkpoints/libero10_student/checkpoints/latest.ckpt \
      --dataset-path data/libero_10 \
      --output-dir outputs/student_vs_vae_vis \
      --num-samples 4 --num-frames 4 --device cuda:0
"""

import argparse
import glob
import os
import pathlib
import random
import re
import sys
from typing import List, Tuple

ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import dill
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont


def torch_load_checkpoint(path: str):
    try:
        return torch.load(
            open(path, "rb"),
            map_location="cpu",
            pickle_module=dill,
            weights_only=False,
        )
    except TypeError:
        return torch.load(open(path, "rb"), map_location="cpu", pickle_module=dill)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trained student tokenizer and frozen VAE encoder heatmaps."
    )
    parser.add_argument("--checkpoint", "-c", required=True, help="Path to .ckpt")
    parser.add_argument(
        "--dataset-path",
        default=None,
        help="Folder containing LIBERO .hdf5 files. Defaults to cfg.task.dataset.dataset_path.",
    )
    parser.add_argument("--output-dir", "-o", default="outputs/student_vs_vae_attention")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--obs-key", default="agentview_rgb")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument(
        "--sample-indices",
        type=int,
        nargs="*",
        default=None,
        help="Optional global demo indices to visualize. Overrides random sampling.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--vae-reduction",
        choices=("mode", "sample"),
        default="mode",
        help="Use deterministic posterior mean or stochastic sample for VAE maps.",
    )
    parser.add_argument(
        "--ema",
        choices=("auto", "yes", "no"),
        default="auto",
        help="Use EMA policy from checkpoint when available.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Heatmap overlay opacity.",
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Also save raw heatmaps as .npz for later analysis.",
    )
    return parser.parse_args()


def load_policy_from_checkpoint(ckpt_path: str, device: torch.device):
    import hydra

    from unified_video_action.workspace.base_workspace import BaseWorkspace
    from omegaconf import open_dict

    payload = torch_load_checkpoint(ckpt_path)
    cfg = payload["cfg"]
    patch_missing_obs_horizon(cfg)
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=".")
    workspace.load_payload(payload, exclude_keys=None, include_keys=None, strict=False)

    use_ema = False
    if hasattr(workspace, "ema_model") and workspace.ema_model is not None:
        use_ema = True

    policy = workspace.ema_model if use_ema else workspace.model
    policy = policy.eval().to(device)
    for p in policy.parameters():
        p.requires_grad_(False)

    if not bool(getattr(policy, "use_student_tokenizer", False)):
        raise RuntimeError(
            "Checkpoint policy has use_student_tokenizer=False. "
            "Please pass a checkpoint trained with the student tokenizer."
        )
    if getattr(policy, "student_tokenizer", None) is None:
        raise RuntimeError("Checkpoint policy does not contain policy.student_tokenizer.")
    return cfg, policy, use_ema


def patch_missing_obs_horizon(cfg) -> None:
    """Old checkpoints may not store per-RGB obs horizon expected by this branch."""
    default_obs_horizon = None
    if "task" in cfg and "dataset" in cfg.task and "n_obs_steps" in cfg.task.dataset:
        default_obs_horizon = int(cfg.task.dataset.n_obs_steps)
    if default_obs_horizon is None:
        default_obs_horizon = 16

    default_action_horizon = None
    if "task" in cfg and "dataset" in cfg.task and "horizon" in cfg.task.dataset:
        default_action_horizon = int(cfg.task.dataset.horizon)
    if default_action_horizon is None:
        default_action_horizon = 32

    from omegaconf import open_dict

    def _patch_shape_meta(shape_meta) -> None:
        if shape_meta is None:
            return
        if "obs" in shape_meta:
            for _, attr in shape_meta.obs.items():
                if attr.get("type", "low_dim") == "rgb" and "horizon" not in attr:
                    attr.horizon = default_obs_horizon
        if "action" in shape_meta and "horizon" not in shape_meta.action:
            shape_meta.action.horizon = default_action_horizon

    with open_dict(cfg):
        if "task" in cfg and "shape_meta" in cfg.task:
            _patch_shape_meta(cfg.task.shape_meta)
        if (
            "model" in cfg
            and "policy" in cfg.model
            and "shape_meta" in cfg.model.policy
        ):
            _patch_shape_meta(cfg.model.policy.shape_meta)


def list_demos(dataset_path: str) -> List[Tuple[str, str]]:
    import h5py

    hdf5_paths = sorted(glob.glob(os.path.join(dataset_path, "*.hdf5")))
    if len(hdf5_paths) == 0:
        raise FileNotFoundError(f"No .hdf5 files found under: {dataset_path}")

    demos: List[Tuple[str, str]] = []
    for hdf5_path in hdf5_paths:
        with h5py.File(hdf5_path, "r") as f:
            for key in sorted(f["data"].keys(), key=lambda x: int(x.split("_")[-1])):
                demos.append((hdf5_path, key))
    if len(demos) == 0:
        raise RuntimeError(f"No demos found under: {dataset_path}")
    return demos


def load_demo_frames(
    hdf5_path: str,
    demo_key: str,
    obs_key: str,
    num_frames: int,
    image_size: int,
) -> Tuple[torch.Tensor, List[int]]:
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        arr = f["data"][demo_key]["obs"][obs_key]
        total = int(arr.shape[0])
        if total <= 0:
            raise RuntimeError(f"Empty demo: {hdf5_path}:{demo_key}")
        frame_ids = np.linspace(0, total - 1, num=min(num_frames, total), dtype=np.int64)
        frames = arr[frame_ids].astype(np.float32) / 255.0

    # Match LiberoReplayImageDataset.__getitem__: HWC -> CHW, rotate 180, flip width.
    frames = np.moveaxis(frames, -1, 1)
    frames = np.rot90(frames, k=2, axes=(2, 3)).copy()
    frames = np.flip(frames, axis=3).copy()

    x = torch.from_numpy(frames).unsqueeze(0)  # [1, T, C, H, W], [0, 1]
    bsz, timesteps, channels, height, width = x.shape
    if height != image_size or width != image_size:
        x = F.interpolate(
            x.reshape(bsz * timesteps, channels, height, width),
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        ).reshape(bsz, timesteps, channels, image_size, image_size)
    return x, frame_ids.tolist()


def normalize_image_for_encoder(images_btchw: torch.Tensor) -> torch.Tensor:
    return rearrange(images_btchw * 2.0 - 1.0, "b t c h w -> b c t h w").contiguous()


def encode_student(policy, x_bcthw: torch.Tensor) -> torch.Tensor:
    latent, _ = policy.student_tokenizer(x_bcthw)
    return latent


def encode_vae(policy, x_bcthw: torch.Tensor, reduction: str) -> torch.Tensor:
    bsz, channels, timesteps, height, width = x_bcthw.shape
    x_flat = rearrange(x_bcthw, "b c t h w -> (b t) c h w")
    posterior = policy.vae_model.encode(x_flat)
    if reduction == "sample":
        z = posterior.sample()
    else:
        z = posterior.mode()
    z = z * 0.2325
    return rearrange(z, "(b t) c h w -> b t c h w", b=bsz, t=timesteps)


def heatmap_from_latent(latent_btchw: torch.Tensor, out_size: int) -> torch.Tensor:
    heat = latent_btchw.float().pow(2).mean(dim=2).sqrt()  # [B, T, h, w]
    heat = F.interpolate(
        heat.flatten(0, 1).unsqueeze(1),
        size=(out_size, out_size),
        mode="nearest",
    ).squeeze(1)
    return heat.reshape(latent_btchw.shape[0], latent_btchw.shape[1], out_size, out_size)


def heatmap_from_input_grad(policy, x_bcthw: torch.Tensor, model_name: str, reduction: str) -> torch.Tensor:
    x = x_bcthw.detach().clone().requires_grad_(True)
    if model_name == "student":
        latent = encode_student(policy, x)
    elif model_name == "vae":
        latent = encode_vae(policy, x, reduction)
    else:
        raise ValueError(model_name)
    score = latent.float().pow(2).mean()
    grad = torch.autograd.grad(score, x, retain_graph=False, create_graph=False)[0]
    heat = grad.detach().abs().mean(dim=1)  # [B, T, H, W]
    return heat


def minmax_per_frame(heat_bthw: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    heat = heat_bthw.detach().float().cpu()
    flat = heat.flatten(2)
    lo = flat.amin(dim=-1).view(heat.shape[0], heat.shape[1], 1, 1)
    hi = flat.amax(dim=-1).view(heat.shape[0], heat.shape[1], 1, 1)
    return ((heat - lo) / (hi - lo + eps)).clamp(0.0, 1.0)


def red_colormap(heat_hw: np.ndarray) -> np.ndarray:
    heat = np.clip(heat_hw, 0.0, 1.0)[..., None]
    low = np.array([255, 255, 255], dtype=np.float32)
    high = np.array([165, 0, 0], dtype=np.float32)
    return (low * (1.0 - heat) + high * heat).astype(np.uint8)


def overlay_heat(image_chw: torch.Tensor, heat_hw: torch.Tensor, alpha: float) -> Image.Image:
    image = (image_chw.detach().cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    heat_rgb = red_colormap(heat_hw.detach().cpu().numpy())
    blended = (image.astype(np.float32) * (1.0 - alpha) + heat_rgb.astype(np.float32) * alpha)
    return Image.fromarray(blended.clip(0, 255).astype(np.uint8))


def draw_label(tile: Image.Image, text: str, fill=(255, 255, 255), outline=(0, 0, 0)) -> Image.Image:
    out = tile.copy()
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    pad = 6
    bbox = draw.textbbox((0, 0), text, font=font)
    rect = (0, 0, bbox[2] + 2 * pad, bbox[3] + 2 * pad)
    draw.rectangle(rect, fill=fill, outline=outline, width=2)
    draw.text((pad, pad), text, fill=(0, 0, 0), font=font)
    return out


def make_grid(
    images_btchw: torch.Tensor,
    vae_heat_bthw: torch.Tensor,
    student_heat_bthw: torch.Tensor,
    alpha: float,
    title: str,
    frame_ids: List[int],
) -> Image.Image:
    images = images_btchw[0].cpu()
    vae_heat = minmax_per_frame(vae_heat_bthw)[0]
    student_heat = minmax_per_frame(student_heat_bthw)[0]
    diff_heat = (student_heat - vae_heat).abs()
    diff_heat = minmax_per_frame(diff_heat.unsqueeze(0))[0]

    rows: List[List[Image.Image]] = []
    labels = ["original", "VAE encoder", "student encoder", "|student - VAE|"]
    for row_idx, label in enumerate(labels):
        row_tiles: List[Image.Image] = []
        for t in range(images.shape[0]):
            if row_idx == 0:
                tile_arr = (images[t].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                tile = Image.fromarray(tile_arr)
            elif row_idx == 1:
                tile = overlay_heat(images[t], vae_heat[t], alpha)
            elif row_idx == 2:
                tile = overlay_heat(images[t], student_heat[t], alpha)
            else:
                tile = overlay_heat(images[t], diff_heat[t], alpha)
            if t == 0:
                tile = draw_label(tile, label)
            else:
                tile = draw_label(tile, f"frame {frame_ids[t]}", fill=(255, 255, 255), outline=(180, 180, 180))
            row_tiles.append(tile)
        rows.append(row_tiles)

    tile_w, tile_h = rows[0][0].size
    gutter = 4
    header_h = 34
    canvas = Image.new(
        "RGB",
        (tile_w * len(rows[0]) + gutter * (len(rows[0]) - 1), header_h + tile_h * len(rows) + gutter * (len(rows) - 1)),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
    except OSError:
        font = ImageFont.load_default()
    draw.text((6, 6), title, fill=(0, 0, 0), font=font)
    y = header_h
    for row in rows:
        x = 0
        for tile in row:
            canvas.paste(tile, (x, y))
            x += tile_w + gutter
        y += tile_h + gutter
    return canvas


def choose_indices(total: int, args: argparse.Namespace) -> List[int]:
    if args.sample_indices is not None and len(args.sample_indices) > 0:
        return [idx for idx in args.sample_indices if 0 <= idx < total]
    rng = random.Random(args.seed)
    count = min(args.num_samples, total)
    return rng.sample(range(total), count)


def safe_filename(text: str, max_len: int = 150) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text[:max_len].strip("._")


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    cfg, policy, checkpoint_has_ema = load_policy_from_checkpoint(args.checkpoint, device)
    if args.ema == "no" and checkpoint_has_ema:
        import hydra

        from unified_video_action.workspace.base_workspace import BaseWorkspace
        from omegaconf import open_dict

        # Reloading without EMA is only needed when explicitly requested.
        payload = torch_load_checkpoint(args.checkpoint)
        patch_missing_obs_horizon(payload["cfg"])
        cls = hydra.utils.get_class(payload["cfg"].model._target_)
        workspace: BaseWorkspace = cls(payload["cfg"], output_dir=".")
        workspace.load_payload(payload, exclude_keys=None, include_keys=None, strict=False)
        policy = workspace.model.eval().to(device)
        for p in policy.parameters():
            p.requires_grad_(False)
    elif args.ema == "yes" and not checkpoint_has_ema:
        raise RuntimeError("Requested --ema yes, but this checkpoint has no ema_model.")

    dataset_path = args.dataset_path
    if dataset_path is None:
        dataset_path = str(cfg.task.dataset.dataset_path)
    demos = list_demos(dataset_path)
    selected = choose_indices(len(demos), args)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Dataset: {dataset_path} ({len(demos)} demos)")
    print(f"Using EMA policy: {checkpoint_has_ema and args.ema != 'no'}")
    print(f"Writing visualizations to: {output_dir}")

    for out_idx, demo_idx in enumerate(selected):
        hdf5_path, demo_key = demos[demo_idx]
        images_btchw, frame_ids = load_demo_frames(
            hdf5_path=hdf5_path,
            demo_key=demo_key,
            obs_key=args.obs_key,
            num_frames=args.num_frames,
            image_size=args.image_size,
        )
        images_btchw_device = images_btchw.to(device)
        x = normalize_image_for_encoder(images_btchw_device)

        with torch.no_grad():
            vae_latent = encode_vae(policy, x, args.vae_reduction)
            student_latent = encode_student(policy, x)
            vae_latent_heat = heatmap_from_latent(vae_latent, args.image_size)
            student_latent_heat = heatmap_from_latent(student_latent, args.image_size)

        vae_grad_heat = heatmap_from_input_grad(policy, x, "vae", args.vae_reduction)
        student_grad_heat = heatmap_from_input_grad(policy, x, "student", args.vae_reduction)

        stem = safe_filename(
            f"sample_{out_idx:03d}_demo_{demo_idx:05d}_{pathlib.Path(hdf5_path).stem}_{demo_key}"
        )
        short_title = f"{pathlib.Path(hdf5_path).name}:{demo_key}"

        latent_grid = make_grid(
            images_btchw=images_btchw,
            vae_heat_bthw=vae_latent_heat.cpu(),
            student_heat_bthw=student_latent_heat.cpu(),
            alpha=args.alpha,
            title=f"latent_norm | {short_title}",
            frame_ids=frame_ids,
        )
        latent_path = output_dir / f"{stem}_latent_norm.png"
        latent_grid.save(latent_path)

        grad_grid = make_grid(
            images_btchw=images_btchw,
            vae_heat_bthw=vae_grad_heat.cpu(),
            student_heat_bthw=student_grad_heat.cpu(),
            alpha=args.alpha,
            title=f"input_grad | {short_title}",
            frame_ids=frame_ids,
        )
        grad_path = output_dir / f"{stem}_input_grad.png"
        grad_grid.save(grad_path)

        if args.save_npy:
            np.savez_compressed(
                output_dir / f"{stem}_heatmaps.npz",
                frame_ids=np.asarray(frame_ids),
                vae_latent=minmax_per_frame(vae_latent_heat).numpy(),
                student_latent=minmax_per_frame(student_latent_heat).numpy(),
                vae_grad=minmax_per_frame(vae_grad_heat).numpy(),
                student_grad=minmax_per_frame(student_grad_heat).numpy(),
            )

        print(f"[{out_idx + 1}/{len(selected)}] saved {latent_path.name}, {grad_path.name}")


if __name__ == "__main__":
    main()
