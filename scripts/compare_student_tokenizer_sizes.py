#!/usr/bin/env python3
"""Compare student tokenizer presets vs the **frozen KL-VAE encoder pathway**.

This repo uses encoder latents distilled from observations; sizing is referenced to
``encoder`` + ``quant_conv`` (**Enc**), not decoder / full AE.

Torch counts modules when available; analytical VAE layout matches ``vaekl.py`` Decoder
defaults (no spatial self-attention in upsampling stages).

Usage:
  python scripts/compare_student_tokenizer_sizes.py
  python scripts/compare_student_tokenizer_sizes.py --size equal
  python scripts/compare_student_tokenizer_sizes.py --config-dir unified_video_action/config
"""

from __future__ import annotations

import argparse
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PRESETS = {
    "small": {
        "hidden_dim": 304,
        "depth": 5,
        "num_heads": 8,
        "note": "compact (~0.6× Enc)",
    },
    "equal": {
        "hidden_dim": 384,
        "depth": 6,
        "num_heads": 8,
        "note": "~parity with frozen Enc+quant (~1.0× Enc)",
    },
    "large": {
        "hidden_dim": 464,
        "depth": 9,
        "num_heads": 8,
        "note": "scaled (~1.8× Enc)",
    },
}

SHARED_STUDENT_KEYS = {
    "img_size": 256,
    "patch_size": 16,
    "in_channels": 3,
    "latent_channels": 16,
    "mlp_ratio": 4.0,
    "dropout": 0.0,
    "use_temporal_mixer": True,
    "temporal_kernel_size": 3,
}

VAE_DDCONFIG = {"vae_embed_dim": 16, "ch_mult": [1, 1, 2, 2, 4]}
ALIGN_PROJECTOR_DIM = 512


def _count_transformer_encoder_layer(d: int, mlp_ratio: float = 4.0) -> int:
    ff = int(d * mlp_ratio)
    return (3 * d * d + 3 * d) + (d * d + d) + (d * ff + ff) + (ff * d + d) + 4 * d


def count_student_tokenizer_analytical(params: dict) -> int:
    d = int(params["hidden_dim"])
    depth = int(params["depth"])
    img_size = int(params["img_size"])
    patch_size = int(params["patch_size"])
    in_channels = int(params["in_channels"])
    latent_channels = int(params["latent_channels"])
    use_temporal_mixer = bool(params.get("use_temporal_mixer", True))
    temporal_kernel_size = int(params.get("temporal_kernel_size", 3))
    mlp_ratio = float(params.get("mlp_ratio", 4.0))

    grid = (img_size // patch_size) ** 2
    stem = in_channels * (d // 2) * 9 + (d // 2)
    stem += (d // 2) * (d // 2) * 9 + (d // 2)
    patch = (d // 2) * d * patch_size * patch_size + d
    pos = grid * d
    blocks = depth * _count_transformer_encoder_layer(d, mlp_ratio=mlp_ratio)
    temporal = (d * temporal_kernel_size + d) if use_temporal_mixer else 0
    norm = 4 * d
    out = d * latent_channels + latent_channels
    return stem + patch + pos + blocks + temporal + norm + out


def count_align_projector_analytical(hidden_dim: int, latent_channels: int = 16) -> int:
    d0, d1, d2 = hidden_dim, ALIGN_PROJECTOR_DIM, latent_channels
    return (d0 * d1 + d1) + (d1 * d1 + d1) + (d1 * d2 + d2)


def count_conv2d(ci: int, co: int, k: int, bias: bool = True) -> int:
    n = ci * co * k * k
    return n + (co if bias else 0)


def count_groupnorm(c: int) -> int:
    return 2 * c


def count_resnet(in_c: int, out_c: int) -> int:
    n = 2 * count_groupnorm(in_c)
    n += count_conv2d(in_c, out_c, 3)
    n += 2 * count_groupnorm(out_c)
    n += count_conv2d(out_c, out_c, 3)
    if in_c != out_c:
        n += count_conv2d(in_c, out_c, 1)
    return n


def count_attn(c: int) -> int:
    n = count_groupnorm(c)
    n += 3 * count_conv2d(c, c, 1)
    n += count_conv2d(c, c, 1)
    return n


def count_vae_analytical() -> dict[str, int]:
    """Match ``AutoencoderKL`` + Encoder / Decoder defaults in ``vaekl.py``."""

    ch = 128
    ch_mult = (1, 1, 2, 2, 4)
    z = 16
    res = 256
    num_res_blocks = 2
    attn_res_encoder = (16,)
    attn_res_decoder: tuple[int, ...] = ()

    in_ch_mult = (1,) + tuple(ch_mult)

    def encoder() -> int:
        n = count_conv2d(3, ch, 3)
        curr = res
        for i_level, mult in enumerate(ch_mult):
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * mult
            for _ in range(num_res_blocks):
                n += count_resnet(block_in, block_out)
                block_in = block_out
                if curr in attn_res_encoder:
                    n += count_attn(block_in)
            if i_level != len(ch_mult) - 1:
                n += count_conv2d(block_in, block_in, 3)
                curr //= 2
        n += count_resnet(block_in, block_in)
        n += count_attn(block_in)
        n += count_resnet(block_in, block_in)
        n += count_groupnorm(block_in)
        n += count_conv2d(block_in, 2 * z, 3)
        return n

    def decoder() -> int:
        block_in = ch * ch_mult[-1]
        n = count_conv2d(z, block_in, 3)
        n += count_resnet(block_in, block_in)
        n += count_attn(block_in)
        n += count_resnet(block_in, block_in)
        curr = res // 2 ** (len(ch_mult) - 1)
        for i_level, mult in reversed(list(enumerate(ch_mult))):
            block_out = ch * mult
            for _ in range(num_res_blocks + 1):
                n += count_resnet(block_in, block_out)
                block_in = block_out
                if curr in attn_res_decoder:
                    n += count_attn(block_in)
            if i_level != 0:
                n += count_conv2d(block_in, block_in, 3)
                curr *= 2
        n += count_groupnorm(block_in)
        n += count_conv2d(block_in, 3, 3)
        return n

    enc = encoder()
    dec = decoder()
    quant = count_conv2d(2 * z, 2 * z, 1)
    post = count_conv2d(z, z, 1)
    return {
        "vae_encoder_quant": enc + quant,
        "vae_decoder": dec + post,
        "vae_total": enc + dec + quant + post,
    }


def count_with_torch(student_params: dict) -> dict[str, int] | None:
    try:
        import contextlib
        import io

        import torch
        from omegaconf import OmegaConf

        from unified_video_action.model.common.student_tokenizer import (
            StudentLatentTokenizer,
        )
        from unified_video_action.vae.vaekl import AutoencoderKL
    except ImportError:
        return None

    def nparams(module) -> int:
        return sum(p.numel() for p in module.parameters())

    student = StudentLatentTokenizer(**student_params)
    with contextlib.redirect_stdout(io.StringIO()):
        vae = AutoencoderKL(autoencoder_path=None, ddconfig=OmegaConf.create(VAE_DDCONFIG))
    hidden_dim = int(student_params["hidden_dim"])
    latent_channels = int(student_params["latent_channels"])
    projector = torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, ALIGN_PROJECTOR_DIM),
        torch.nn.SiLU(),
        torch.nn.Linear(ALIGN_PROJECTOR_DIM, ALIGN_PROJECTOR_DIM),
        torch.nn.SiLU(),
        torch.nn.Linear(ALIGN_PROJECTOR_DIM, latent_channels),
    )
    encq = nparams(vae.encoder) + nparams(vae.quant_conv)
    return {
        "student": nparams(student),
        "align_projector": nparams(projector),
        "student_total": nparams(student) + nparams(projector),
        "vae_encoder_quant": encq,
        "vae_decoder": nparams(vae.decoder) + nparams(vae.post_quant_conv),
        "vae_total": nparams(vae),
        "backend": "torch",
    }


def build_student_params(size: str) -> dict:
    if size not in PRESETS:
        raise ValueError(f"Unknown size '{size}'. Choose from: {', '.join(PRESETS)}")
    return {**SHARED_STUDENT_KEYS, **{k: v for k, v in PRESETS[size].items() if k != "note"}}


def summarize_size(size: str, use_torch: bool) -> dict:
    student_params = build_student_params(size)
    counts = None
    if use_torch:
        counts = count_with_torch(student_params)
    if counts is None:
        vae = count_vae_analytical()
        student_n = count_student_tokenizer_analytical(student_params)
        proj_n = count_align_projector_analytical(student_params["hidden_dim"])
        counts = {
            "student": student_n,
            "align_projector": proj_n,
            "student_total": student_n + proj_n,
            **vae,
            "backend": "analytical",
        }
    enc = counts["vae_encoder_quant"]
    counts["size"] = size
    counts["note"] = PRESETS[size]["note"]
    counts["student_config"] = student_params
    counts["ratio_student_vs_encoder"] = counts["student"] / enc
    counts["ratio_student_total_vs_encoder"] = counts["student_total"] / enc
    return counts


def _fmt_m(n: int) -> str:
    return f"{n / 1e6:7.2f}M ({n:,})"


def print_report(rows: list[dict]) -> None:
    backend = rows[0]["backend"]
    enc_ref = rows[0]["vae_encoder_quant"]
    tot_ref = rows[0]["vae_total"]
    print(f"Parameter backend: {backend}")
    print(
        "(**Enc** = frozen VAE `encoder` + `quant_conv`, the pathway used into latents.)\n"
    )
    header = (
        f"{'preset':<6} {'student':>14} {'+projector':>14} {'vs Enc':>8} "
        f"{'Enc+quant ref':>16} {'VAE full':>14}  note"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['size']:<6} {_fmt_m(row['student']):>14} "
            f"{_fmt_m(row['student_total']):>14} "
            f"{row['ratio_student_vs_encoder']:>7.2f}x "
            f"{_fmt_m(enc_ref):>16} {_fmt_m(tot_ref):>14}  "
            f"{row['note']}"
        )
    print()


def maybe_load_hydra_preset(config_dir: pathlib.Path, size: str) -> dict | None:
    try:
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from omegaconf import OmegaConf
    except ImportError:
        return None

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="uva_libero10_student",
            overrides=[f"model/student_tokenizer={size}"],
        )
    return OmegaConf.to_container(cfg.model.policy.student_tokenizer_params, resolve=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare student tokenizer presets to frozen KL-VAE encoder (+ quant_conv)."
        )
    )
    parser.add_argument(
        "--size",
        choices=["small", "equal", "large", "all"],
        default="all",
        help="Which preset to report (default: all).",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=str(ROOT / "unified_video_action" / "config"),
        help="Hydra config directory (optional YAML cross-check).",
    )
    parser.add_argument(
        "--no-torch",
        action="store_true",
        help="Force analytical counting even if PyTorch is installed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = list(PRESETS) if args.size == "all" else [args.size]
    use_torch = not args.no_torch

    rows = [summarize_size(size, use_torch=use_torch) for size in sizes]
    print_report(rows)

    config_dir = pathlib.Path(args.config_dir).resolve()
    if config_dir.is_dir() and len(sizes) == 1:
        try:
            hydra_params = maybe_load_hydra_preset(config_dir, sizes[0])
        except Exception as exc:
            hydra_params = None
            print(f"Hydra preset check skipped: {exc}")
        else:
            if hydra_params is not None:
                built = build_student_params(sizes[0])
                if hydra_params != built:
                    print("Warning: Hydra YAML preset differs from script table:")
                    print(f"  hydra:  {hydra_params}")
                    print(f"  script: {built}")
                else:
                    print("Hydra preset matches script table.")

    if len(rows) == 1:
        cfg = rows[0]["student_config"]
        print("Student config:")
        for key in sorted(cfg):
            print(f"  {key}: {cfg[key]}")


if __name__ == "__main__":
    main()
