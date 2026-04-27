#!/usr/bin/env python3
import argparse
import os
import sys
from types import SimpleNamespace

# Ensure the repo root is importable when running as:
# `python scripts/compare_student_vae_params.py`
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from unified_video_action.model.common.student_tokenizer import StudentLatentTokenizer
from unified_video_action.vae.vaekl import AutoencoderKL


def count_params(module):
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total, trainable


def fmt(n):
    return f"{n:,}"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare parameter counts between student tokenizer and "
            "VAE encoder+quant_conv."
        )
    )
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=384)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-temporal-mixer", action="store_true", default=True)
    parser.add_argument("--no-temporal-mixer", action="store_false", dest="use_temporal_mixer")
    parser.add_argument("--temporal-kernel-size", type=int, default=3)

    parser.add_argument("--vae-embed-dim", type=int, default=16)
    parser.add_argument(
        "--vae-ch-mult",
        type=int,
        nargs="+",
        default=[1, 1, 2, 2, 4],
        help="Example: --vae-ch-mult 1 1 2 2 4",
    )
    parser.add_argument(
        "--autoencoder-path",
        type=str,
        default="pretrained_models/vae/kl16.ckpt",
        help="Only used if file exists; architecture is identical either way.",
    )
    args = parser.parse_args()

    student = StudentLatentTokenizer(
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        latent_channels=args.latent_channels,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        use_temporal_mixer=args.use_temporal_mixer,
        temporal_kernel_size=args.temporal_kernel_size,
    )

    ddconfig = SimpleNamespace(
        vae_embed_dim=args.vae_embed_dim,
        ch_mult=list(args.vae_ch_mult),
    )
    vae = AutoencoderKL(
        autoencoder_path=args.autoencoder_path,
        ddconfig=ddconfig,
    )

    student_total, student_trainable = count_params(student)
    vae_enc_total, vae_enc_trainable = count_params(vae.encoder)
    vae_q_total, vae_q_trainable = count_params(vae.quant_conv)
    vae_target_total = vae_enc_total + vae_q_total
    vae_target_trainable = vae_enc_trainable + vae_q_trainable

    ratio = student_total / vae_target_total if vae_target_total > 0 else float("nan")

    print("=== Parameter Count Comparison ===")
    print(
        "Student tokenizer config: "
        f"img={args.img_size}, patch={args.patch_size}, in={args.in_channels}, "
        f"latent={args.latent_channels}, hidden={args.hidden_dim}, depth={args.depth}, "
        f"heads={args.num_heads}, mlp_ratio={args.mlp_ratio}, dropout={args.dropout}, "
        f"temporal_mixer={args.use_temporal_mixer}, tk={args.temporal_kernel_size}"
    )
    print(
        "VAE encoder config: "
        f"embed_dim={args.vae_embed_dim}, ch_mult={list(args.vae_ch_mult)} "
        "(counting encoder + quant_conv only)"
    )
    print()
    print(f"Student tokenizer total params      : {fmt(student_total)}")
    print(f"VAE (encoder) total    : {fmt(vae_target_total)}")
    print(f"Student / VAE ratio  : {ratio:.4f}x")


if __name__ == "__main__":
    main()
