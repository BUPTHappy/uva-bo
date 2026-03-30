#!/usr/bin/env python3
"""
Count trainable parameters: StudentLatentTokenizer (student_tokenizer_params) vs KL-VAE.

Run from anywhere after `pip install -e .` in the project root, or:
  cd /path/to/unified_video_action && python scripts/count_student_vae_params.py

Optional: point to a Hydra/OmegaConf YAML that defines `policy.student_tokenizer_params`
and `policy.vae_model_params` (same structure as config/model/uva.yaml).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Project root: parent of this file's directory (folder that contains `unified_video_action/` package)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch.nn as nn

try:
    from omegaconf import OmegaConf
except ImportError as e:
    raise SystemExit(
        "OmegaConf is required (same env as training). Install: pip install omegaconf"
    ) from e

from unified_video_action.model.common.student_tokenizer import StudentLatentTokenizer
from unified_video_action.vae.vaekl import AutoencoderKL


def _num_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _default_policy_blocks() -> tuple[dict[str, Any], dict[str, Any]]:
    """Defaults aligned with unified_video_action/config/model/uva.yaml."""
    student_tokenizer_params = {
        "img_size": 256,
        "patch_size": 16,
        "in_channels": 3,
        "latent_channels": 16,
        "hidden_dim": 256,
        "depth": 4,
        "num_heads": 8,
        "mlp_ratio": 4.0,
        "dropout": 0.0,
    }
    vae_model_params = {
        "autoencoder_path": None,
        "ddconfig": SimpleNamespace(
            vae_embed_dim=16, ch_mult=[1, 1, 2, 2, 4]
        ),
    }
    return student_tokenizer_params, vae_model_params


def _load_from_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = OmegaConf.load(path)
    policy = cfg.get("policy", cfg)
    if policy is None:
        raise ValueError(f"No `policy` block in {path}")
    st = policy.get("student_tokenizer_params")
    vae = policy.get("vae_model_params")
    if st is None:
        raise ValueError(f"{path} has no policy.student_tokenizer_params")
    if vae is None:
        raise ValueError(f"{path} has no policy.vae_model_params")
    # Plain dicts for **kwargs (OmegaConf also works for AutoencoderKL attribute access)
    st = OmegaConf.to_container(st, resolve=True)
    vae = OmegaConf.to_container(vae, resolve=True)
    assert isinstance(st, dict) and isinstance(vae, dict)
    # Never load weights in this script — structure-only param count
    vae["autoencoder_path"] = None
    dd = vae.get("ddconfig")
    if isinstance(dd, dict):
        vae["ddconfig"] = SimpleNamespace(**dd)
    return st, vae


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parameter count: student tokenizer vs KL-VAE (encode path & full)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML with policy.student_tokenizer_params and policy.vae_model_params "
        "(e.g. config/model/uva.yaml). If omitted, uses built-in defaults from uva.yaml.",
    )
    args = parser.parse_args()

    if args.config is not None:
        st_kwargs, vae_kwargs = _load_from_config(args.config.resolve())
    else:
        st_kwargs, vae_kwargs = _default_policy_blocks()

    student = StudentLatentTokenizer(**st_kwargs)
    vae = AutoencoderKL(**vae_kwargs)

    n_student = _num_params(student)
    n_vae_total = _num_params(vae)
    n_vae_encoder = _num_params(vae.encoder)
    n_quant = _num_params(vae.quant_conv)
    n_vae_encode_stack = n_vae_encoder + n_quant
    n_vae_decoder = _num_params(vae.decoder)
    n_post_quant = _num_params(vae.post_quant_conv)

    ratio_vs_encode = n_student / n_vae_encode_stack if n_vae_encode_stack else float("nan")
    ratio_vs_full = n_student / n_vae_total if n_vae_total else float("nan")

    print("=" * 72)
    print("Parameter counts (random init — same as checkpoint structure)")
    print("=" * 72)
    print()
    print("StudentLatentTokenizer (student_tokenizer_params → trains instead of using VAE encoder for latents)")
    print(f"  Total:     {n_student:>14,}  ({n_student / 1e6:.3f} M)")
    print()
    print("KL-VAE (frozen teacher in policy — AutoencoderKL)")
    print(f"  Encoder:   {n_vae_encoder:>14,}  ({n_vae_encoder / 1e6:.3f} M)")
    print(f"  quant_conv:{n_quant:>14,}  ({n_quant / 1e6:.3f} M)")
    print(f"  Encode path (encoder + quant_conv, comparable role to student): {n_vae_encode_stack:>14,}  ({n_vae_encode_stack / 1e6:.3f} M)")
    print(f"  post_quant:{n_post_quant:>14,}  ({n_post_quant / 1e6:.3f} M)")
    print(f"  Decoder:   {n_vae_decoder:>14,}  ({n_vae_decoder / 1e6:.3f} M)")
    print(f"  Full VAE:  {n_vae_total:>14,}  ({n_vae_total / 1e6:.3f} M)")
    print()
    print("Ratios (student / …)")
    print(f"  vs VAE encode stack: {ratio_vs_encode:.4f}x")
    print(f"  vs full VAE:         {ratio_vs_full:.4f}x")
    print("=" * 72)


if __name__ == "__main__":
    main()
