import os
import sys
import pathlib
import random

import click
import dill
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt

from omegaconf import open_dict

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unified_video_action.workspace.base_workspace import BaseWorkspace


def _load_policy(ckpt_path, output_dir, device):
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    policy.to(device)
    policy.eval()
    return policy


def _default_param_names(diffactloss_module):
    names = []
    res_blocks = getattr(diffactloss_module.net, "res_blocks", None)
    n_blocks = len(res_blocks) if res_blocks is not None else 0
    last_block = max(n_blocks - 1, 0)

    candidates = [
        "interpolate.weight",
        "refine.0.weight",
        "refine.2.weight",
        "net.cond_embed.weight",
        "net.res_blocks.0.mlp.0.weight",
        f"net.res_blocks.{last_block}.adaLN_modulation.1.weight",
        "net.final_layer.linear.weight",
    ]
    param_map = dict(diffactloss_module.named_parameters())
    for key in candidates:
        if key in param_map:
            names.append(key)
    return names


def _plot_param_hists(diffactloss_module, param_names, out_path, bins):
    param_map = dict(diffactloss_module.named_parameters())
    existing = [name for name in param_names if name in param_map]
    missing = [name for name in param_names if name not in param_map]

    if len(existing) == 0:
        raise RuntimeError("None of the selected parameters were found in diffactloss.")

    n = len(existing)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 3.6 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, name in enumerate(existing):
        values = param_map[name].detach().float().cpu().reshape(-1).numpy()
        q_low, q_high = np.quantile(values, [0.001, 0.999])
        clipped = np.clip(values, q_low, q_high)

        ax = axes[i]
        ax.hist(clipped, bins=bins, density=True, alpha=0.8)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("value (clipped 0.1%-99.9%)", fontsize=9)
        ax.set_ylabel("density", fontsize=9)
        ax.grid(alpha=0.2)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("DiffActLoss parameter distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    return existing, missing


@click.command()
@click.option(
    "--checkpoint",
    default="checkpoints/pusht.ckpt",
    type=str,
    show_default=True,
    help="Path to checkpoint.",
)
@click.option(
    "--output-dir",
    default="out_diagram",
    type=str,
    show_default=True,
    help="Directory to save figure.",
)
@click.option("--device", default="cuda:0", type=str, show_default=True)
@click.option("--bins", default=120, type=int, show_default=True)
def main(checkpoint, output_dir, device, bins):
    output_dir = os.path.abspath(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading checkpoint: {checkpoint}")
    policy = _load_policy(checkpoint, output_dir=output_dir, device=device)

    if not hasattr(policy.model, "diffactloss"):
        raise RuntimeError("Checkpoint policy.model has no diffactloss.")

    diffactloss = policy.model.diffactloss
    param_names = _default_param_names(diffactloss)

    out_path = os.path.join(output_dir, "diffact_params_hist.png")
    existing, missing = _plot_param_hists(diffactloss, param_names, out_path, bins=bins)

    print(f"[INFO] Saved figure: {out_path}")
    print(f"[INFO] Plotted parameters ({len(existing)}):")
    for name in existing:
        print(f"  - {name}")
    if len(missing) > 0:
        print(f"[WARN] Missing parameters ({len(missing)}):")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
