import os
import random
import pathlib
import sys

import click
import dill
import hydra
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from omegaconf import open_dict

ROOT_DIR = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.data_utils import resize_image
from unified_video_action.eval.eval import prepare_data_predict_action


def _load_policy_and_cfg(
    ckpt_path, output_dir, device, act_diff_testing_steps, dataset_path
):
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        cfg.model.policy.autoregressive_model_params.act_diff_testing_steps = str(
            act_diff_testing_steps
        )
        if dataset_path is not None and hasattr(cfg.task, "dataset"):
            if "dataset_path" in cfg.task.dataset:
                cfg.task.dataset.dataset_path = dataset_path
            if "zarr_path" in cfg.task.dataset:
                cfg.task.dataset.zarr_path = dataset_path

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace: BaseWorkspace = cls(cfg, output_dir=output_dir)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if workspace.ema_model is not None else workspace.model
    policy.to(device)
    policy.eval()

    effective_steps = policy.model.diffactloss.gen_diffusion.num_timesteps
    print(
        f"[INFO] Effective action diffusion sampling steps: {effective_steps} "
        f"(requested: {act_diff_testing_steps})"
    )
    return cfg, policy


def _build_val_loader(cfg):
    if cfg.task.task_type == "multiple_datasets":
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        val_dataset = dataset.split_unused_episodes()
        return val_dataset.get_dataloader()

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    val_dataset = dataset.get_validation_dataset()
    return DataLoader(val_dataset, **cfg.val_dataloader)


def _resolve_block_index(block_index, n_blocks):
    if block_index < 0:
        block_index = n_blocks + block_index
    if block_index < 0 or block_index >= n_blocks:
        raise ValueError(f"block_index={block_index} out of range for n_blocks={n_blocks}")
    return block_index


def _maybe_downsample(values, max_points):
    if values.shape[0] <= max_points:
        return values
    idx = np.random.choice(values.shape[0], size=max_points, replace=False)
    return values[idx]


@torch.no_grad()
def _collect_internal_values(
    cfg,
    policy,
    loader,
    device,
    max_batches,
    target,
    block_index,
    max_points_per_call,
):
    if not hasattr(policy.model, "diffactloss"):
        raise RuntimeError("Current checkpoint does not have action diffusion head (diffactloss).")

    res_blocks = policy.model.diffactloss.net.res_blocks
    n_blocks = len(res_blocks)
    block_index = _resolve_block_index(block_index, n_blocks)
    target_block = res_blocks[block_index]
    print(f"[INFO] Collecting from ResBlock index: {block_index} / {n_blocks - 1}")

    collected = []

    def hook_fn(module, inputs, _output):
        x, y = inputs
        shift_mlp, scale_mlp, gate_mlp = module.adaLN_modulation(y).chunk(3, dim=-1)
        h = module.mlp(module.in_ln(x) * (1 + scale_mlp) + shift_mlp)
        delta = gate_mlp * h

        if target == "gate_mlp":
            values = gate_mlp.detach().float().reshape(-1).cpu().numpy()
        elif target == "delta":
            values = delta.detach().float().reshape(-1).cpu().numpy()
        else:
            raise ValueError(f"Unsupported target: {target}")

        values = _maybe_downsample(values, max_points_per_call)
        collected.append(values)

    handle = target_block.register_forward_hook(hook_fn)
    try:
        for n, batch in enumerate(loader):
            if n >= max_batches:
                break

            batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
            actions = batch["action"]

            if cfg.model.policy.use_history_action:
                batch = dict_apply(batch, lambda x: x[:, 1:])

            batch = resize_image(cfg, batch)
            bsz, T, _, _, _ = batch["obs"]["image"].size()

            if cfg.task.dataset.language_emb_model is not None:
                if "language" in batch["obs"]:
                    language_goal = batch["obs"]["language"]
                    del batch["obs"]["language"]
                elif "language_latents" in batch:
                    language_goal = batch["language_latents"]
                    del batch["language_latents"]
                else:
                    raise NotImplementedError("Language model enabled but no language input found.")
            else:
                language_goal = None

            (
                _x,
                _real,
                _latent_size,
                c,
                text_latents,
                history_trajectory,
                trajectory,
                proprioception_input,
            ) = prepare_data_predict_action(
                cfg, batch, actions, policy, T, device, language_goal=language_goal
            )

            policy.model.sample_tokens(
                bsz=bsz,
                cond=c,
                text_latents=text_latents,
                num_iter=cfg.model.policy.autoregressive_model_params.num_iter,
                cfg=cfg.model.policy.autoregressive_model_params.cfg,
                cfg_schedule=cfg.model.policy.autoregressive_model_params.cfg_schedule,
                temperature=cfg.model.policy.autoregressive_model_params.temperature,
                history_nactions=history_trajectory,
                nactions=trajectory,
                proprioception_input=proprioception_input,
                task_mode="policy_model",
            )
    finally:
        handle.remove()

    if len(collected) == 0:
        raise RuntimeError("No internal values collected. Increase --max-batches.")

    return np.concatenate(collected, axis=0), block_index


def _plot_hist(values, label, out_path, target, bins=120):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, density=True, alpha=0.75, label=label)
    plt.xlabel(f"{target} value")
    plt.ylabel("density")
    plt.title(f"{target} distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


@click.command()
@click.option(
    "--checkpoint",
    default="checkpoints/pusht.ckpt",
    type=str,
    show_default=True,
    help="Path to checkpoint.",
)
@click.option("--label", default="uva_2step", type=str, show_default=True)
@click.option(
    "--output-dir",
    default="out_diagram",
    type=str,
    show_default=True,
    help="Directory to save npz and png.",
)
@click.option("--device", default="cuda:0", type=str, show_default=True)
@click.option("--max-batches", default=50, type=int, show_default=True)
@click.option("--act-steps", default=2, type=int, show_default=True)
@click.option(
    "--target",
    default="gate_mlp",
    type=click.Choice(["gate_mlp", "delta"]),
    show_default=True,
    help="Which internal diffusion variable to collect.",
)
@click.option(
    "--block-index",
    default=-1,
    type=int,
    show_default=True,
    help="ResBlock index; -1 means last block.",
)
@click.option("--bins", default=120, type=int, show_default=True)
@click.option(
    "--max-points-per-call",
    default=200000,
    type=int,
    show_default=True,
    help="Randomly keep at most this many values for each denoiser forward call.",
)
@click.option(
    "--dataset-path",
    default="data/pusht/pusht_cchi_v7_replay.zarr",
    type=str,
    show_default=True,
    help="Dataset path override for cfg.task.dataset.",
)
def main(
    checkpoint,
    label,
    output_dir,
    device,
    max_batches,
    act_steps,
    target,
    block_index,
    bins,
    max_points_per_call,
    dataset_path,
):
    output_dir = os.path.abspath(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Collecting {target} from checkpoint: {checkpoint}")
    dataset_path = os.path.abspath(dataset_path)
    print(f"[INFO] Using dataset path: {dataset_path}")

    cfg, policy = _load_policy_and_cfg(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device,
        act_diff_testing_steps=act_steps,
        dataset_path=dataset_path,
    )
    loader = _build_val_loader(cfg)
    values, used_block = _collect_internal_values(
        cfg=cfg,
        policy=policy,
        loader=loader,
        device=device,
        max_batches=max_batches,
        target=target,
        block_index=block_index,
        max_points_per_call=max_points_per_call,
    )

    npz_path = os.path.join(output_dir, f"{label}_{target}_block{used_block}.npz")
    np.savez_compressed(
        npz_path,
        **{target: values},
        block_index=used_block,
        mean=values.mean(),
        std=values.std(),
    )
    print(f"[INFO] Saved values: {npz_path}")
    print(f"[INFO] {label} {target}: mean={values.mean():.6f}, std={values.std():.6f}")

    fig_path = os.path.join(output_dir, f"{target}_block{used_block}_hist.png")
    _plot_hist(values, label=label, out_path=fig_path, target=target, bins=bins)
    print(f"[INFO] Saved histogram: {fig_path}")


if __name__ == "__main__":
    main()
