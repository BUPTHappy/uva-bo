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


def _load_policy_and_cfg(ckpt_path, output_dir, device, act_diff_testing_steps):
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


@torch.no_grad()
def _collect_z_refine(cfg, policy, loader, device, max_batches):
    if not hasattr(policy.model, "diffactloss"):
        raise RuntimeError("Current checkpoint does not have action diffusion head (diffactloss).")

    policy.model.diffactloss.capture_debug = True
    policy.model.diffactloss.debug_tensors = {}

    all_vals = []
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
        ) = prepare_data_predict_action(cfg, batch, actions, policy, T, device, language_goal=language_goal)

        _z, _act_out = policy.model.sample_tokens(
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

        z_refine = policy.model.diffactloss.debug_tensors.get("z_refine")
        if z_refine is None:
            raise RuntimeError("z_refine was not captured. Check act_model_type path and debug switch.")

        all_vals.append(z_refine.reshape(-1).numpy())

    if len(all_vals) == 0:
        raise RuntimeError("No z_refine collected. Try increasing --max_batches.")

    return np.concatenate(all_vals, axis=0)


def _plot_hist(values, label, out_path, bins=120):
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins, density=True, alpha=0.7, label=label)
    plt.xlabel("z_refine value")
    plt.ylabel("density")
    plt.title("z_refine distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
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
def main(checkpoint, label, output_dir, device, max_batches, act_steps):
    output_dir = os.path.abspath(output_dir)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Collecting z_refine from checkpoint: {checkpoint}")
    cfg, policy = _load_policy_and_cfg(
        checkpoint, output_dir, device, act_diff_testing_steps=act_steps
    )
    loader = _build_val_loader(cfg)
    values = _collect_z_refine(cfg, policy, loader, device, max_batches)
    np.savez_compressed(
        os.path.join(output_dir, f"{label}_z_refine.npz"),
        z_refine=values,
        mean=values.mean(),
        std=values.std(),
    )
    print(f"[INFO] {label}: mean={values.mean():.6f}, std={values.std():.6f}")

    fig_path = os.path.join(output_dir, "z_refine_hist.png")
    _plot_hist(values, label, fig_path)
    print(f"[INFO] Saved histogram: {fig_path}")


if __name__ == "__main__":
    main()
