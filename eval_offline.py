import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import json
import os
import pathlib
import random

import click
import dill
import hydra
import numpy as np
import torch
import wandb
from omegaconf import open_dict
from torch.utils.data import DataLoader

from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.common.pytorch_util import dict_apply
from unified_video_action.utils.data_utils import resize_image
from unified_video_action.eval.eval import test_action_l2
from unified_video_action.dataset.umi_multi_dataset import UmiMultiDataset
from unified_video_action.dataset.base_dataset import BaseImageDataset


@click.command()
@click.option("-c", "--checkpoint", required=True, help="Path to .ckpt")
@click.option("-o", "--output_dir", required=True, help="Directory to write metrics")
@click.option("-d", "--device", default="cuda:0")
@click.option(
    "--disable_vae_cond_eval/--no-disable_vae_cond_eval",
    default=False,
    help="Eval-only: remove VAE conditioning frames by feeding a zero latent.",
)
@click.option(
    "--max_batches",
    default=None,
    type=int,
    help="Optional cap on number of val batches.",
)
@click.option(
    "--umi_dataset_root_dir",
    default=None,
    type=str,
    help="Override cfg.task.dataset.dataset_root_dir for UMI multi-dataset eval.",
)
@click.option(
    "--umi_used_episode_indices_file",
    default=None,
    type=str,
    help=(
        "Override cfg.task.dataset.used_episode_indices_file. "
        "Use empty string '' to disable."
    ),
)
@click.option(
    "--umi_dataset_names",
    default=None,
    type=str,
    help=(
        "Optional comma-separated dataset names to eval (overrides cfg.task.dataset.dataset_configs keys). "
        "Example: 'cup_arrangement_0,cloth_folding_0,dish_washing_0,dynamic_tossing_0'."
    ),
)
@click.option(
    "--disable_language/--no-disable_language",
    default=False,
    help=(
        "Disable language conditioning for offline eval without changing model structure "
        "(keeps text modules loaded but skips using them)."
    ),
)
def main(
    checkpoint,
    output_dir,
    device,
    disable_vae_cond_eval,
    max_batches,
    umi_dataset_root_dir,
    umi_used_episode_indices_file,
    umi_dataset_names,
    disable_language,
):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    payload = torch.load(open(checkpoint, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        cfg.model.policy.disable_vae_cond_eval = bool(disable_vae_cond_eval)

        if max_batches is not None:
            cfg.training.max_val_steps = int(max_batches)

        # Do NOT change cfg.task.dataset.language_emb_model here; it would change
        # model structure and break checkpoint loading. Instead, set an eval flag
        # to skip using language conditioning.
        if disable_language:
            if not hasattr(cfg, "eval"):
                cfg.eval = {}
            cfg.eval.disable_language = True

        # UMI multi-dataset overrides (ckpt may contain cluster-specific paths).
        if getattr(cfg, "task", None) is not None and cfg.task.task_type == "multiple_datasets":
            if umi_dataset_root_dir is not None:
                cfg.task.dataset.dataset_root_dir = umi_dataset_root_dir
            if umi_used_episode_indices_file is not None:
                # allow disabling by passing ''
                cfg.task.dataset.used_episode_indices_file = umi_used_episode_indices_file
            if umi_dataset_names is not None:
                names = [n.strip() for n in umi_dataset_names.split(",") if n.strip()]
                cfg.task.dataset.dataset_configs = {
                    n: {"include_episode_num": -1, "mask_mirror": False} for n in names
                }

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()

    # Build val dataloader (match training workspace behavior).
    if cfg.task.task_type == "multiple_datasets":
        dataset: UmiMultiDataset = hydra.utils.instantiate(cfg.task.dataset)
        val_dataset = dataset.split_unused_episodes()
        val_dataloader = val_dataset.get_dataloader()
    else:
        dataset: BaseImageDataset = hydra.utils.instantiate(cfg.task.dataset)
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

    # Optional cap for quicker debugging.
    if max_batches is not None:
        # Wrap iterator by breaking early inside test_action_l2 via cfg.training.debug style.
        # Here we just set cfg.training.debug to True would be too aggressive;
        # max_val_steps is already consumed by training workspace, so we mimic it by truncating.
        pass

    # Ensure device transfer + resize happens inside test_action_l2; it already resizes in eval/eval.py.
    # But for consistency with training, keep images in expected shape in batches.
    # test_action_l2 handles device transfer itself.
    log = test_action_l2(
        cfg=cfg,
        model=policy,
        loader=val_dataloader,
        it=0,
        output_dir=output_dir,
        device=device,
        name_label="offline_",
    )

    # Dump log to json (handle wandb.Video)
    json_log = dict()
    for key, value in log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    out_path = os.path.join(output_dir, f"offline_eval_{os.path.basename(checkpoint)}.json")
    print("Saving log to %s" % out_path)
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)

    for k, v in json_log.items():
        print(k, v)


if __name__ == "__main__":
    main()

