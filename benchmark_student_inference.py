import argparse
import json
import pathlib
import random
import time
from typing import Any, Dict, Optional

import dill
import hydra
import numpy as np
import torch
from omegaconf import open_dict

from unified_video_action.workspace.base_workspace import BaseWorkspace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark policy inference latency (student tokenizer setup), unit: ms"
    )
    parser.add_argument(
        "--checkpoint",
        "-c",
        type=str,
        required=True,
        help="Path to ckpt file",
    )
    parser.add_argument(
        "--device",
        "-d",
        type=str,
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Inference batch size",
    )
    parser.add_argument(
        "--n-obs-steps",
        type=int,
        default=None,
        help="Observation horizon T; defaults to cfg.task.env_runner.n_obs_steps or 16",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=100,
        help="Timed iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--use-past-action",
        action="store_true",
        help="Attach synthetic past_action if policy expects history action",
    )
    parser.add_argument(
        "--language-goal",
        type=str,
        default="pick and place the object",
        help="Language goal string used when language embedding is enabled and not UMI",
    )
    parser.add_argument(
        "--require-student-tokenizer",
        action="store_true",
        help="Fail if checkpoint policy is not configured with student tokenizer",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save benchmark stats as json",
    )
    return parser.parse_args()


def _to_plain_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _build_dummy_obs(
    shape_meta: Dict[str, Any], batch_size: int, n_obs_steps: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    obs_dict = {}
    for key, attr in shape_meta["obs"].items():
        shape = tuple(attr["shape"])
        obs_type = attr.get("type", "low_dim")

        if obs_type == "rgb":
            if len(shape) != 3:
                raise ValueError(f"RGB obs '{key}' expects shape [C,H,W], got: {shape}")
            c, h, w = shape
            # match real pipeline convention: normalized float in [0, 1]
            value = torch.rand(batch_size, n_obs_steps, c, h, w, device=device)
        else:
            value = torch.randn(batch_size, n_obs_steps, *shape, device=device)

        obs_dict[key] = value
    return obs_dict


def _maybe_language_goal(cfg, batch_size: int, device: torch.device, text: str) -> Optional[Any]:
    language_emb_model = cfg.task.dataset.language_emb_model
    if language_emb_model is None:
        return None
    if "umi" in cfg.task.name:
        raise ValueError(
            "UMI task with language embedding enabled requires precomputed latent language_goal. "
            "Please pass your real language latent through a custom runner."
        )
    return [text] * batch_size


def _sync_if_needed(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    args = parse_args()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    ckpt_path = pathlib.Path(args.checkpoint)
    payload = torch.load(open(ckpt_path, "rb"), map_location="cpu", pickle_module=dill)
    cfg = payload["cfg"]

    with open_dict(cfg):
        if "autoregressive_model_params" in cfg.model.policy:
            # keep inference-time benchmark comparable and explicit
            cfg.model.policy.autoregressive_model_params.num_sampling_steps = str(
                cfg.model.policy.autoregressive_model_params.num_sampling_steps
            )

    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=".")
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    device = torch.device(args.device)
    policy = policy.eval().to(device)
    policy.reset()

    use_student = bool(getattr(policy, "use_student_tokenizer", False))
    if args.require_student_tokenizer and not use_student:
        raise RuntimeError(
            "This checkpoint is not using student tokenizer (use_student_tokenizer=False)."
        )

    n_obs_steps = args.n_obs_steps
    if n_obs_steps is None:
        env_runner_cfg = getattr(cfg.task, "env_runner", None)
        n_obs_steps = _to_plain_int(
            getattr(env_runner_cfg, "n_obs_steps", None) if env_runner_cfg is not None else None,
            16,
        )

    obs_dict = _build_dummy_obs(
        shape_meta=cfg.task.shape_meta,
        batch_size=args.batch_size,
        n_obs_steps=n_obs_steps,
        device=device,
    )
    if args.use_past_action:
        action_dim = int(cfg.task.shape_meta["action"]["shape"][0])
        # runner usually feeds two chunks of predicted actions -> total 2 * n_action_steps
        obs_dict["past_action"] = torch.randn(
            args.batch_size, 2 * int(policy.n_action_steps), action_dim, device=device
        )

    language_goal = _maybe_language_goal(
        cfg=cfg, batch_size=args.batch_size, device=device, text=args.language_goal
    )

    with torch.no_grad():
        for _ in range(args.warmup):
            _ = policy.predict_action(obs_dict=obs_dict, language_goal=language_goal)
        _sync_if_needed(device)

        times_ms = []
        for _ in range(args.repeat):
            _sync_if_needed(device)
            start = time.perf_counter()
            _ = policy.predict_action(obs_dict=obs_dict, language_goal=language_goal)
            _sync_if_needed(device)
            end = time.perf_counter()
            times_ms.append((end - start) * 1000.0)

    arr = np.array(times_ms, dtype=np.float64)
    stats = {
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "batch_size": args.batch_size,
        "n_obs_steps": n_obs_steps,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "use_student_tokenizer": use_student,
        "mean_ms": float(arr.mean()),
        "std_ms": float(arr.std()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }

    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.output_json is not None:
        output_path = pathlib.Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"Saved benchmark stats to: {output_path}")


if __name__ == "__main__":
    main()
