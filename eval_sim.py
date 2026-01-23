import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import numpy as np
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
from omegaconf import open_dict, OmegaConf
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option(
    "--num_sampling_steps",
    type=int,
    default=None,
    show_default=True,
    help="Number of sampling steps to use."
)
@click.option(
    "--cfg",
    type=float,
    default=None,
    show_default=True,
    help="Classifier-free guidance factor."
)
@click.option(
    "--temperature",
    type=float,
    default=None,
    show_default=True,
    help="Temperature for sampling."
)
@click.option(
    "--n_test",
    type=int,
    default=None,
    show_default=True,
    help="Override number of test rollouts per task."
)
def main(checkpoint, output_dir, device, num_sampling_steps, cfg, temperature, n_test):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    config = payload["cfg"]
    
    # Update parameters if provided
    with open_dict(config.model.policy.autoregressive_model_params):
        if num_sampling_steps is not None:
            config.model.policy.autoregressive_model_params.num_sampling_steps = str(num_sampling_steps)
            config.model.policy.autoregressive_model_params.act_diff_testing_steps = str(num_sampling_steps)
        if cfg is not None:
            config.model.policy.autoregressive_model_params.cfg = cfg
        if temperature is not None:
            config.model.policy.autoregressive_model_params.temperature = temperature

    # set seed
    seed = config.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(config):
        config.output_dir = output_dir
        
    # configure workspace
    cls = hydra.utils.get_class(config.model._target_)
    workspace = cls(config, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    
    # get policy from workspace
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    # Override n_test if provided
    if n_test is not None:
        with open_dict(config):
            if "libero" in config.task.name:
                config.task.env_runner.n_test = n_test
            else:
                config.task.env_runner.n_test = n_test

    env_runners = load_env_runner(config, output_dir)

    if "libero" in config.task.name:
        step_log = {}
        for env_runner in env_runners:
            runner_log = env_runner.run(policy)
            step_log.update(runner_log)
            print(step_log)

        assert "test_mean_score" not in step_log
        all_test_mean_score = {
            k: v for k, v in step_log.items() if "test/" in k and "_mean_score" in k
        }
        step_log["test_mean_score"] = np.mean(list(all_test_mean_score.values()))

        runner_log = step_log
    else:
        env_runner = env_runners
        runner_log = env_runner.run(policy)

    # dump log to json
    json_log = dict()
    for key, value in runner_log.items():
        if isinstance(value, wandb.sdk.data_types.video.Video):
            json_log[key] = value._path
        else:
            json_log[key] = value

    for k, v in json_log.items():
        print(k, v)

    out_path = os.path.join(output_dir, f'eval_log_{checkpoint.split("/")[-1]}.json')
    print("Saving log to %s" % out_path)
    json.dump(json_log, open(out_path, "w"), indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
