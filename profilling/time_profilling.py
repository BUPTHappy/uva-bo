import os
import sys
import torch
import hydra
from contextlib import contextmanager
from omegaconf import DictConfig, OmegaConf, open_dict

# ===================== CONFIG =====================
NUM_WARMUP = 20   # warmup iterations
NUM_ITERS  = 100  # timed iterations
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1

# ===================== HELPERS =====================
def _get(obj, name, default=None):
    return getattr(obj, name, default)

def _find_sample_tokens_owner(root):
    """
    Tries common places where sample_tokens may live.
    Returns (owner_obj, attr_name) or (None, None) if not found.
    """
    owner = _get(root, "model")
    if owner is not None and hasattr(owner, "sample_tokens"):
        return owner, "sample_tokens"
    if hasattr(root, "sample_tokens"):
        return root, "sample_tokens"
    # shallow search of attributes
    for _, child in root.__dict__.items():
        if hasattr(child, "sample_tokens"):
            return child, "sample_tokens"
    return None, None

def _find_diff_sample(root):
    """
    Finds policy.model.diffactloss.sample if present.
    Returns (diffactloss_module, "sample") or (None, None).
    """
    # typical structure: policy.model.diffactloss.sample
    inner = _get(root, "model", root)
    diffm = _get(inner, "diffactloss")
    if diffm is not None and hasattr(diffm, "sample"):
        return diffm, "sample"
    # fallback shallow search
    for _, child in inner.__dict__.items():
        if hasattr(child, "sample"):
            # heuristic: name == diffactloss
            if getattr(child, "__class__", None) and "diff" in child.__class__.__name__.lower():
                return child, "sample"
    return None, None

@contextmanager
def cuda_event_timer():
    """Context manager returning (start_event, end_event) via a lambda, elapsed in ms with CUDA events."""
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    yield lambda: (s, e)
    e.record()
    torch.cuda.synchronize()

# ===================== MAIN =====================
@hydra.main(
    version_base=None,
    config_path="../unified_video_action/config",
    config_name="config",
)
def main(cfg: DictConfig):
    if DEVICE != "cuda":
        raise RuntimeError("This timing script requires CUDA for accurate measurement.")

    LOG_PATH = cfg.save_folder

    # cfg tweaks (copied from your script)
    OmegaConf.resolve(cfg)
    if cfg.model.policy.action_model_params.predict_action is False:
        cfg.checkpoint.topk.monitor_key = "video_fvd"
        cfg.checkpoint.topk.format_str = "epoch={epoch:04d}-video_fvd={video_fvd:.3f}.ckpt"
        cfg.checkpoint.topk.mode = "min"

    with open_dict(cfg):
        cfg.n_gpus = torch.cuda.device_count()
        cfg.model.policy.debug = cfg.training.debug

    # ---------- Instantiate policy ----------
    language_emb_model = cfg.task.dataset.language_emb_model

    torch.backends.cudnn.benchmark = True
    with torch.inference_mode():
        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

        model = hydra.utils.instantiate(
            cfg.model.policy,
            task_name=cfg.task.name,
            task_modes=cfg.task.task_modes,
            normalizer_type=cfg.task.dataset.normalizer_type,
            language_emb_model=language_emb_model,
        ).to(DEVICE).eval()

        # ---------- Dummy normalizer (identity) ----------
        from unified_video_action.model.common.normalizer import (
            LinearNormalizer,
            SingleFieldLinearNormalizer,
        )
        dummy_norm = LinearNormalizer()
        dummy_norm["image"]  = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32).to(DEVICE)
        dummy_norm["action"] = SingleFieldLinearNormalizer.create_identity(dtype=torch.float32).to(DEVICE)
        model.set_normalizer(dummy_norm)

        # ---------- Dummy inputs ----------
        LANG = "KITCHEN SCENE6 put the yellow and white mug in the microwave and close it"
        image_tensor = torch.randn(
            BATCH_SIZE, 16, 3, 128, 128, device=DEVICE, dtype=torch.float32
        )
        obs_dict = {
            # Provide only the normalized key to avoid normalizer mismatches.
            "image": image_tensor,
        }
        language_goal = [LANG] * BATCH_SIZE  # length BATCH_SIZE

        # ---------- Warm-up ----------
        for _ in range(NUM_WARMUP):
            _ = model.predict_action(obs_dict, language_goal)

        # ---------- Monkey-patch timings ----------
        sample_owner, sample_attr = _find_sample_tokens_owner(model)
        diff_owner, diff_attr = _find_diff_sample(model)

        orig_sample_tokens = None
        orig_diff_sample = None
        orig_vae_encode = None

        sample_tokens_times = []  # per-iteration, we reset and append per-call ms
        diff_sample_times = []    # per-iteration, we reset and append per-call ms
        vae_encode_times = []     # per-iteration, we reset and append per-call ms

        # Patch VAE encoder
        if hasattr(model, 'vae_model'):
            from unified_video_action.utils.data_utils import extract_latent_autoregressive
            import unified_video_action.utils.data_utils as data_utils
            orig_vae_encode = data_utils.extract_latent_autoregressive
            
            def timed_vae_encode(vae_model, x):
                with cuda_event_timer() as evs:
                    out = orig_vae_encode(vae_model, x)
                    s, e = evs()
                vae_encode_times.append(s.elapsed_time(e))
                return out
            
            data_utils.extract_latent_autoregressive = timed_vae_encode

        if sample_owner is None:
            print("[WARN] Could not find sample_tokens; split timing limited to total only.")
        else:
            orig_sample_tokens = getattr(sample_owner, sample_attr)

            # patch diffactloss.sample if present
            if diff_owner is not None:
                orig_diff_sample = getattr(diff_owner, diff_attr)

                def timed_diff_sample(*args, **kwargs):
                    with cuda_event_timer() as evs:
                        out = orig_diff_sample(*args, **kwargs)
                        s, e = evs()
                    diff_sample_times.append(s.elapsed_time(e))
                    return out

                setattr(diff_owner, diff_attr, timed_diff_sample)
            else:
                print("[WARN] diffactloss.sample not found; nested timing unavailable.")

            # patch sample_tokens (outer)
            def timed_sample_tokens(*args, **kwargs):
                with cuda_event_timer() as evs:
                    out = orig_sample_tokens(*args, **kwargs)
                    s, e = evs()
                sample_tokens_times.append(s.elapsed_time(e))
                return out

            setattr(sample_owner, sample_attr, timed_sample_tokens)

        # ---------- Measurement loop ----------
        total_times = []
        tokens_times_per_iter = []
        diff_times_per_iter = []
        tokens_other_times = []
        pre_tokens_times = []
        vae_encode_times_per_iter = []

        for _ in range(NUM_ITERS):
            # reset per-iter buckets
            del sample_tokens_times[:]
            del diff_sample_times[:]
            del vae_encode_times[:]

            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            _ = model.predict_action(obs_dict, language_goal)
            e.record()
            torch.cuda.synchronize()

            total_ms = s.elapsed_time(e)
            total_times.append(total_ms)

            st_ms = float(sum(sample_tokens_times)) if sample_tokens_times else 0.0
            df_ms = float(sum(diff_sample_times)) if diff_sample_times else 0.0
            vae_ms = float(sum(vae_encode_times)) if vae_encode_times else 0.0

            tokens_times_per_iter.append(st_ms)
            diff_times_per_iter.append(df_ms)
            vae_encode_times_per_iter.append(vae_ms)

            pre_ms = max(0.0, total_ms - st_ms - vae_ms)
            other_ms = max(0.0, st_ms - df_ms)

            pre_tokens_times.append(pre_ms)
            tokens_other_times.append(other_ms)

        # ---------- Averages ----------
        avg_total = sum(total_times) / len(total_times)
        avg_tokens = sum(tokens_times_per_iter) / len(tokens_times_per_iter)
        avg_diff   = sum(diff_times_per_iter) / len(diff_times_per_iter)
        avg_tokens_other = sum(tokens_other_times) / len(tokens_other_times)
        avg_pre    = sum(pre_tokens_times) / len(pre_tokens_times)
        avg_vae    = sum(vae_encode_times_per_iter) / len(vae_encode_times_per_iter) if vae_encode_times_per_iter else 0.0
        avg_transformer = avg_tokens_other  # Transformer time is the "other" part inside sample_tokens

        # ---------- Restore originals ----------
        if sample_owner is not None and orig_sample_tokens is not None:
            setattr(sample_owner, sample_attr, orig_sample_tokens)
        if diff_owner is not None and orig_diff_sample is not None:
            setattr(diff_owner, diff_attr, orig_diff_sample)
        if orig_vae_encode is not None:
            import unified_video_action.utils.data_utils as data_utils
            data_utils.extract_latent_autoregressive = orig_vae_encode

        # ---------- Write log ----------
        with open(LOG_PATH, "w") as f:
            f.write(f"Command: {' '.join(sys.argv)}\n")
            f.write(f"\n=== Component-wise Timing (over {NUM_ITERS} iterations) ===\n\n")
            f.write(f"Total predict_action         : {avg_total:.3f} ms\n\n")
            f.write(f"Breakdown:\n")
            if avg_vae > 0:
                f.write(f"  VAE Image Encoder         : {avg_vae:.3f} ms\n")
            f.write(f"  Transformer               : {avg_transformer:.3f} ms\n")
            if diff_owner is not None and orig_diff_sample is not None:
                f.write(f"  Action Diffusion          : {avg_diff:.3f} ms\n")
            if avg_pre > 0:
                f.write(f"  Other preprocessing       : {avg_pre:.3f} ms\n")
            f.write(f"\n=== Detailed Breakdown ===\n\n")
            f.write(f"Avg full predict_action      : {avg_total:.3f} ms over {NUM_ITERS} iters\n")
            f.write(f"Avg sample_tokens (total)    : {avg_tokens:.3f} ms\n")
            if diff_owner is not None and orig_diff_sample is not None:
                f.write(f"  ├─ Avg diffactloss.sample  : {avg_diff:.3f} ms\n")
                f.write(f"  └─ Avg other inside tokens : {avg_tokens_other:.3f} ms\n")
            if avg_vae > 0:
                f.write(f"Avg VAE encoder             : {avg_vae:.3f} ms\n")
            f.write(f"Avg pre-sample_tokens        : {avg_pre:.3f} ms\n")

        print(f"[✓] Timing log written to {LOG_PATH}")

if __name__ == "__main__":
    main()
