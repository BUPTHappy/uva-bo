import torch
import os
from typing import Dict, Tuple
import torch.nn.functional as F
import random
import numpy as np

from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.common.pytorch_util import dict_apply

from unified_video_action.utils.data_utils import (
    process_data,
    extract_latent_autoregressive,
    get_trajectory,
    get_vae_latent,
    resize_image_eval,
)
from unified_video_action.utils.data_utils import (
    normalize_action,
    normalize_obs,
    normalize_past_action,
    unnormalize_future_action,
)
from unified_video_action.model.autoregressive import mar_con_unified as mar
from unified_video_action.vae.vaekl import AutoencoderKL
from unified_video_action.utils.language_model import (
    get_text_model,
    extract_text_features,
)
from unified_video_action.model.common.student_tokenizer import StudentLatentTokenizer

class UnifiedVideoActionPolicy(BaseImagePolicy):
    def __init__(
        self,
        vae_model_params,
        autoregressive_model_params,
        action_model_params,
        shape_meta: dict,
        n_action_steps,
        shift_action=True,
        language_emb_model=None,
        task_name=None,
        task_modes=[],
        **kwargs
     ):
        super().__init__()

        self.task_name = task_name
        self.task_modes = task_modes
        self.autoregressive_model_params = autoregressive_model_params
        self.n_action_steps = n_action_steps
        self.shift_action = shift_action
        self.language_emb_model = language_emb_model
        self.action_dim = shape_meta.action.shape[0]

        self.kwargs = kwargs
        self.normalizer_type = kwargs["normalizer_type"]
        self.selected_training_mode = kwargs["selected_training_mode"]

        self.use_history_action = kwargs["use_history_action"]
        self.use_proprioception = kwargs["use_proprioception"]

        # =========================== REPA-style student tokenizer config ===========================
        self.use_student_tokenizer = bool(kwargs.get("use_student_tokenizer", False))
        self.student_tokenizer_params = kwargs.get("student_tokenizer_params", None)
        self.align_params = kwargs.get("align_params", None)

        # alignment controls 
        self.use_alignment = False
        self.align_coeff = 0.0
        self.align_warmup_steps = 0
        self.align_pretrain_steps = 10000
        self.align_post_start_coeff = 0.5
        self.align_min_coeff = 0.0
        self.align_decay_steps = 50000
        self.align_loss_type = "cosine"
        self.align_use_projector = True
        self.align_projector = None
        self.align_teacher_use_mode = True
        self.align_debug = False
        self.align_debug_every = 200
        self._align_step = 0
        self._main_model_frozen = False

        if self.align_params is not None:
            self.use_alignment = bool(self.align_params.get("enable", False))
            self.align_coeff = float(self.align_params.get("coeff", 0.0))
            self.align_warmup_steps = int(self.align_params.get("warmup_steps", 0))
            self.align_pretrain_steps = int(self.align_params.get("pretrain_steps", 10000))
            self.align_post_start_coeff = float(
                self.align_params.get("post_start_coeff", 0.5)
            )
            self.align_min_coeff = float(self.align_params.get("min_coeff", 0.0))
            self.align_decay_steps = int(
                self.align_params.get("decay_steps", max(1, self.align_warmup_steps))
            )
            self.align_loss_type = str(self.align_params.get("loss_type", "cosine"))
            self.align_use_projector = bool(self.align_params.get("use_projector", True))
            self.align_teacher_use_mode = bool(self.align_params.get("teacher_use_mode", True))
            self.align_debug = bool(self.align_params.get("debug", False))
            self.align_debug_every = int(self.align_params.get("debug_every", 200))

        # =========================== load vae model (teacher / legacy path) ===========================
        with torch.no_grad():
            self.vae_model = AutoencoderKL(**vae_model_params)
        self.vae_model.eval()
        for param in self.vae_model.parameters():
            param.requires_grad = False

        # =========================== student tokenizer ===========================
        self.student_tokenizer = None
        if self.use_student_tokenizer:
            if self.student_tokenizer_params is None:
                raise ValueError(
                    "use_student_tokenizer=True but student_tokenizer_params is not provided."
                )
            self.student_tokenizer = StudentLatentTokenizer(**self.student_tokenizer_params)

            if self.use_alignment and self.align_use_projector:
                latent_channels = int(
                    self.student_tokenizer_params.get(
                        "latent_channels", autoregressive_model_params.vae_embed_dim
                    )
                )
                hidden_dim = int(self.student_tokenizer_params.get("hidden_dim", 256))
                projector_dim = int(self.align_params.get("projector_dim", 512))
                self.align_projector = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, projector_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(projector_dim, projector_dim),
                    torch.nn.SiLU(),
                    torch.nn.Linear(projector_dim, latent_channels),
                )

        # =========================== load language model ===========================
        self.text_model, self.tokenizer, self.max_length = get_text_model(
            task_name, language_emb_model
        )
        if self.text_model is not None:
            self.text_model.eval()
            for param in self.text_model.parameters():
                param.requires_grad = False

        # =========================== main model ===========================
        self.model = mar.__dict__[autoregressive_model_params.model_size](
            img_size=autoregressive_model_params.img_size,
            vae_stride=autoregressive_model_params.vae_stride,
            patch_size=autoregressive_model_params.patch_size,
            vae_embed_dim=autoregressive_model_params.vae_embed_dim,
            mask_ratio_min=autoregressive_model_params.mask_ratio_min,
            label_drop_prob=autoregressive_model_params.label_drop_prob,
            attn_dropout=autoregressive_model_params.attn_dropout,
            proj_dropout=autoregressive_model_params.proj_dropout,
            diffloss_d=autoregressive_model_params.diffloss_d,
            diffloss_w=autoregressive_model_params.diffloss_w,
            diffloss_act_d=autoregressive_model_params.diffloss_act_d,
            diffloss_act_w=autoregressive_model_params.diffloss_act_w,
            num_sampling_steps=autoregressive_model_params.num_sampling_steps,
            diffusion_batch_mul=autoregressive_model_params.diffusion_batch_mul,
            grad_checkpointing=autoregressive_model_params.grad_checkpointing,
            predict_video=autoregressive_model_params.predict_video,
            act_diff_training_steps=self.autoregressive_model_params.act_diff_training_steps,
            act_diff_testing_steps=self.autoregressive_model_params.act_diff_testing_steps,
            action_model_params=action_model_params,
            use_history_action=kwargs["use_history_action"],
            action_mask_ratio=kwargs["action_mask_ratio"],
            use_proprioception=kwargs["use_proprioception"],
            predict_wrist_img=kwargs["predict_wrist_img"],
            different_history_freq=kwargs["different_history_freq"],
            predict_proprioception=kwargs["predict_proprioception"],
            task_name=self.task_name,
            language_emb_model=language_emb_model,
            shape_meta=shape_meta,
        )

        # =========================== load pretrained model ===========================
        self.pretrained_model_path = autoregressive_model_params.pretrained_model_path
        if self.pretrained_model_path is not None:
            if os.path.exists(self.pretrained_model_path):
                self.load_pretrained_model()
            else:
                print("pretrained model not found: ", self.pretrained_model_path)

        self.normalizer = LinearNormalizer()

        if self.selected_training_mode is None:
            if len(self.task_modes) == 0:
                self.task_modes = [
                    "video_model",
                    "dynamic_model",
                    "policy_model",
                    "inverse_model",
                    "full_dynamic_model",
                ]
        else:
            if self.selected_training_mode == "policy_model_full_dynamics_model":
                self.task_modes = ["policy_model", "full_dynamic_model"]
            else:
                self.task_modes = [self.selected_training_mode]
        print("----------------------------------------------------------------------")
        print("task_modes", self.task_modes)
        print("----------------------------------------------------------------------")

    def load_pretrained_model(self):
        print("----------------------------------------------------------------------")
        print("Loading pretrained model: ", self.pretrained_model_path)
        print("----------------------------------------------------------------------")

        pretrained_diffusion_model_ckpt = torch.load(
            self.pretrained_model_path, map_location="cpu", weights_only=False
        )

        if "state_dicts" in pretrained_diffusion_model_ckpt:
            if "ema_model" in pretrained_diffusion_model_ckpt["state_dicts"]:
                print("load from previous ema model")
                ## load from previous checkpoint
                pretrained_diffusion_model_ckpt_ = {
                    k[6:]: v
                    for k, v in pretrained_diffusion_model_ckpt["state_dicts"][
                        "ema_model"
                    ].items()
                    if k.startswith("model.")
                }  # remove 'model.'

                model_state_dict = self.model.state_dict()
                pretrained_state_dict = {
                    k: v
                    for k, v in pretrained_diffusion_model_ckpt_.items()
                    if k in model_state_dict and model_state_dict[k].size() == v.size()
                }
                
                pretrained_state_dict_mismatch = {
                    k: v
                    for k, v in model_state_dict.items()
                    if k not in pretrained_diffusion_model_ckpt_
                    or pretrained_diffusion_model_ckpt_[k].size() != v.size()
                }
                
                print("----------------------------------------------------------------------")
                print(
                    "pretrained_state_dict_mismatch: ",
                    pretrained_state_dict_mismatch.keys(),
                )
                print("----------------------------------------------------------------------")
                
                assert len(model_state_dict) > 0
                assert len(pretrained_state_dict) > 0
                model_state_dict.update(pretrained_state_dict)

                missing_keys, unexpected_keys = self.model.load_state_dict(
                    model_state_dict, strict=False
                )
            else:
                raise NotImplementedError

        elif "model_ema" in pretrained_diffusion_model_ckpt:
            ## load from MAR pretrained mdoel
            pretrained_diffusion_model_ckpt_ = pretrained_diffusion_model_ckpt[
                "model_ema"
            ]

            model_state_dict = self.model.state_dict()
            pretrained_state_dict = {
                k: v
                for k, v in pretrained_diffusion_model_ckpt_.items()
                if k in model_state_dict and model_state_dict[k].size() == v.size()
            }
            assert len(model_state_dict) > 0
            assert len(pretrained_state_dict) > 0
            model_state_dict.update(pretrained_state_dict)

            missing_keys, unexpected_keys = self.model.load_state_dict(
                model_state_dict, strict=False
            )

        else:
            raise NotImplementedError

        print("---------------------------------------------------------------")
        print("Model Missing keys:", missing_keys)
        print("Model Unexpected keys:", unexpected_keys)
        print("---------------------------------------------------------------")


    def predict_action(
        self, obs_dict: Dict[str, torch.Tensor], language_goal=None
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        
        obs_dict = resize_image_eval(self.task_name, obs_dict)
        B, T, C, H, W = obs_dict["image"].shape

        ## language goal
        text_latents = None
        if self.language_emb_model is not None:
            if "umi" in self.task_name:
                text_latents = language_goal
            else:
                print("predict_action language_goal: ", language_goal)
                print(self.task_name, "max_length", self.max_length)

                if self.language_emb_model == "clip":
                    text_tokens = self.tokenizer(
                        language_goal,
                        padding="max_length",
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    text_latents = extract_text_features(
                        self.text_model,
                        text_tokens,
                        language_emb_model=self.language_emb_model,
                    )
                else:
                    text_latents = None

        ## history action
        history_nactions = None
        if self.use_history_action:
            if "past_action" in obs_dict:
                history_nactions = normalize_past_action(
                    normalizer=self.normalizer,
                    normalizer_type=self.normalizer_type,
                    actions=obs_dict["past_action"],
                )
                del obs_dict["past_action"]

        ## normalize observations
        batch = normalize_obs(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            batch={"obs": obs_dict},
        )
        obs_dict = batch["obs"]

        c, proprioception_input, _ = process_data(
            {"obs": obs_dict}, task_name=self.task_name, eval=True, **self.kwargs
        )

        if self.use_student_tokenizer and self.student_tokenizer is not None:
            if self.use_proprioception and proprioception_input is not None:
                if "second_image" in proprioception_input:
                    second_image_z, _ = self._encode_student_latent(
                        proprioception_input["second_image"]
                    )
                    proprioception_input["second_image_z"] = second_image_z
            c, _ = self._encode_student_latent(c.detach())
        else:
            if self.use_proprioception and proprioception_input is not None:
                if "second_image" in proprioception_input:
                    second_image_z, _ = extract_latent_autoregressive(
                        self.vae_model, proprioception_input["second_image"]
                    )
                    proprioception_input["second_image_z"] = second_image_z
            c, _ = extract_latent_autoregressive(self.vae_model, c.detach())

        z, act_out = self.model.sample_tokens(
            bsz=B,
            cond=c,
            text_latents=text_latents,
            num_iter=self.autoregressive_model_params.num_iter,
            cfg=self.autoregressive_model_params.cfg,
            cfg_schedule=self.autoregressive_model_params.cfg_schedule,
            temperature=self.autoregressive_model_params.temperature,
            history_nactions=history_nactions,
            proprioception_input=proprioception_input,
            task_mode="policy_model",
            vae_model=self.vae_model,
        )

        # unnormalize prediction
        Da = self.action_dim

        naction_pred = act_out[..., :Da]

        action_pred = unnormalize_future_action(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            actions=naction_pred,
        )

        action = action_pred[:, : self.n_action_steps]

        result = {
            "action": action,
            "action_pred": action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def add_weight_decay(self, model, weight_decay=1e-5, skip_list=()):
        decay = []
        no_decay = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue  # frozen weights
            if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
                no_decay.append(param)  # no weight decay on bias, norm and diffloss
            else:
                decay.append(param)

        return [
            {"params": no_decay, "weight_decay": 0.0},
            {"params": decay, "weight_decay": weight_decay},
        ]

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
    ) -> torch.optim.Optimizer:
        optim_groups = []

        optim_groups.extend(self.add_weight_decay(self.model, weight_decay=weight_decay))

        if self.use_student_tokenizer and self.student_tokenizer is not None:
            optim_groups.extend(
                self.add_weight_decay(self.student_tokenizer, weight_decay=weight_decay)
            )

        if self.align_projector is not None:
            optim_groups.extend(
                self.add_weight_decay(self.align_projector, weight_decay=weight_decay)
            )

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)

        for param_group in optimizer.param_groups:
            if "initial_lr" not in param_group:
                param_group["initial_lr"] = param_group["lr"]

        return optimizer

    def _extract_teacher_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Teacher path (frozen VAE): x [B, C, T, H, W] -> z [B, T, C_lat, H_lat, W_lat]
        """
        x = x.float()
        B, C, T, H, W = x.size()
        with torch.no_grad():
            x_ = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            posterior = self.vae_model.encode(x_)
            if self.align_teacher_use_mode and hasattr(posterior, "mode"):
                z = posterior.mode()
            else:
                z = posterior.sample()
            z = z.mul_(0.2325)
            z = z.reshape(B, T, z.shape[1], z.shape[2], z.shape[3])
        return z

    def _latent_to_tokens(self, z: torch.Tensor) -> torch.Tensor:
        """
        z [B, T, C, H, W] -> [B, T, S, C]
        """
        B, T, C, H, W = z.shape
        return z.permute(0, 1, 3, 4, 2).reshape(B, T, H * W, C)

    def _encode_student_latent(self, x: torch.Tensor):
        """
        x [B, C, T, H, W] -> latent [B, T, C_lat, H_lat, W_lat], token_feat [B, T, S, D]
        """
        latent, token_feat = self.student_tokenizer(x)
        return latent, token_feat

    def _get_align_coeff(self) -> float:
        if not self.use_alignment:
            return 0.0
        if self._align_step < self.align_pretrain_steps:
            return 1.0

        # Smoothly decay coeff after alignment-only pretraining stage.
        if self.align_decay_steps <= 0:
            return self.align_coeff
        post_step = self._align_step - self.align_pretrain_steps
        progress = min(1.0, max(0.0, float(post_step) / float(self.align_decay_steps)))
        smooth = 0.5 * (1.0 + np.cos(np.pi * progress))
        start_coeff = self.align_post_start_coeff
        end_coeff = max(0.0, self.align_min_coeff)
        return end_coeff + (start_coeff - end_coeff) * smooth

    def _set_main_model_trainable(self, trainable: bool):
        if trainable and not self._main_model_frozen:
            return
        if (not trainable) and self._main_model_frozen:
            return
        for param in self.model.parameters():
            param.requires_grad = trainable
        self._main_model_frozen = not trainable

    def _compute_alignment_loss(
        self,
        student_tokens: torch.Tensor,
        teacher_tokens: torch.Tensor,
        return_metrics: bool = False,
    ):
        """
        student_tokens: [B, T, S, D_s] (from student tokenizer hidden features)
        teacher_tokens: [B, T, S, D_t] (from VAE latent tokens)
        """
        student_raw = student_tokens
        if self.align_projector is not None:
            student_tokens = self.align_projector(student_tokens)

        assert student_tokens.shape == teacher_tokens.shape, (
            student_tokens.shape,
            teacher_tokens.shape,
        )

        metrics = None
        if self.align_loss_type.lower() == "mse":
            loss = F.mse_loss(student_tokens, teacher_tokens)
            student_tokens_norm = F.normalize(student_tokens, dim=-1)
            teacher_tokens_norm = F.normalize(teacher_tokens, dim=-1)
            cosine = (student_tokens_norm * teacher_tokens_norm).sum(dim=-1).mean()
        else:
            # default: cosine (REPA style)
            student_tokens_norm = F.normalize(student_tokens, dim=-1)
            teacher_tokens_norm = F.normalize(teacher_tokens, dim=-1)
            cosine = (student_tokens_norm * teacher_tokens_norm).sum(dim=-1).mean()
            loss = 1.0 - cosine

        if return_metrics:
            with torch.no_grad():
                metrics = {
                    "cosine": cosine.detach(),
                    "student_raw_std": student_raw.detach().std(),
                    "student_std": student_tokens.detach().std(),
                    "teacher_std": teacher_tokens.detach().std(),
                    "student_norm": student_tokens.detach().norm(dim=-1).mean(),
                    "teacher_norm": teacher_tokens.detach().norm(dim=-1).mean(),
                }

        if return_metrics:
            return loss, metrics
        return loss


    def compute_loss(self, batch, **kwargs):
        B, T, C, H, W = batch["obs"]["image"].size()
        schedule_step = self._align_step
        if "global_step" in kwargs and kwargs["global_step"] is not None:
            schedule_step = max(schedule_step, int(kwargs["global_step"]))
            self._align_step = schedule_step

        text_latents = None
        if self.language_emb_model == "clip":
            if "language" in batch["obs"]:
                language_goal = batch["obs"]["language"]
                del batch["obs"]["language"]
                text_tokens = {
                    "input_ids": language_goal[:, 0].long()[:, 0],
                    "attention_mask": language_goal[:, 0].long()[:, 1],
                }
                text_latents = extract_text_features(
                    self.text_model,
                    text_tokens,
                    language_emb_model=self.language_emb_model,
                )
            elif "language_latents" in batch:
                text_latents = batch["language_latents"]
            else:
                raise NotImplementedError

        nactions = normalize_action(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            actions=batch["action"],
        )
        batch = normalize_obs(
            normalizer=self.normalizer,
            normalizer_type=self.normalizer_type,
            batch=batch,
        )

        if self.use_history_action:
            batch = dict_apply(batch, lambda x: x[:, 1:])

        x, proprioception_input, _ = process_data(
            batch, task_name=self.task_name, **self.kwargs
        )

        align_loss = torch.tensor(0.0, device=x.device)
        align_cos = torch.tensor(0.0, device=x.device)
        align_coeff_value = torch.tensor(0.0, device=x.device)
        align_student_norm = torch.tensor(0.0, device=x.device)
        align_teacher_norm = torch.tensor(0.0, device=x.device)
        if self.use_student_tokenizer and self.student_tokenizer is not None:
            c_img, x_img = torch.chunk(x, 2, dim=2)
            
            z, z_feat = self._encode_student_latent(x_img)
            c, c_feat = self._encode_student_latent(c_img)

            if proprioception_input is not None:
                if "second_image" in proprioception_input:
                    second_image_z, _ = self._encode_student_latent(proprioception_input["second_image"])
                    proprioception_input["second_image_z"] = second_image_z
                if "pred_second_image" in proprioception_input:
                    pred_second_image_z, _ = self._encode_student_latent(proprioception_input["pred_second_image"])
                    proprioception_input["pred_second_image_z"] = pred_second_image_z

            if self.use_alignment:
                teacher_z = self._extract_teacher_latent(x_img)
                teacher_c = self._extract_teacher_latent(c_img)

                teacher_z_tokens = self._latent_to_tokens(teacher_z)
                teacher_c_tokens = self._latent_to_tokens(teacher_c)
                assert z_feat.shape[:3] == teacher_z_tokens.shape[:3], (
                    z_feat.shape,
                    teacher_z_tokens.shape,
                )
                assert c_feat.shape[:3] == teacher_c_tokens.shape[:3], (
                    c_feat.shape,
                    teacher_c_tokens.shape,
                )

                align_loss_z, align_metrics_z = self._compute_alignment_loss(
                    z_feat, teacher_z_tokens, return_metrics=True
                )
                align_loss_c, align_metrics_c = self._compute_alignment_loss(
                    c_feat, teacher_c_tokens, return_metrics=True
                )
                align_loss = 0.5 * (align_loss_z + align_loss_c)
                align_cos = 0.5 * (
                    align_metrics_z["cosine"] + align_metrics_c["cosine"]
                )
                align_student_norm = 0.5 * (
                    align_metrics_z["student_norm"] + align_metrics_c["student_norm"]
                )
                align_teacher_norm = 0.5 * (
                    align_metrics_z["teacher_norm"] + align_metrics_c["teacher_norm"]
                )

                if (
                    self.align_debug
                    and self.training
                    and (schedule_step % max(1, self.align_debug_every) == 0)
                ):
                    print(
                        "[ALIGN DEBUG] "
                        f"step={schedule_step} "
                        f"z_feat={tuple(z_feat.shape)} teacher_z={tuple(teacher_z_tokens.shape)} "
                        f"c_feat={tuple(c_feat.shape)} teacher_c={tuple(teacher_c_tokens.shape)} "
                        f"loss={align_loss.item():.6f} cos={align_cos.item():.6f} "
                        f"s_norm={align_student_norm.item():.6f} t_norm={align_teacher_norm.item():.6f} "
                        f"s_std={0.5 * (align_metrics_z['student_std'] + align_metrics_c['student_std']):.6f} "
                        f"t_std={0.5 * (align_metrics_z['teacher_std'] + align_metrics_c['teacher_std']):.6f}"
                    )
        else:
             x, z, c, _, proprioception_input = get_vae_latent(
            x, self.vae_model, eval=False, proprioception_input=proprioception_input
        )

        align_only_stage = (
            self.training
            and self.use_student_tokenizer
            and self.use_alignment
            and schedule_step < self.align_pretrain_steps
        )
        self._set_main_model_trainable(not align_only_stage)

        if align_only_stage:
            loss = align_loss
            video_loss = torch.zeros_like(align_loss)
            act_loss = torch.zeros_like(align_loss)
        else:
            history_trajectory, trajectory = get_trajectory(
                nactions, T, self.shift_action, use_history_action=self.use_history_action
            )

            selected_mode = random.choice(self.task_modes)

            loss, video_loss, act_loss = self.model(
                z,
                c,
                history_trajectory,
                trajectory,
                text_latents,
                task_mode=selected_mode,
                proprioception_input=proprioception_input,
            )

        if self.use_student_tokenizer and self.use_alignment and not align_only_stage:
            coeff = self._get_align_coeff()
            loss = loss + coeff * align_loss
            align_coeff_value = torch.tensor(coeff, device=x.device, dtype=align_loss.dtype)
        elif align_only_stage:
            align_coeff_value = torch.tensor(1.0, device=x.device, dtype=align_loss.dtype)

        if self.use_student_tokenizer and self.use_alignment and self.training:
            self._align_step = schedule_step + 1

        ## not recommended, fix the problem in DDM unused parameters
        for param in self.model.parameters():
            if param.grad is None:  # Likely unused in loss computation
                loss += 0 * param.sum()

        if self.student_tokenizer is not None:
            for param in self.student_tokenizer.parameters():
                if param.grad is None:
                    loss += 0 * param.sum()

        if self.align_projector is not None:
            for param in self.align_projector.parameters():
                if param.grad is None:
                    loss += 0 * param.sum()

        return loss, (
            video_loss,
            act_loss,
            align_loss,
            align_cos,
            align_coeff_value,
            align_student_norm,
            align_teacher_norm,
        )

    def forward(self, batch, **kwargs):
        return self.compute_loss(batch, **kwargs)
