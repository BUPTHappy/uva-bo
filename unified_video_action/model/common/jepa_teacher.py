import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# torch.hub entry in facebookresearch/vjepa2 -> (embed_dim, default_img_size)
VJEPA2_HUB_MODELS = {
    "vjepa2_vit_large": (1024, 256),
    "vjepa2_vit_huge": (1280, 256),
    "vjepa2_vit_giant": (1408, 256),
    "vjepa2_vit_giant_384": (1408, 384),
    "vjepa2_1_vit_base_384": (768, 384),
    "vjepa2_1_vit_large_384": (1024, 384),
    "vjepa2_1_vit_giant_384": (1408, 384),
    "vjepa2_1_vit_gigantic_384": (1664, 384),
}

VJEPA2_CHECKPOINT_URLS = {
    "vjepa2_vit_large": "https://dl.fbaipublicfiles.com/vjepa2/vitl.pt",
    "vjepa2_vit_huge": "https://dl.fbaipublicfiles.com/vjepa2/vith.pt",
    "vjepa2_vit_giant": "https://dl.fbaipublicfiles.com/vjepa2/vitg.pt",
    "vjepa2_vit_giant_384": "https://dl.fbaipublicfiles.com/vjepa2/vitg-384.pt",
}


def _clean_vjepa_encoder_state_dict(state_dict):
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "").replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _load_vjepa2_encoder_from_checkpoint(
    encoder: nn.Module, checkpoint_path: str, checkpoint_key: str = "target_encoder"
):
    payload = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(payload, dict) and checkpoint_key in payload:
        encoder_state = payload[checkpoint_key]
    elif isinstance(payload, dict) and "encoder" in payload:
        encoder_state = payload["encoder"]
    else:
        encoder_state = payload
    encoder_state = _clean_vjepa_encoder_state_dict(encoder_state)
    msg = encoder.load_state_dict(encoder_state, strict=False)
    print(f"Loaded V-JEPA2 encoder weights from {checkpoint_path}")
    if msg.missing_keys:
        print("V-JEPA2 encoder missing keys:", msg.missing_keys)
    if msg.unexpected_keys:
        print("V-JEPA2 encoder unexpected keys:", msg.unexpected_keys)


class JEPATeacher(nn.Module):
    """
    Frozen V-JEPA 2 encoder teacher for latent-space alignment.

    Each frame is encoded as a 2-frame clip (tubelet_size=2) so we can reuse the
  video encoder frame-by-frame on [B, C, T, H, W] inputs.

    Input:
        x: [B, C, T, H, W] in [0, 1]
    Output:
        tokens: [B, T, S, D] patch tokens in JEPA latent space
    """

    def __init__(
        self,
        model_name: str = "vjepa2_vit_large",
        img_size: int = 256,
        tubelet_size: int = 2,
        checkpoint_path: Optional[str] = None,
        checkpoint_key: str = "target_encoder",
        loader: str = "torch_hub",
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = int(img_size)
        self.tubelet_size = int(tubelet_size)
        self.loader = str(loader).lower()
        self.checkpoint_key = checkpoint_key

        if model_name not in VJEPA2_HUB_MODELS:
            raise ValueError(
                f"Unsupported JEPA model_name={model_name!r}. "
                f"Expected one of {list(VJEPA2_HUB_MODELS.keys())}."
            )

        default_feat_dim, default_img_size = VJEPA2_HUB_MODELS[model_name]
        if self.img_size != default_img_size:
            print(
                f"Warning: JEPA model {model_name} is usually used at "
                f"{default_img_size}px, but img_size={self.img_size} was requested."
            )

        if self.loader == "torch_hub":
            hub_out = torch.hub.load("facebookresearch/vjepa2", model_name, pretrained=False)
            if isinstance(hub_out, tuple):
                self.encoder = hub_out[0]
            else:
                self.encoder = hub_out
            print(f"Initialized V-JEPA2 encoder via torch.hub: {model_name}")
            if checkpoint_path is not None and os.path.exists(checkpoint_path):
                _load_vjepa2_encoder_from_checkpoint(
                    self.encoder, checkpoint_path, checkpoint_key=self.checkpoint_key
                )
            else:
                url = VJEPA2_CHECKPOINT_URLS.get(model_name)
                if url is not None:
                    state = torch.hub.load_state_dict_from_url(url, map_location="cpu")
                    encoder_state = state.get(self.checkpoint_key, state.get("encoder", state))
                    encoder_state = _clean_vjepa_encoder_state_dict(encoder_state)
                    self.encoder.load_state_dict(encoder_state, strict=False)
                    print(f"Loaded V-JEPA2 encoder weights from {url}")
                else:
                    print(
                        f"No default checkpoint URL for {model_name}; "
                        "provide jepa_teacher_params.checkpoint_path."
                    )
        elif self.loader == "checkpoint":
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    "loader='checkpoint' requires an existing checkpoint_path."
                )
            hub_out = torch.hub.load("facebookresearch/vjepa2", model_name, pretrained=False)
            self.encoder = hub_out[0] if isinstance(hub_out, tuple) else hub_out
            _load_vjepa2_encoder_from_checkpoint(
                self.encoder, checkpoint_path, checkpoint_key=self.checkpoint_key
            )
        else:
            raise ValueError(
                f"Unsupported loader={loader!r}. Expected 'torch_hub' or 'checkpoint'."
            )

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.feat_dim = int(getattr(self.encoder, "embed_dim", default_feat_dim))
        self.patch_size = int(getattr(self.encoder, "patch_size", 16))
        self.grid_size = self.img_size // self.patch_size
        self.num_spatial_tokens = self.grid_size * self.grid_size

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False
        )

    def _normalize_video(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def _resize_video(self, x: torch.Tensor) -> torch.Tensor:
        bsz, channels, timesteps, height, width = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(bsz * timesteps, channels, height, width)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return x.view(bsz, timesteps, channels, self.img_size, self.img_size).permute(
            0, 2, 1, 3, 4
        )

    def _encode_clip(self, clip: torch.Tensor) -> torch.Tensor:
        # clip: [N, C, T_clip, H, W], T_clip must be >= tubelet_size
        tokens = self.encoder(clip)
        if tokens.shape[1] != self.num_spatial_tokens:
            tokens = self._resample_tokens(tokens, self.num_spatial_tokens)
        return tokens

    @staticmethod
    def _resample_tokens(tokens: torch.Tensor, target_tokens: int) -> torch.Tensor:
        batch_size, num_tokens, feat_dim = tokens.shape
        side = int(num_tokens**0.5)
        if side * side != num_tokens:
            raise ValueError(f"Token count {num_tokens} is not a perfect square.")
        target_side = int(target_tokens**0.5)
        if target_side * target_side != target_tokens:
            raise ValueError(f"Target token count {target_tokens} is not a perfect square.")

        token_map = tokens.transpose(1, 2).reshape(batch_size, feat_dim, side, side)
        token_map = F.interpolate(
            token_map,
            size=(target_side, target_side),
            mode="bilinear",
            align_corners=False,
        )
        return token_map.flatten(2).transpose(1, 2)

    @torch.no_grad()
    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        bsz, channels, timesteps, height, width = x.size()
        x = self._resize_video(x)
        x = self._normalize_video(x)

        # Encode each frame as a short clip for the video encoder (tubelet_size=2).
        frames = x.permute(0, 2, 1, 3, 4).reshape(
            bsz * timesteps, channels, self.img_size, self.img_size
        )
        clip = torch.stack([frames, frames], dim=2)
        tokens = self._encode_clip(clip)
        num_tokens = tokens.shape[1]
        feat_dim = tokens.shape[2]
        return tokens.reshape(bsz, timesteps, num_tokens, feat_dim)
