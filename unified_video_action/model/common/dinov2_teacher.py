import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Official torch.hub names -> timm model names (Python 3.9 compatible).
DINOV2_HUB_TO_TIMM = {
    "dinov2_vits14": "vit_small_patch14_dinov2.lvd142m",
    "dinov2_vitb14": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_vitl14": "vit_large_patch14_dinov2.lvd142m",
    "dinov2_vitg14": "vit_giant_patch14_dinov2.lvd142m",
}


def _load_dinov2_torch_hub(model_name: str, pretrained: bool = True):
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "torch.hub DINOv2 requires Python 3.10+ due to upstream type hints. "
            "Use loader='timm' (default) on Python 3.9."
        )
    return torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)


def _load_dinov2_timm(model_name: str, pretrained: bool = True):
    import timm

    if model_name in DINOV2_HUB_TO_TIMM:
        timm_name = DINOV2_HUB_TO_TIMM[model_name]
    elif model_name in DINOV2_HUB_TO_TIMM.values():
        timm_name = model_name
    else:
        raise ValueError(
            f"Unsupported DINOv2 model_name={model_name!r}. "
            f"Expected one of {list(DINOV2_HUB_TO_TIMM.keys())} "
            f"or timm names {list(DINOV2_HUB_TO_TIMM.values())}."
        )

    model = timm.create_model(timm_name, pretrained=pretrained, num_classes=0)
    print(f"Loaded DINOv2 teacher via timm: {timm_name}")
    return model


def _extract_patch_tokens(model, features):
    if isinstance(features, dict):
        if "x_norm_patchtokens" in features:
            return features["x_norm_patchtokens"]
        if "x_norm_clstoken" in features:
            raise ValueError("DINOv2 features dict missing x_norm_patchtokens.")
    prefix_tokens = int(getattr(model, "num_prefix_tokens", 1))
    return features[:, prefix_tokens:, :]


class DINOv2Teacher(nn.Module):
    """
    Frozen DINOv2 teacher for student-token alignment.

    Input:
        x: [B, C, T, H, W] in [0, 1]
    Output:
        tokens: [B, T, S, D] patch tokens (x_norm_patchtokens)
    """

    def __init__(
        self,
        model_name: str = "dinov2_vits14",
        img_size: int = 224,
        checkpoint_path: str = None,
        loader: str = "timm",
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = int(img_size)
        self.loader = str(loader).lower()

        if self.loader not in ("timm", "torch_hub"):
            raise ValueError(
                f"Unsupported loader={loader!r}. Expected 'timm' or 'torch_hub'."
            )

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            if self.loader == "timm":
                self.model = _load_dinov2_timm(model_name, pretrained=False)
            else:
                self.model = _load_dinov2_torch_hub(model_name, pretrained=False)
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(f"Loaded DINOv2 teacher from local checkpoint: {checkpoint_path}")
            if msg.missing_keys:
                print("DINOv2 missing keys:", msg.missing_keys)
            if msg.unexpected_keys:
                print("DINOv2 unexpected keys:", msg.unexpected_keys)
        elif self.loader == "timm":
            self.model = _load_dinov2_timm(model_name, pretrained=True)
        else:
            self.model = _load_dinov2_torch_hub(model_name, pretrained=True)
            print(f"Loaded DINOv2 teacher via torch.hub: {model_name}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.feat_dim = int(
            getattr(self.model, "embed_dim", getattr(self.model, "num_features", None))
        )
        if self.feat_dim is None:
            raise AttributeError("Could not infer DINOv2 feature dimension.")

        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    @torch.no_grad()
    def extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        bsz, channels, timesteps, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(bsz * timesteps, channels, height, width)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.mean) / self.std
        features = self.model.forward_features(x)
        tokens = _extract_patch_tokens(self.model, features)
        num_tokens = tokens.shape[1]
        feat_dim = tokens.shape[2]
        return tokens.reshape(bsz, timesteps, num_tokens, feat_dim)
