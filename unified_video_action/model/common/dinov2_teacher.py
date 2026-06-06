import os

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    ):
        super().__init__()
        self.model_name = model_name
        self.img_size = int(img_size)

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.model = torch.hub.load(
                "facebookresearch/dinov2", model_name, pretrained=False
            )
            state_dict = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=True)
            print(f"Loaded DINOv2 teacher from local checkpoint: {checkpoint_path}")
        else:
            self.model = torch.hub.load("facebookresearch/dinov2", model_name)
            print(f"Loaded DINOv2 teacher via torch.hub: {model_name}")

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.feat_dim = int(self.model.embed_dim)
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
        tokens = features["x_norm_patchtokens"]
        num_tokens = tokens.shape[1]
        feat_dim = tokens.shape[2]
        return tokens.reshape(bsz, timesteps, num_tokens, feat_dim)
