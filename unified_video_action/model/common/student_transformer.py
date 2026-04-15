import torch
import torch.nn as nn


class StudentTransformerBridge(nn.Module):
    """
    Lightweight replacement for MAR transformer trunk.
    Input: cond latent tokens [B, T, S, C_in]
    Output: transformer output tokens z [B, T*S, C_out]
    """

    def __init__(
        self,
        token_dim=16,
        hidden_dim=512,
        out_dim=768,
        depth=4,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.0,
    ):
        super().__init__()
        self.in_proj = nn.Linear(token_dim, hidden_dim)
        self.cond_proj = nn.Linear(token_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=int(hidden_dim * mlp_ratio),
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, out_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.constant_(self.in_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.cond_proj.weight)
        nn.init.constant_(self.cond_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, cond_tokens: torch.Tensor) -> torch.Tensor:
        """
        cond_tokens: [B, T, S, C_in]
        """
        bsz, t, s, _ = cond_tokens.shape
        x = cond_tokens.reshape(bsz, t * s, -1).float()
        x = self.in_proj(x) + self.cond_proj(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        z = self.out_proj(x)
        return z
