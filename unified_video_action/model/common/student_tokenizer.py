import torch
import torch.nn as nn

class StudentLatentTokenizer(nn.Module):
    """
    tokenizer used to replace VAE encoder

    Input: [B, C, T, H, W]
    Output: 
        latent:    [B, T, C_latent, H_patch, W_patch]
        token_feat:[B, T, S, D]  (for REPA-style alignment head)
    """
    def __init__(self, img_size=256, patch_size=16, in_channels=3, latent_channels=16, hidden_dim=256, depth=4, num_heads=8, mlp_ratio=4.0, dropout=0.0,):
        super().__init__()
        assert img_size % patch_size == 0, "image size must be divisible by patch size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.hidden_dim = hidden_dim

        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2

        self.patch_embed = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            bias=True,
        )

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, hidden_dim))

        blocks = []
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        for _ in range(depth):
            blocks.append(
                nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=mlp_hidden_dim,
                    dropout=dropout,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
            )

        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, latent_channels)

        self._init_weights()
    
    
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
           nn.init.constant_(self.patch_embed.bias, 0.0)
        
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
           nn.init.constant_(self.out_proj.bias, 0.0)
    

    def forward(self, x):
        """
        x:[B, C, T, H, W]
        """
        x=x.float()
        bsz, channels, timesteps, height, width = x.shape
        assert channels == self.in_channels, f"Expected {self.in_channels} channels, but got {channels}"
        assert height == self.img_size and width == self.img_size, f"Expected image size {self.img_size}, but got {height}x{width}"

        x = x.permute(0, 2, 1, 3, 4).reshape(bsz * timesteps, channels, height, width) # [B, C, T, H, W] -> [B*T, C, H, W] 
        x = self.patch_embed(x) # [B*T, D, H', W']
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
        token_feat = self.norm(x)

        token_latent = self.out_proj(token_feat)
        h_patch = self.grid_size
        w_patch = self.grid_size
        latent = token_latent.transpose(1,2).reshape(
          bsz*timesteps, self.latent_channels, h_patch, w_patch
        )
        latent = latent.reshape(bsz, timesteps, self.latent_channels, h_patch, w_patch)
        token_feat = token_feat.reshape(bsz, timesteps, self.num_patches, self.hidden_dim)
        return latent, token_feat

