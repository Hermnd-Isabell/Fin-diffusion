
import torch
import torch.nn as nn
import math

class DiTBlock(nn.Module):
    """
    Diffusion Transformer Block with Self-Attention and Cross-Attention.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        # Self-Attention
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # Cross-Attention
        x = x + self.cross_attn(self.norm2(x), context, context)[0]
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x

class LatentDiT(nn.Module):
    """
    Latent Diffusion Transformer.
    Operates on (4, 4) Latent patches.
    """
    def __init__(self, latent_dim=4, spatial_size=4, embed_dim=128, depth=4, num_heads=4, condition_dim=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.spatial_size = spatial_size
        self.embed_dim = embed_dim
        
        # 1. Patchify / Linear Projection
        # Input is (B, latent_dim, spatial, spatial)
        # Flatten to (B, spatial*spatial, latent_dim) = (B, 16, 4)
        # Project to embed_dim
        self.patch_proj = nn.Linear(latent_dim, embed_dim)
        
        # 2. Positional Embedding (Learnable or Fixed)
        self.num_patches = spatial_size * spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        # 3. Time Embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, num_heads) for _ in range(depth)
        ])
        
        # 5. Output Projection
        self.norm_final = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, latent_dim)
        
    def forward(self, x, t, context):
        """
        x: (B, latent_dim, H, W) e.g., (B, 4, 4, 4)
        t: (B, 1) Timestep
        context: (B, 1, embed_dim) Market conditions
        """
        B, C, H, W = x.shape
        
        # Flatten spatial dims: (B, C, H*W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        
        # Linear Projection
        x = self.patch_proj(x) # (B, 16, embed_dim)
        
        # Add Positional Embedding
        x = x + self.pos_embed
        
        # Add Time Embedding (Broadcast)
        t_emb = self.time_embed(t).unsqueeze(1) # (B, 1, embed_dim)
        x = x + t_emb
        
        # Transformer Blocks
        for block in self.blocks:
            x = block(x, context)
            
        # Output Projection
        x = self.norm_final(x)
        x = self.output_proj(x) # (B, 16, latent_dim)
        
        # Reshape back to image
        x = x.transpose(1, 2).view(B, C, H, W)
        
        return x
