
import torch
import torch.nn as nn

class MarketConditioner(nn.Module):
    """
    Embeds market conditions (e.g., iVIX, Historical Vol, Returns) into a context vector
    for Cross-Attention in the Diffusion Transformer.
    """
    def __init__(self, condition_dim=3, embed_dim=128):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Simple MLP to project scalar conditions to embedding dimension
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )
        
    def forward(self, conditions):
        """
        Args:
            conditions: (B, condition_dim) or (B, L, condition_dim)
        Returns:
            context: (B, 1, embed_dim) - Sequence length 1 for global context
        """
        if conditions.dim() == 2:
             conditions = conditions.unsqueeze(1) # (B, 1, condition_dim)
             
        # Normalize? Assuming inputs are standardized z-scores.
        
        embedding = self.mlp(conditions) # (B, 1, embed_dim)
        return embedding
