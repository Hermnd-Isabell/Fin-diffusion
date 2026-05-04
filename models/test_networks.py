
import torch
import numpy as np
from models.autoencoder import VAE
from models.conditions import MarketConditioner
from models.diffusion import LatentDiT

def test_neural_backbone():
    print("--- Testing Neural Backbone (Phase 3) ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Test VAE
    print("\n[Test 1] VAE (Compression)")
    # Input: (B, 1, 32, 32)
    x = torch.randn(2, 1, 32, 32).to(device)
    vae = VAE(input_dim=1, latent_dim=4).to(device)
    
    # Forward
    recons, input_x, mu, log_var = vae(x)
    
    print(f"Input Shape: {x.shape}")
    print(f"Recons Shape: {recons.shape}")
    print(f"Latent Mu Shape: {mu.shape}") # Should be (B, 4, 4, 4)
    
    if recons.shape == x.shape and mu.shape == (2, 4, 4, 4):
        print("PASS: VAE Shapes Correct.")
    else:
        print("FAIL: VAE Shapes Mismatch!")
        return

    # 2. Test Market Conditioner
    print("\n[Test 2] Market Conditioner")
    # Input: (B, 3) scalar conditions
    conds = torch.randn(2, 3).to(device)
    conditioner = MarketConditioner(condition_dim=3, embed_dim=128).to(device)
    
    context = conditioner(conds)
    print(f"Condition Input: {conds.shape}")
    print(f"Context Output: {context.shape}") # Should be (B, 1, 128)
    
    if context.shape == (2, 1, 128):
        print("PASS: Conditioner Shapes Correct.")
    else:
        print("FAIL: Conditioner Shapes Mismatch!")
        return

    # 3. Test Latent DiT
    print("\n[Test 3] Latent DiT (Generator)")
    # Input: Latent (B, 4, 4, 4)
    z = torch.randn(2, 4, 4, 4).to(device)
    t = torch.tensor([[0.5], [0.8]]).to(device) # (B, 1)
    
    dit = LatentDiT(latent_dim=4, spatial_size=4, embed_dim=128, depth=2, num_heads=4).to(device)
    
    noise_pred = dit(z, t, context)
    
    print(f"Latent Input: {z.shape}")
    print(f"Noise Pred: {noise_pred.shape}")
    
    if noise_pred.shape == z.shape:
        print("PASS: DiT Output matches Latent Input.")
    else:
        print("FAIL: DiT Shape Mismatch!")
        return

    print("\n--- All Neural Modules Verified ---")

if __name__ == "__main__":
    test_neural_backbone()
