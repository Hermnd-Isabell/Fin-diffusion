
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.autoencoder import VAE
from models.conditions import MarketConditioner
from models.diffusion import LatentDiT
from physics.sde_solver import LogSpaceGBM
from engine.trainer import PILCDMTrainer
from engine.sampler import PILCDMSampler

def test_engine():
    print("--- Testing Phase 4: Engine & PINA ---")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Instantiate Models (Small configs for testing)
    vae = VAE(input_dim=1, latent_dim=4).to(device)
    conditioner = MarketConditioner(condition_dim=3, embed_dim=128).to(device)
    dit = LatentDiT(latent_dim=4, spatial_size=4, embed_dim=128, depth=2).to(device)
    sde = LogSpaceGBM(num_timesteps=10)
    optimizer = optim.Adam(list(vae.parameters()) + list(dit.parameters()) + list(conditioner.parameters()), lr=1e-4)
    
    # 2. Mock Data Batch
    B = 2
    iv_surface = torch.rand(B, 1, 32, 32).to(device) * 0.5 + 0.1 # Realostic vol range
    conditions = torch.rand(B, 3).to(device)
    S = torch.tensor(3.0).to(device)
    r = torch.tensor(0.03).to(device)
    
    # Grids
    k_vals = torch.linspace(2.5, 3.5, 32).to(device)
    t_vals = torch.linspace(0.1, 1.0, 32).to(device)
    T_grid, K_grid = torch.meshgrid(t_vals, k_vals, indexing='ij')
    T_grid = T_grid.expand(B, 1, 32, 32)
    K_grid = K_grid.expand(B, 1, 32, 32)
    
    # 3. Test Trainer Step
    print("\n[Test 1] Training Step (Dynamic SNR + Physics Loss)")
    trainer = PILCDMTrainer(vae, conditioner, dit, sde, optimizer, device)
    
    logs = trainer.train_step(iv_surface, conditions, S, K_grid, T_grid, r)
    print(f"Training Logs: {logs}")
    
    # Check if loss is valid
    if np.isfinite(logs['loss_total']):
        print("PASS: Training step completed with finite loss.")
    else:
        print("FAIL: Loss is NaN/Inf!")
        return

    # 4. Test Sampler (PINA Corrector)
    print("\n[Test 2] PINA Sampler (Inference)")
    sampler = PILCDMSampler(vae, conditioner, dit, sde, device)
    
    # Sample with correction
    # Using small steps to be fast
    final_ivs = sampler.sample(conditions, S, K_grid, T_grid, r, num_steps=5, pina_lr=0.1)
    
    print(f"Generated IVS Shape: {final_ivs.shape}")
    print(f"Generated IVS Stats: Min={final_ivs.min().item():.4f}, Max={final_ivs.max().item():.4f}")
    
    if final_ivs.shape == (B, 1, 32, 32):
        print("PASS: Sampler output shape correct.")
    else:
        print("FAIL: Sampler output shape mismatch!")
    
    print("\n--- Phase 4 Verified Successfully ---")

if __name__ == "__main__":
    test_engine()
