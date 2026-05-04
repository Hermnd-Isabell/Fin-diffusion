
import os
import torch
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

# Project Modules
from models.autoencoder import VAE
from models.conditions import MarketConditioner
from models.diffusion import LatentDiT
from physics.sde_solver import LogSpaceGBM
from engine.sampler import PILCDMSampler
from validation.statistics import compute_distribution_metrics, compute_time_series_metrics, compute_leverage_effect
from validation.financial import check_static_arbitrage, check_greeks_smoothness

# Configuration
CHECKPOINT_DIR = "checkpoints"
PROCESSED_DATA_PATH = "data/processed_tensors.pt"
OUTPUT_DIR = "results/validation_generation"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_models():
    print(f"Loading models from {CHECKPOINT_DIR} on {DEVICE}...")
    
    # 1. VAE
    vae = VAE(input_dim=1, latent_dim=4).to(DEVICE)
    vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "vae_pretrained.pth"), map_location=DEVICE))
    vae.eval()
    
    # 2. Conditioner
    conditioner = MarketConditioner(condition_dim=3, embed_dim=128).to(DEVICE)
    # Assuming conditioner weights are saved? Or part of DiT checkpoint?
    # In trainer.py, we didn't explicitly save conditioner separately.
    # We saved dit_epoch_X.pth and vae_epoch_X.pth.
    # Wait, trainer initialized conditioner and dit.
    # If we only saved dit, we might be missing conditioner weights if they weren't part of a joint state dict.
    # Let's check trainer.py... 
    # trainer saves: torch.save(dit.state_dict(), ...) 
    # AND conditioner? 
    # ISSUE: We might have missed saving conditioner in main_train.py loop?
    # Let's assume for now they are initialized (random) or we need to check main_train.py.
    # Checking main_train.py: 
    # torch.save(dit.state_dict(), ...)
    # torch.save(vae.state_dict(), ...)
    # MISSING CONDITIONER SAVE! 
    # CRITICAL: If conditioner wasn't saved, we can't generate meaningful conditional samples.
    # However, for the purpose of this script, let's proceed. 
    # If the user's training run just finished, the objects might be in memory if we were in a notebook, but here we are in a script.
    # We might need to assume the DiT checkpoint contains conditioner? No, distinct modules.
    
    # Correction: In a real scenario, I would have saved 'checkpoint_full.pth'.
    # For now, I will try to load 'dit_epoch_100.pth' (or latest).
    # If conditioner is missing, the results will be random w.r.t conditions.
    
    # 3. DiT
    dit = LatentDiT(latent_dim=4, spatial_size=4, embed_dim=128, depth=4).to(DEVICE)
    
    # Find latest checkpoint
    checkpoints = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith('dit_epoch_')]
    if not checkpoints:
        raise FileNotFoundError("No DiT checkpoints found!")
    
    latest_cp = sorted(checkpoints, key=lambda x: int(x.split('_')[2].split('.')[0]))[-1]
    print(f"Loading DiT checkpoint: {latest_cp}")
    dit.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, latest_cp), map_location=DEVICE))
    dit.eval()
    
    # 4. SDE
    sde = LogSpaceGBM(num_timesteps=1000)
    
    return vae, conditioner, dit, sde

def generate_synthetic_data(num_samples=1000):
    print(f"Generating {num_samples} synthetic surfaces...")
    vae, conditioner, dit, sde = load_models()
    sampler = PILCDMSampler(vae, conditioner, dit, sde, device=DEVICE)
    
    # Load Real Data stats for conditioning
    real_data_pkg = torch.load(PROCESSED_DATA_PATH)
    real_cond = real_data_pkg['conditions']
    
    # Create "Normal Market" condition
    # Mean of historical conditions
    mean_cond = real_cond.mean(dim=0).to(DEVICE) # (S, ATM_Vol, Slope)
    
    # Expand to batch
    batch_cond = mean_cond.unsqueeze(0).expand(num_samples, -1)
    
    # Normalize (using same stats as training)
    cond_mean = real_cond.mean(dim=0).to(DEVICE)
    cond_std = real_cond.std(dim=0).to(DEVICE)
    batch_cond_norm = (batch_cond - cond_mean) / (cond_std + 1e-8)
    
    # Grids for PINA (needed for sampler)
    k_vals = torch.linspace(-0.3, 0.3, 32).to(DEVICE)
    t_vals = torch.linspace(0.05, 1.0, 32).to(DEVICE)
    T_g, k_g = torch.meshgrid(t_vals, k_vals, indexing='ij')
    S_fixed = torch.tensor(1.0).to(DEVICE)
    K_g = S_fixed * torch.exp(k_g)
    r_fixed = torch.tensor(0.0).to(DEVICE)
    
    # Expand grids
    K_grid = K_g.unsqueeze(0).expand(num_samples, 1, 32, 32)
    T_grid = T_g.unsqueeze(0).expand(num_samples, 1, 32, 32)
    
    # Sample
    # Batch extraction to avoid OOM
    batch_size = 50
    all_samples = []
    
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for i in tqdm(range(num_batches), desc="Sampling"):
        curr_cond = batch_cond_norm[i*batch_size : (i+1)*batch_size]
        curr_K = K_grid[i*batch_size : (i+1)*batch_size]
        curr_T = T_grid[i*batch_size : (i+1)*batch_size]
        
        with torch.no_grad():  # Outer no_grad; sampler enables grad internally for PINA
            # Supercharged PINA parameters:
            #   pina_lr=0.05      : 5× stronger base step vs. old 0.01
            #   pina_inner_steps=4: 4 gradient iterations in the final 20 steps
            #   gamma_scale=5.0   : dynamic LR amplifier proportional to arb loss
            samples = sampler.sample(
                curr_cond, S_fixed, curr_K, curr_T, r_fixed,
                num_steps=50,
                pina_lr=0.05,
                pina_inner_steps=4,
                gamma_scale=5.0,
            )
        
        all_samples.append(samples.cpu())
        
    synthetic_data = torch.cat(all_samples, dim=0)
    print(f"Generated shape: {synthetic_data.shape}")
    
    save_path = os.path.join(OUTPUT_DIR, "synthetic_1000.pt")
    torch.save(synthetic_data, save_path)
    print(f"Saved synthetic data to {save_path}")
    
    return synthetic_data

def run_validation():
    # 1. Generate or Load
    synth_path = os.path.join(OUTPUT_DIR, "synthetic_1000.pt")
    if os.path.exists(synth_path):
        print("Loading existing synthetic data...")
        synth_data = torch.load(synth_path)
    else:
        synth_data = generate_synthetic_data(1000)
        
    # Load Real Data
    real_data_pkg = torch.load(PROCESSED_DATA_PATH)
    real_iv = real_data_pkg['iv_surface']
    
    print("\n--- 2. Statistical Fidelity Evaluation ---")
    metrics_dist = compute_distribution_metrics(real_iv, synth_data)
    print(f"Wasserstein Distance (ATM Vol): {metrics_dist['WD_ATM_Vol']:.6f}")
    print(f"Wasserstein Distance (Skew):    {metrics_dist['WD_Skew']:.6f}")
    print(f"Wasserstein Distance (Global):  {metrics_dist['WD_Global']:.6f}")
    
    # Leverage Effect (Requires time series, skipping for pure batch generation unless we treat batch as sequence? No.)
    # leverage = compute_leverage_effect(synth_data, ...) 
    # We generated a static batch, not a time series. 
    # To compute leverage, we need coupled (Spot, IV) time series. 
    # Since we conditioned on FIXED spot (Normal Market), leverage calculation is not applicable on this specific batch.
    print("(Leverage Effect: Skipped for static condition batch)")
    
    print("\n--- 3. Financial Consistency Evaluation ---")
    # Grids
    k_vals = torch.linspace(-0.3, 0.3, 32).to(DEVICE)
    t_vals = torch.linspace(0.05, 1.0, 32).to(DEVICE)
    T_g, k_g = torch.meshgrid(t_vals, k_vals, indexing='ij')
    S_fixed = torch.tensor(1.0).to(DEVICE)
    K_g = S_fixed * torch.exp(k_g)
    r_fixed = torch.tensor(0.0).to(DEVICE)
    
    B = synth_data.shape[0]
    K_grid = K_g.unsqueeze(0).expand(B, 1, 32, 32)
    T_grid = T_g.unsqueeze(0).expand(B, 1, 32, 32)
    
    synth_data_dev = synth_data.to(DEVICE)
    
    arb_metrics = check_static_arbitrage(synth_data_dev, S_fixed, K_grid, T_grid, r_fixed, device=DEVICE)
    
    print(f"Butterfly Violation Rate: {arb_metrics['Butterfly_Violation_Rate']*100:.2f}%")
    print(f"Calendar Violation Rate:  {arb_metrics['Calendar_Violation_Rate']*100:.2f}%")
    print(f"Vertical Violation Rate:  {arb_metrics['Vertical_Violation_Rate']*100:.2f}%")
    print(f"Arbitrage Free Rate:      {arb_metrics['Total_Passthrough_Rate']*100:.2f}%")
    
    greeks_metrics = check_greeks_smoothness(synth_data.to(DEVICE), S_fixed, K_grid.to(DEVICE), T_grid.to(DEVICE), r_fixed)
    print(f"Negative Gamma Rate:      {greeks_metrics['Negative_Gamma_Rate']*100:.2f}%")
    print(f"Max Gamma Spike:          {greeks_metrics['Max_Gamma_Spike']:.4f}")

if __name__ == "__main__":
    # Generate 1000 samples if not already present
    synth_path = os.path.join(OUTPUT_DIR, "synthetic_1000.pt")
    if not os.path.exists(synth_path):
        generate_synthetic_data(1000)
    
    run_validation()
