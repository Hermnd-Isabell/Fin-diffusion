
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse

# Project Modules
from data_factory.raw_loader import RawLoader
from data_factory.svi_calibrator import SVICalibrator
from data_factory.grid_mapper import GridMapper
from models.autoencoder import VAE
from models.conditions import MarketConditioner
from models.diffusion import LatentDiT
from physics.sde_solver import LogSpaceGBM
from engine.trainer import PILCDMTrainer

# Configuration
DATA_PATH = "50etf_options.xlsx"
PROCESSED_DATA_PATH = "data/processed_tensors.pt"
CHECKPOINT_DIR = "checkpoints"
os.makedirs("data", exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on {DEVICE}")

def process_data(args):
    """
    1. Load Raw Data
    2. Calibrate SVI for ALL dates
    3. Map to Grid
    4. Save Tensors
    """
    print("--- Starting Data Processing Pipeline ---")
    # Load all data first (Expensive!)
    print(f"Loading raw data from {DATA_PATH} (this might take a while)...")
    df = pd.read_excel(DATA_PATH)
    print(f"Loaded {len(df)} rows.")
    
    loader = RawLoader(DATA_PATH)
    loader.df = df # Optimization: Inject loaded DF to avoid reloading
    
    dates = sorted(df['trade_date'].unique())
    print(f"Found {len(dates)} unique trading dates.")
    
    # Optional Debug Limit
    if args.limit_data:
        # Try Recent Dates (more likely to have data)
        print(f"LIMITING DATA TO LAST {args.limit_data} DATES FOR DEMO")
        dates = dates[-args.limit_data:]
        print(f"Selected dates: {dates}")
        
    mapper = GridMapper()
    
    iv_surfaces = []
    conditions_list = []
    valid_dates = []
    
    # Batch processing
    # Disable tqdm for debug
    for i, date in enumerate(dates):
        try:
            if i % 10 == 0:
                print(f"Processing date {i}/{len(dates)}: {date}")
            
            # date is likely int (YYYYMMDD) from unique()
            date_str = str(date)
            # Verify if it needs formatting? 
            # If date is Timestamp, strftime is needed. If int, str() is enough.
            # Safe way:
            if isinstance(date, (int, np.integer)):
                date_str = str(date)
            else:
                date_str = pd.to_datetime(date).strftime('%Y%m%d')
            
            # 1. Clean Data
            daily_data = loader.clean_data(target_date=date_str)
            if daily_data is None or daily_data.empty: 
                print(f"No data after cleaning for {date_str}")
                continue
            
            print(f"Date {date_str}: Found {len(daily_data)} rows. S: {daily_data['S'].iloc[0]:.2f}")
            
            spot = daily_data['S'].iloc[0]
            r = 0.03 # Fixed for now, can be dynamic
            
            # 2. Fit SVI for each expiry
            expiries = daily_data['expiry'].unique()
            daily_svi_params = []
            
            for expiry in expiries:
                slice_data = daily_data[daily_data['expiry'] == expiry]
                
                # Expiry is likely int (YYYYMMDD)
                expiry_str = str(expiry)
                date_str_fmt = str(date) # ensure date is str
                
                try:
                    T_date = pd.to_datetime(expiry_str, format='%Y%m%d')
                    Current_date = pd.to_datetime(date_str_fmt, format='%Y%m%d')
                    T = (T_date - Current_date).days / 365.0
                except:
                    # Fallback if format is different
                    T_date = pd.to_datetime(expiry_str)
                    Current_date = pd.to_datetime(date_str_fmt)
                    T = (T_date - Current_date).days / 365.0
                
                if T < 0.02: 
                    # print(f"Skipping expiry {expiry} (T={T:.4f})")
                    continue 
                
                calibrator = SVICalibrator(
                    strikes=slice_data['strike'].values,
                    ivs=slice_data['iv'].values,
                    T=T, S=spot, r=r
                )
                
                params, error = calibrator.calibrate()
                if params is not None:
                    daily_svi_params.append((expiry, T, params))
            
            if not daily_svi_params: 
                print(f"Date {date_str}: SVI Calibration failed for all expiries.")
                continue
            
            # 3. Grid Mapping
            # (32, 32)
            surface = mapper.interpolate_surface(daily_svi_params, date)
            
            # 4. Feature Engineering (Conditions)
            # Placeholder: [Spot, iVIX_proxy, Slope]
            # Ideally fetch real iVIX. calculating simple proxy:
            atm_vol = surface[0, 16] # Short-term ATM
            slope = surface[-1, 16] - surface[0, 16] # Term structure
            cond = [spot, atm_vol, slope]
            
            iv_surfaces.append(surface)
            conditions_list.append(cond)
            valid_dates.append(date_str)
            print(f"Date {date_str}: SUCCESS")
            
        except Exception as e:
            print(f"Error processing {date}: {e}")
            continue

    # Stack
    iv_tensor = torch.tensor(np.array(iv_surfaces), dtype=torch.float32).unsqueeze(1) # (N, 1, 32, 32)
    cond_tensor = torch.tensor(np.array(conditions_list), dtype=torch.float32) # (N, 3)
    
    print(f"Successfully processed {len(valid_dates)} dates.")
    print(f"IV Tensor: {iv_tensor.shape}")
    print(f"Conditions: {cond_tensor.shape}")
    
    torch.save({
        'iv_surface': iv_tensor,
        'conditions': cond_tensor,
        'dates': valid_dates
    }, PROCESSED_DATA_PATH)
    print(f"Saved processed data to {PROCESSED_DATA_PATH}")

def pretrain_vae(dataloader, epochs=50):
    print("\n--- Pre-training VAE ---")
    vae = VAE(input_dim=1, latent_dim=4).to(DEVICE)
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    
    vae.train()
    for epoch in range(epochs):
        total_loss = 0
        mse_loss = 0
        kld_loss = 0
        
        for batch_idx, (iv, _) in enumerate(dataloader):
            iv = iv.to(DEVICE)
            
            optimizer.zero_grad()
            recons, input_x, mu, log_var = vae(iv)
            
            # VAE Loss
            losses = vae.loss_function(recons, input_x, mu, log_var, kld_weight=0.00025)
            loss = losses['loss']
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            mse_loss += losses['Reconstruction_Loss'].item()
            kld_loss += losses['KLD'].item()
            
        if (epoch+1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | MSE: {mse_loss/len(dataloader):.6f} | KLD: {kld_loss/len(dataloader):.6f}")
            
    # Save VAE
    torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, "vae_pretrained.pth"))
    print("VAE Pre-training Complete. Saved to checkpoints/vae_pretrained.pth")
    return vae

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--process_data', action='store_true', help='Run data processing pipeline')
    parser.add_argument('--pretrain_vae', action='store_true', help='Run VAE pre-training')
    parser.add_argument('--train_full', action='store_true', help='Run full PI-LCDM training')
    parser.add_argument('--limit_data', type=int, default=0, help='Limit number of dates to process')
    args = parser.parse_args()
    
    # 1. Data Processing
    if args.process_data or not os.path.exists(PROCESSED_DATA_PATH):
        # We need to pass args to process_data, but process_data doesn't take args currently.
        # Let's make args global or pass it?
        # Better: Pass args to process_data
        process_data(args)
        
    # 2. Load Data
    print(f"Loading data from {PROCESSED_DATA_PATH}...")
    data = torch.load(PROCESSED_DATA_PATH)
    iv_tensor = data['iv_surface']
    cond_tensor = data['conditions']
    
    # Normalize Conditions?
    cond_mean = cond_tensor.mean(dim=0)
    cond_std = cond_tensor.std(dim=0)
    cond_tensor = (cond_tensor - cond_mean) / (cond_std + 1e-8)
    
    dataset = TensorDataset(iv_tensor, cond_tensor)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    # 3. VAE Pre-training
    vae = None
    if args.pretrain_vae:
        vae = pretrain_vae(dataloader)
    else:
        # Load if exists
        vae_path = os.path.join(CHECKPOINT_DIR, "vae_pretrained.pth")
        vae = VAE(input_dim=1, latent_dim=4).to(DEVICE)
        if os.path.exists(vae_path):
            print("Loading pre-trained VAE...")
            vae.load_state_dict(torch.load(vae_path, map_location=DEVICE))
        else:
            print("Warning: No pre-trained VAE found. Random initialization.")
            
    # 4. Full Training (DiT + VAE + Physics)
    if args.train_full:
        print("\n--- Starting Full PI-LCDM Training ---")
        conditioner = MarketConditioner(condition_dim=3, embed_dim=128).to(DEVICE)
        dit = LatentDiT(latent_dim=4, spatial_size=4, embed_dim=128, depth=4).to(DEVICE)
        sde = LogSpaceGBM(num_timesteps=1000)
        
        # Optimizer
        optimizer = optim.AdamW(
            list(vae.parameters()) + list(dit.parameters()) + list(conditioner.parameters()), 
            lr=2e-4, weight_decay=1e-5
        )
        
        trainer = PILCDMTrainer(vae, conditioner, dit, sde, optimizer, device=DEVICE)
        
        # Dummy Grids for Physics Loss (Should be properly mapped in real scenario)
        # Assuming fixed grid for all samples for simplicity now, or passed in batch
        # In reality, K/T depend on the day.
        # Approximation: Use standardized grid (Moneyness/Tenor) for gradients.
        # Physics loss checks convexity on the *Output Grid*.
        # Since input is normalized grid, we can enforce convexity on normalized grid directly?
        # Or we need to map back.
        # Let's use standard grids (2.5-3.5 etc) as placeholders for the Loss function constraints.
        # The constraints (Convexity) are property of the shape, not absolute level.
        
        k_vals = torch.linspace(-0.3, 0.3, 32).to(DEVICE) # Log-Moneyness k = log(K/S)
        t_vals = torch.linspace(0.05, 1.0, 32).to(DEVICE) # Years
        T_grid_fixed, k_grid_out = torch.meshgrid(t_vals, k_vals, indexing='ij')
        
        S_fixed = torch.tensor(1.0).to(DEVICE) # Normalized S=1
        r_fixed = torch.tensor(0.0).to(DEVICE) 
        
        # Convert log-moneyness to Strike: K = S * exp(k)
        K_grid_fixed = S_fixed * torch.exp(k_grid_out)
        
        # Training Loop
        EPOCHS = 100
        for epoch in range(EPOCHS):
            epoch_loss = 0
            
            for iv, cond in dataloader:
                iv = iv.to(DEVICE)
                cond = cond.to(DEVICE)
                
                # Expand grids to batch
                B = iv.shape[0]
                T_g = T_grid_fixed.expand(B, 1, 32, 32)
                K_g = K_grid_fixed.expand(B, 1, 32, 32)
                
                logs = trainer.train_step(iv, cond, S_fixed, K_g, T_g, r_fixed)
                epoch_loss += logs['loss_total']
                
            print(f"Epoch {epoch+1}/{EPOCHS} | Total Loss: {epoch_loss/len(dataloader):.6f}")
            
            if (epoch+1) % 10 == 0:
                torch.save(dit.state_dict(), os.path.join(CHECKPOINT_DIR, f"dit_epoch_{epoch+1}.pth"))
                torch.save(vae.state_dict(), os.path.join(CHECKPOINT_DIR, f"vae_epoch_{epoch+1}.pth"))
                torch.save(conditioner.state_dict(), os.path.join(CHECKPOINT_DIR, f"conditioner_epoch_{epoch+1}.pth"))
                
        print("Full Training Complete.")

if __name__ == "__main__":
    main()
