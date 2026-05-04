
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Imports from other modules
from physics.market_ops import BSModule
from physics.arbitrage_loss import ArbitrageLoss

class PILCDMTrainer:
    """
    Physics-Informed Latent Conditional Diffusion Model Trainer.
    Implements Dynamic SNR Weighting for Physics Loss.
    """
    def __init__(self, vae, conditioner, dit, sde, optimizer, device='cpu'):
        self.vae = vae
        self.conditioner = conditioner
        self.dit = dit
        self.sde = sde
        self.optimizer = optimizer
        self.device = device
        
        # Physics Loss Module
        self.arb_loss_fn = ArbitrageLoss().to(device)
        
    def train_step(self, iv_surface, conditions, S, K_grid, T_grid, r):
        """
        Single training step.
        Args:
            iv_surface: (B, 1, 32, 32)
            conditions: (B, 3)
            S, K_grid, T_grid, r: Market parameters for Physics Loss
        """
        self.optimizer.zero_grad()
        
        # 1. VAE Encoding -> Latent z_0
        with torch.no_grad():
            mu, log_var = self.vae.encode(iv_surface)
            z_0 = self.vae.reparameterize(mu, log_var)
            
        # 2. Add Noise (Forward Diffusion)
        # Sample random t
        B = iv_surface.shape[0]
        t = torch.rand(B, 1, device=self.device)
        
        # Add SDE noise
        noise = torch.randn_like(z_0)
        z_t = self.sde.q_sample(z_0, t, noise)
        
        # 3. Condition Embedding
        context = self.conditioner(conditions)
        
        # 4. DiT Prediction
        # DEBUG SHAPES
        # print(f"DEBUG: z_t shape: {z_t.shape}")
        noise_pred = self.dit(z_t, t, context)
        
        # 5. DSM Loss (MSE)
        loss_dsm = nn.functional.mse_loss(noise_pred, noise)
        
        # 6. Physics-Informed Loss
        # We need to estimate z_0 from z_t and noise_pred to calculate Physics Loss
        # z_t = z_0 + std(t) * noise
        # z_0_hat = z_t - std(t) * noise_pred
        # (Assuming mean(t) = z_0 for simple additive noise in log-space)
        
        # Get std(t) for reconstruction
        _, std_t = self.sde.marginal_prob(z_t, t) # std_t shape (B, 1, 1, 1)
        z_0_hat = z_t - std_t * noise_pred
        
        # Decode to IVS Surface
        # VAE Decode: z -> IVS
        ivs_hat = self.vae.decode(z_0_hat)
        
        # Stability: Sanitize IVS for Physics Loss
        # Prevent NaNs/Infs from crashing Black-Scholes
        ivs_hat = torch.nan_to_num(ivs_hat, nan=0.0, posinf=5.0, neginf=0.0)
        ivs_hat = torch.clamp(ivs_hat, min=1e-4, max=5.0) # sigma in [0.01%, 500%]
        
        # Compute Arbitrage Loss
        loss_arb, _ = self.arb_loss_fn(ivs_hat, S, K_grid, T_grid, r)
        
        # 7. Dynamic SNR Weighting
        # SNR(t) = mean^2 / var = 1 / std(t)^2 (approx, assuming signal is unit variance)
        # gamma(t) = SNR / (SNR + 1) = 1 / (1 + std(t)^2)
        # As t -> 1, std -> large, SNR -> small, gamma -> small.
        # As t -> 0, std -> 0, SNR -> large, gamma -> 1.
        # This focuses Physics Loss on early diffusion steps (low noise), where structure matters.
        
        snr_weight = 1.0 / (1.0 + std_t.mean()**2 + 1e-8) # Simple scalar weight per batch or per item?
        # Let's use mean for stability
        
        total_loss = loss_dsm + snr_weight * loss_arb
        
        # 8. Optimization
        total_loss.backward()
        
        # Stability: Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.conditioner.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(self.dit.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Check for NaNs in loss to warn
        if torch.isnan(total_loss):
            print("WARNING: Total Loss is NaN!")
            print(f"DSM: {loss_dsm.item()}, Arb: {loss_arb.item()}")
        
        return {
            'loss_total': total_loss.item(),
            'loss_dsm': loss_dsm.item(),
            'loss_arb': loss_arb.item(),
            'snr_weight': snr_weight.item()
        }
