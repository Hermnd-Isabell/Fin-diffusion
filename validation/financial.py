
import torch
import numpy as np
from physics.market_ops import BSModule
from physics.arbitrage_loss import ArbitrageLoss

def check_static_arbitrage(iv_surface, S, K_grid, T_grid, r, device='cpu'):
    """
    Check for static arbitrage violations in a batch of surfaces.
    Returns percentage of violations.
    """
    # Use ArbitrageLoss module functionality but return boolean mask instead of loss
    # Or manually re-implement basic checks for clarity
    
    bs_module = BSModule()
    prices = bs_module.bs_price(iv_surface, S, K_grid, T_grid, r)
    
    # 1. Butterfly Arbitrage (Call Convexity w.r.t Strike)
    # C(K-dK) - 2C(K) + C(K+dK) >= 0
    # Convolution kernel [1, -2, 1]
    kernel_fly = torch.tensor([[[[1.0, -2.0, 1.0]]]], device=device)
    butterfly = torch.nn.functional.conv2d(prices, kernel_fly)
    
    # Count violations (where butterfly < -epsilon)
    # Epsilon for float precision
    fly_violations = (butterfly < -1e-4).float().mean().item()
    
    # 2. Calendar Arbitrage (Call Monotonicity w.r.t Time)
    # C(T2) >= C(T1) for same K
    # diff_time = C(T+dT) - C(T)
    # kernel_cal = [[[-1], [1]]] along time dimension (dim 2)
    kernel_cal = torch.tensor([[[[-1.0], [1.0]]]], device=device)
    # Need to reshape kernel for conv2d or just ensure we diff correct axis
    # Dimensions: (B, C, T, M)
    # Diff along T (dim 2)
    calendar_diff = prices[:, :, 1:, :] - prices[:, :, :-1, :]
    
    # Count violations (where C_long < C_short)
    cal_violations = (calendar_diff < -1e-4).float().mean().item()
    
    # 3. Vertical Spread (Call Monotonicity w.r.t Strike)
    # C(K1) >= C(K2) for K1 < K2
    # diff_strike = C(K+dK) - C(K) should be <= 0
    vertical_diff = prices[:, :, :, 1:] - prices[:, :, :, :-1]
    
    # Violations where C(Higher Strike) > C(Lower Strike)
    vert_violations = (vertical_diff > 1e-4).float().mean().item()
    
    return {
        "Butterfly_Violation_Rate": fly_violations,
        "Calendar_Violation_Rate": cal_violations,
        "Vertical_Violation_Rate": vert_violations,
        "Total_Passthrough_Rate": 1.0 - (fly_violations + cal_violations + vert_violations)
    }

def check_greeks_smoothness(iv_surface, S, K_grid, T_grid, r):
    """
    Calculate Greeks and check for smoothness (Gamma > 0 and no spikes).
    """
    # Enable grad for Greeks
    iv_surface.requires_grad_(True)
    if not S.requires_grad: S = S.detach().clone().requires_grad_(True)
    
    prices = BSModule.bs_price(iv_surface, S, K_grid, T_grid, r)
    
    # Delta = dC/dS
    delta = torch.autograd.grad(prices.sum(), S, create_graph=True)[0]
    
    # Gamma = dDelta/dS
    gamma = torch.autograd.grad(delta.sum(), S, create_graph=True)[0]
    
    # Gamma should be positive (for long positions)
    # Note: S is usually a scalar broadcasted, so autograd might sum gradients.
    # For accurate element-wise Greeks, better to use FD or complex step if not training.
    # But here we want to check the *Model's* implied Greeks.
    
    # Alternative: Finite Difference on Price Surface
    dK = K_grid[:, :, :, 1:] - K_grid[:, :, :, :-1]
    dCdK = (prices[:, :, :, 1:] - prices[:, :, :, :-1]) / (dK + 1e-8) # Approx -Delta 
    
    d2CdK2 = (dCdK[:, :, :, 1:] - dCdK[:, :, :, :-1]) / (dK[:, :, :, 1:] + 1e-8) # Approx Gamma * S^2 density
    
    # Smoothness: Second derivative should not change sign rapidly or explode
    # Check proportion of negative Gamma (Butterflies check this too)
    neg_gamma_rate = (d2CdK2 < -1e-4).float().mean().item()
    
    # Check max spike
    max_gamma = d2CdK2.max().item()
    
    return {
        "Negative_Gamma_Rate": neg_gamma_rate,
        "Max_Gamma_Spike": max_gamma
    }
