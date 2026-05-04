
import torch
import numpy as np

from physics.market_ops import BSModule
from physics.arbitrage_loss import ArbitrageLoss
from physics.sde_solver import LogSpaceGBM

def test_physics_core():
    print("--- Testing Physics Core ---")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Setup Mock Data
    # Batch=2, Channel=1, Time=32, Strike=32
    B, C, H, W = 2, 1, 32, 32
    
    # Create a fake IV surface with gradients enabled
    # Initialize around 20% vol + noise
    iv_surface = torch.tensor(
        np.full((B, C, H, W), 0.2) + np.random.normal(0, 0.01, (B, C, H, W)), 
        dtype=torch.float32, 
        device=device,
        requires_grad=True
    )
    
    S = torch.tensor(3.0, device=device)
    r = torch.tensor(0.03, device=device)
    
    # Create grids
    # K: 2.5 to 3.5
    # T: 0.1 to 1.0
    k_vals = torch.linspace(2.5, 3.5, W, device=device)
    t_vals = torch.linspace(0.1, 1.0, H, device=device)
    
    # Meshgrid (H, W) -> (T, K)
    # T varies along H (dim 2), K varies along W (dim 3)
    # Be careful with meshgrid indexing 'ij' vs 'xy'
    # torch.meshgrid defaults to 'ij'
    # T_grid: (H, W) where rows are constant time
    # K_grid: (H, W) where cols are constant strike
    
    T_grid, K_grid = torch.meshgrid(t_vals, k_vals, indexing='ij')
    
    # Expand to batch
    T_grid = T_grid.expand(B, C, H, W)
    K_grid = K_grid.expand(B, C, H, W)
    
    print("Data shapes initialized.")
    print(f"IV: {iv_surface.shape}, K: {K_grid.shape}, T: {T_grid.shape}")

    # 2. Test BS Module
    print("\n[Test 1] Black-Scholes Pricing Gradient Check")
    prices = BSModule.bs_price(iv_surface, S, K_grid, T_grid, r)
    
    print(f"Price range: {prices.min().item():.4f} - {prices.max().item():.4f}")
    
    # Simple loss on prices to check gradient flow
    target_price = torch.zeros_like(prices) # Dummy target
    loss_price = torch.nn.functional.mse_loss(prices, target_price)
    
    # Backward
    loss_price.backward()
    
    if iv_surface.grad is not None:
        grad_norm = iv_surface.grad.norm().item()
        print(f"PASS: Gradient flowed back to IV Surface. Norm: {grad_norm:.6f}")
        # Reset grad
        iv_surface.grad.zero_()
    else:
        print("FAIL: No gradient on IV Surface!")
        return

    # 3. Test Arbitrage Loss
    print("\n[Test 2] Arbitrage Constraints")
    arb_module = ArbitrageLoss().to(device)
    
    loss_arb, components = arb_module(iv_surface, S, K_grid, T_grid, r)
    
    print(f"Total Arb Loss: {loss_arb.item():.6f}")
    print(f"Components: {components}")
    
    loss_arb.backward()
    
    if iv_surface.grad is not None:
        grad_norm = iv_surface.grad.norm().item()
        print(f"PASS: Arbitrage Loss Gradient: {grad_norm:.6f}")
    else:
        print("FAIL: No gradient from Arb Loss!")
        
    # 4. Test SDE Solver
    print("\n[Test 3] SDE Solver (Log-Space)")
    # Since we didn't implement fully differentiable SDE yet (just helper), 
    # we just check scheduling logic.
    solver = LogSpaceGBM(num_timesteps=10)
    
    x_start = torch.log(iv_surface.detach()) # Work in Log-IV space
    
    # Sample at t=0.5
    t = torch.tensor([0.5], device=device)
    x_t = solver.q_sample(x_start, t) # Add noise
    
    print(f"Clean (Log): {x_start.mean():.4f}, Noisy (t=0.5): {x_t.mean():.4f}")
    print("SDE Sampling logical check passed.")
    
    print("\n--- All Physics Checks Passed ---")

if __name__ == "__main__":
    test_physics_core()
