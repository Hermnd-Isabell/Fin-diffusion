"""
Diagnose: are the butterfly violations from the generated surfaces
structural (large violations) or numerical (tiny epsilon violations)?
"""
import torch
import os
from physics.market_ops import BSModule

# Load the cached synthetic data from the last run
synth_path = "results/validation_generation/synthetic_1000.pt"
if not os.path.exists(synth_path):
    print("No synthetic data found. Run validation first.")
    exit()

synth_data = torch.load(synth_path)
B = synth_data.shape[0]
device = 'cpu'

k_vals = torch.linspace(-0.3, 0.3, 32)
t_vals = torch.linspace(0.05, 1.0, 32)
T_g, k_g = torch.meshgrid(t_vals, k_vals, indexing='ij')
S = torch.tensor(1.0)
K_grid = (S * torch.exp(k_g)).unsqueeze(0).expand(B, 1, 32, 32)
T_grid = T_g.unsqueeze(0).expand(B, 1, 32, 32)
r = torch.tensor(0.0)

# Compute prices
prices = BSModule.bs_price(synth_data, S, K_grid, T_grid, r)

# Butterfly violations
kernel_fly = torch.tensor([[[[1.0, -2.0, 1.0]]]])
butterfly = torch.nn.functional.conv2d(prices, kernel_fly)

fly_violations = butterfly[butterfly < -1e-4]
print(f"Butterfly violations (epsilon=1e-4):")
print(f"  Count: {len(fly_violations)} out of {butterfly.numel()} total cells")
print(f"  Fraction: {len(fly_violations)/butterfly.numel()*100:.2f}%")
print(f"  Min (largest violation): {fly_violations.min().item():.6e}")
print(f"  Mean violation: {fly_violations.mean().item():.6e}")
print(f"  Std violation: {fly_violations.std().item():.6e}")

# Distribution of violation magnitudes
import numpy as np
viol_np = fly_violations.numpy()
thresholds = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
print(f"\n  Violation size distribution:")
for i in range(len(thresholds)-1):
    mask = (np.abs(viol_np) >= thresholds[i]) & (np.abs(viol_np) < thresholds[i+1])
    print(f"    {thresholds[i]:.0e} to {thresholds[i+1]:.0e}: {mask.sum()} ({mask.mean()*100:.1f}%)")
mask = np.abs(viol_np) >= thresholds[-1]
print(f"    >= {thresholds[-1]:.0e}: {mask.sum()} ({mask.mean()*100:.1f}%)")

print(f"\nButterfly (all values):")
print(f"  Min: {butterfly.min().item():.6e}")
print(f"  Max: {butterfly.max().item():.6e}")
print(f"  Mean: {butterfly.mean().item():.6e}")
