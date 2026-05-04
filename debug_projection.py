"""
Quick test: apply the price-space PAVA projection + Newton-Raphson
inversion to the cached synthetic data and check violations.
"""
import torch
from physics.market_ops import BSModule
from engine.sampler import _project_prices_to_convex, _prices_to_iv_newton

synth_data = torch.load("results/validation_generation/synthetic_1000.pt")
B = synth_data.shape[0]

k_vals = torch.linspace(-0.3, 0.3, 32)
t_vals = torch.linspace(0.05, 1.0, 32)
T_g, k_g = torch.meshgrid(t_vals, k_vals, indexing='ij')
S = torch.tensor(1.0)
K_grid = (S * torch.exp(k_g)).unsqueeze(0).expand(B, 1, 32, 32)
T_grid = T_g.unsqueeze(0).expand(B, 1, 32, 32)
r = torch.tensor(0.0)

print("=== Before projection ===")
prices_raw = BSModule.bs_price(synth_data, S, K_grid, T_grid, r)
kernel_fly = torch.tensor([[[[1.0, -2.0, 1.0]]]])
butterfly_before = torch.nn.functional.conv2d(prices_raw, kernel_fly)
fly_viol_before = (butterfly_before < -1e-4).float().mean().item()
cal_diff_before = prices_raw[:, :, 1:, :] - prices_raw[:, :, :-1, :]
cal_viol_before = (cal_diff_before < -1e-4).float().mean().item()
vert_diff_before = prices_raw[:, :, :, 1:] - prices_raw[:, :, :, :-1]
vert_viol_before = (vert_diff_before > 1e-4).float().mean().item()
arb_free_before = 1.0 - (fly_viol_before + cal_viol_before + vert_viol_before)
print(f"  Butterfly: {fly_viol_before*100:.2f}%  Calendar: {cal_viol_before*100:.2f}%  Vertical: {vert_viol_before*100:.2f}%")
print(f"  Arb-Free Rate: {arb_free_before*100:.2f}%")

print("\n=== After price-space projection ===")
with torch.no_grad():
    prices_proj = _project_prices_to_convex(prices_raw)

butterfly_after = torch.nn.functional.conv2d(prices_proj, kernel_fly)
fly_viol_after = (butterfly_after < -1e-4).float().mean().item()
cal_diff_after = prices_proj[:, :, 1:, :] - prices_proj[:, :, :-1, :]
cal_viol_after = (cal_diff_after < -1e-4).float().mean().item()
vert_diff_after = prices_proj[:, :, :, 1:] - prices_proj[:, :, :, :-1]
vert_viol_after = (vert_diff_after > 1e-4).float().mean().item()
arb_free_after = 1.0 - (fly_viol_after + cal_viol_after + vert_viol_after)
print(f"  Butterfly: {fly_viol_after*100:.2f}%  Calendar: {cal_viol_after*100:.2f}%  Vertical: {vert_viol_after*100:.2f}%")
print(f"  Arb-Free Rate: {arb_free_after*100:.2f}%")

print("\n=== After Newton-Raphson IV inversion ===")
with torch.no_grad():
    iv_corrected = _prices_to_iv_newton(prices_proj, S, K_grid, T_grid, r, n_iter=10)

print(f"  IV range: [{iv_corrected.min().item():.4f}, {iv_corrected.max().item():.4f}]")
print(f"  IV mean: {iv_corrected.mean().item():.4f}  std: {iv_corrected.std().item():.4f}")
print(f"  NaN count: {iv_corrected.isnan().sum().item()}")

# Re-check violations on the inverted IV
prices_recheck = BSModule.bs_price(iv_corrected, S, K_grid, T_grid, r)
butterfly_recheck = torch.nn.functional.conv2d(prices_recheck, kernel_fly)
fly_recheck = (butterfly_recheck < -1e-4).float().mean().item()
cal_recheck = ((prices_recheck[:, :, 1:, :] - prices_recheck[:, :, :-1, :]) < -1e-4).float().mean().item()
vert_recheck = ((prices_recheck[:, :, :, 1:] - prices_recheck[:, :, :, :-1]) > 1e-4).float().mean().item()
arb_free_recheck = 1.0 - (fly_recheck + cal_recheck + vert_recheck)
print(f"\n  Re-priced violations:")
print(f"  Butterfly: {fly_recheck*100:.2f}%  Calendar: {cal_recheck*100:.2f}%  Vertical: {vert_recheck*100:.2f}%")
print(f"  Arb-Free Rate: {arb_free_recheck*100:.2f}%")

# Gamma spike check (FD)
dK = K_grid[:, :, :, 1:] - K_grid[:, :, :, :-1]
dCdK = (prices_recheck[:, :, :, 1:] - prices_recheck[:, :, :, :-1]) / (dK + 1e-8)
d2CdK2 = (dCdK[:, :, :, 1:] - dCdK[:, :, :, :-1]) / (dK[:, :, :, 1:] + 1e-8)
print(f"  Max Gamma Spike: {d2CdK2.max().item():.4f}")
print(f"  Neg Gamma Rate: {(d2CdK2 < -1e-4).float().mean().item()*100:.2f}%")
