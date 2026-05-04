import torch
import torch.nn.functional as F
from physics.arbitrage_loss import ArbitrageLoss

device = 'cpu'
arb_loss_fn = ArbitrageLoss(weights={'fly': 5.0, 'cal': 1.0, 'vert': 1.0}).to(device)

B = 4
torch.manual_seed(42)
ivs = torch.rand(B, 1, 32, 32) * 0.3 + 0.1
k_vals = torch.linspace(-0.3, 0.3, 32)
t_vals = torch.linspace(0.05, 1.0, 32)
T_g, k_g = torch.meshgrid(t_vals, k_vals, indexing='ij')
S = torch.tensor(1.0)
K_grid = (S * torch.exp(k_g)).unsqueeze(0).expand(B, 1, 32, 32)
T_grid = T_g.unsqueeze(0).expand(B, 1, 32, 32)
r = torch.tensor(0.0)

loss_before, d = arb_loss_fn(ivs, S, K_grid, T_grid, r)
print(f"Loss BEFORE: {loss_before.item():.6f}  fly:{d['fly']:.6f} cal:{d['cal']:.6f} vert:{d['vert']:.6f}")

# --- Clean simple gradient descent with very small fixed LR ---
ivs_refined = ivs.detach().clone()
n_steps = 300
lr = 0.0005   # tiny, stable LR

for step_idx in range(n_steps):
    with torch.enable_grad():
        ivs_in = ivs_refined.detach().requires_grad_(True)
        loss_arb, loss_dict = arb_loss_fn(ivs_in, S, K_grid, T_grid, r)
        if loss_arb.item() < 1e-7:
            print(f"Converged at step {step_idx}")
            break
        grad = torch.autograd.grad(loss_arb, ivs_in)[0]

    ivs_refined = (ivs_refined.detach() - lr * grad.detach()).clamp(min=1e-4)

    if (step_idx + 1) % 60 == 0:
        print(f"  Step {step_idx+1:3d}: loss={loss_arb.item():.6f} fly:{loss_dict['fly']:.5f}")

loss_after, d = arb_loss_fn(ivs_refined, S, K_grid, T_grid, r)
print(f"Loss AFTER:  {loss_after.item():.6f}  fly:{d['fly']:.6f} cal:{d['cal']:.6f} vert:{d['vert']:.6f}")
print(f"Reduction: {(loss_before.item() - loss_after.item()) / loss_before.item() * 100:.1f}%")
print(f"IVS mean before: {ivs.mean().item():.4f}  after: {ivs_refined.mean().item():.4f}")
print(f"IVS std  before: {ivs.std().item():.4f}  after: {ivs_refined.std().item():.4f}")
