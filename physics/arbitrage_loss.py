
import torch
import torch.nn as nn
from physics.market_ops import BSModule

class ArbitrageLoss(nn.Module):
    """
    Physics-Informed Loss Function enforcing No-Arbitrage constraints.
    Calculates penalties for Butterfly, Calendar, and Vertical spread violations.
    """
    def __init__(self, weights=None):
        super().__init__()
        # Default weights: [Butterfly, Calendar, Vertical]
        if weights is None:
            self.weights = {'fly': 1.0, 'cal': 1.0, 'vert': 1.0}
        else:
            self.weights = weights
        
        # Convolutions for efficient local difference checks
        # Butterfly: C(K-1) - 2C(K) + C(K+1) >= 0 (Convexity)
        # Kernel: [1, -2, 1] applied along Strike dimension
        self.fly_kernel = torch.tensor([[[[1.0, -2.0, 1.0]]]]) # (Out, In, H, W) -> (1, 1, 1, 3) 
        
    def forward(self, iv_surface, S, K_grid, T_grid, r):
        """
        Args:
            iv_surface: (B, 1, H, W)
            S, K_grid, T_grid, r: Inputs for BS pricing
        """
        # 1. Differentiable Pricing
        # Price Surface C(K, T)
        # If input is (B, C, H, W), we expect C=1 (IV surface)
        prices = BSModule.bs_price(iv_surface, S, K_grid, T_grid, r)
        
        # 2. Butterfly Arbitrage (Convexity w.r.t Strike)
        # C_xx >= 0
        # Discrete: C(K_i-1) - 2C(K_i) + C(K_i+1) >= 0
        # Since K is not uniform, we should properly normalize by dK^2? 
        # For regularization, just penalizing the violation is often enough.
        # Ideally: (C3 - C2)/(K3 - K2) - (C2 - C1)/(K2 - K1) >= 0
        # Simplified (assuming uniform grid for efficiency): standard conv
        
        # Applying convolution along last dimension (Strikes)
        # prices shape: (B, 1, T_dim, K_dim)
        # Padding to keep dimensions same, but we only care about valid interior points
        # fly_val = conv2d(prices)
        
        # Using functional conv2d
        # Reshape kernel relative to device
        kernel = self.fly_kernel.to(prices.device)
        
        # Stride=1, Valid padding to avoid boundary artifacts
        fly_val = torch.nn.functional.conv2d(prices, kernel, stride=1, padding=0)
        
        # Constraint: fly_val >= 0
        # Violation: fly_val < 0
        # Loss: ReLU(-fly_val)
        loss_fly = torch.mean(torch.relu(-fly_val))
        
        # 3. Calendar Arbitrage (Monotonicity w.r.t Time)
        # C(T2) >= C(T1) for T2 > T1  (Assuming r >= q, effectively)
        # Simple diff along Time dimension (dim=-2)
        # C_{t+1} - C_t >= 0
        diff_time = prices[:, :, 1:, :] - prices[:, :, :-1, :]
        
        # Constraint: diff_time >= 0
        # Loss: ReLU(-diff_time)
        loss_cal = torch.mean(torch.relu(-diff_time))
        
        # 4. Vertical Arbitrage (Monotonicity w.r.t Strike)
        # Call Prices must decrease as Strike increases
        # C(K2) <= C(K1) for K2 > K1
        # C_{k+1} - C_k <= 0  =>  C_k - C_{k+1} >= 0
        diff_strike = prices[:, :, :, :-1] - prices[:, :, :, 1:]
        
        # Constraint: diff_strike >= 0
        # Loss: ReLU(-diff_strike)
        loss_vert = torch.mean(torch.relu(-diff_strike))
        
        total_loss = (self.weights['fly'] * loss_fly + 
                      self.weights['cal'] * loss_cal + 
                      self.weights['vert'] * loss_vert)
                      
        return total_loss, {'fly': loss_fly.item(), 'cal': loss_cal.item(), 'vert': loss_vert.item()}
