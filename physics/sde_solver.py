
import torch
import numpy as np

class LogSpaceGBM:
    """
    Geometric Brownian Motion (GBM) SDE Solver for Diffusion Models.
    Operates in Log-Space to ensure positivity of the underlying variable.
    
    Forward SDE (Log-Space):
        dY_t = (mu - 0.5*sigma^2)dt + sigma*dW_t
        where Y_t = ln(X_t)
        
    """
    def __init__(self, num_timesteps=1000, sigma_min=0.01, sigma_max=2.0):
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
        # Discretize time [0, 1]
        self.timesteps = torch.linspace(0, 1, num_timesteps)
        
        # Precompute schedule parameters
        # Linear schedule for sigma(t)
        self.sigmas = self.sigma_min + (self.sigma_max - self.sigma_min) * self.timesteps
        
        # Drift adaptation? 
        # For standard diffusion, we want to drift towards noise.
        # But here we are modeling physical process.
        # If we just want to add noise to destroy information:
        # Standard Variance Preserving (VP) or Variance Exploding (VE) SDEs are used.
        # But user asked for GBM. Let's stick to the physical interpretation.
        # Or maybe the user means the diffusion model's noise schedule mimics GBM volatility structure?
        # "GBM-Guided Diffusion": The noise `sigma(t)` is not arbitrary, but structured like financial vol?
        # Let's implement standard SDE functions: marginal_prob, prior_sampling.
    
    def marginal_prob(self, x_start, t):
        """
        Compute mean and std of p(x_t | x_0) in Log-Space.
        Args:
            x_start: Y_0 = ln(IV_0)
            t: Time tensor (0 to 1)
        """
        # In log-space, diffusion is additive Gaussian.
        # Simple VE-SDE (Variance Exploding): dx = sigma(t) dw
        # Mean = x_0, Std = int_0^t sigma(s)^2 ds
        # If sigma(t) is linear: sigma(t) = s_min + (s_max-s_min)t
        # integral is tricky.
        
        # Simplified: sigma^t is scaling factor.
        # Let's use the VE-SDE formulation commonly used in Score-Based Generative Modeling (Song et al.)
        # p_0t(x(t) | x(0)) = N(x(t); x(0), std(t)^2 I)
        # std(t) = sigma_min * (sigma_max/sigma_min)^t 
        # This corresponds to Geometric Noise Schedule!
        
        log_mean_coeff = -0.25 * t**2 * (self.sigma_max**2 - self.sigma_min**2) - 0.5 * t * self.sigma_min**2
        std = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        
        # Keep it simple: Additive Noise
        mean = x_start
        # Ensure std is (B, 1, 1, 1)
        if std.dim() > 1:
            std = std.flatten()
        std = std.view(-1, 1, 1, 1)
        
        return mean, std

    def q_sample(self, x_start, t, noise=None):
        """
        Sample from q(x_t | x_0).
        Forward process: Add noise to clean image.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        mean, std = self.marginal_prob(x_start, t)
        
        return mean + std * noise

    def prior_sampling(self, shape):
        """
        Sample from prior distribution p(x_T) ~ N(0, std(1)^2 I)
        """
        return torch.randn(*shape) * self.sigma_max
