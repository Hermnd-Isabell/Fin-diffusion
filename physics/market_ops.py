
import torch
import numpy as np

class BSModule:
    """
    Differentiable Black-Scholes Module.
    Supports pricing and Greeks calculation using PyTorch autograd.
    """
    @staticmethod
    def bs_price(iv_surface, S, K_grid, T_grid, r, q=0.0):
        """
        Calculate Call Option Price Surface.
        
        Args:
            iv_surface: (B, C, H, W) Tensor or (B, H, W). Implied Volatility (sigma).
            S: (B, 1, 1, 1) or Scalar. Spot Price.
            K_grid: (B, C, H, W) or (H, W). Strike Prices.
            T_grid: (B, C, H, W) or (H, W). Time to Maturity.
            r: Scalar or Tensor. Risk-free rate.
            q: Scalar or Tensor. Dividend yield.
            
        Returns:
            price_surface: (B, C, H, W) Call Prices.
        """
        # Ensure inputs are tensors
        if not isinstance(iv_surface, torch.Tensor):
            iv_surface = torch.tensor(iv_surface)
            
        # Avoid division by zero
        T_safe = torch.clamp(T_grid, min=1e-5)
        sigma_safe = torch.clamp(iv_surface, min=1e-5)
        
        # d1, d2
        # d1 = (ln(S/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
        d1 = (torch.log(S / K_grid) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * torch.sqrt(T_safe))
        d2 = d1 - sigma_safe * torch.sqrt(T_safe)
        
        # Standard Normal CDF
        normal = torch.distributions.Normal(0, 1)
        
        # Call Price: C = S*e^{-qT}*N(d1) - K*e^{-rT}*N(d2)
        price = S * torch.exp(-q * T_safe) * normal.cdf(d1) - K_grid * torch.exp(-r * T_safe) * normal.cdf(d2)
        
        return price

    @staticmethod
    def delta(iv_surface, S, K_grid, T_grid, r, q=0.0):
        """
        Analytical Delta.
        """
        # Re-implement d1 calculation or use autodiff on S? 
        # Analytical is faster normally, but we can trust autodiff too.
        # Let's use analytical for customary Greeks.
        T_safe = torch.clamp(T_grid, min=1e-5)
        sigma_safe = torch.clamp(iv_surface, min=1e-5)
        d1 = (torch.log(S / K_grid) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * torch.sqrt(T_safe))
        normal = torch.distributions.Normal(0, 1)
        
        return torch.exp(-q * T_safe) * normal.cdf(d1)

    @staticmethod
    def vega(iv_surface, S, K_grid, T_grid, r, q=0.0):
        """
        Analytical Vega.
        """
        T_safe = torch.clamp(T_grid, min=1e-5)
        sigma_safe = torch.clamp(iv_surface, min=1e-5)
        d1 = (torch.log(S / K_grid) + (r - q + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * torch.sqrt(T_safe))
        
        # PDF: 1/sqrt(2pi) * exp(-x^2/2)
        # Using torch.exp(normal.log_prob(d1))
        normal = torch.distributions.Normal(0, 1)
        pdf_d1 = torch.exp(normal.log_prob(d1))
        
        return S * torch.exp(-q * T_safe) * torch.sqrt(T_safe) * pdf_d1
