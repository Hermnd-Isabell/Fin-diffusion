
import numpy as np
from scipy.optimize import minimize

class SVICalibrator:
    """
    Calibrates SVI parameters for a single slice of option data (same expiry).
    Model:
        sigma_SVI^2(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where k = log(K/F)
    """
    def __init__(self, strikes, ivs, T, S, r, volumes=None):
        """
        strikes: Array of strike prices
        ivs: Array of implied volatilities
        T: Time to maturity
        S: Spot price
        r: Risk-free rate
        volumes: Array of trading volumes (optional, for weighting)
        """
        self.strikes = np.array(strikes)
        self.ivs = np.array(ivs)
        self.T = T
        self.S = S
        self.r = r
        if volumes is not None:
             self.volumes = np.array(volumes)
        else:
             self.volumes = None
        
        # Calculate Forward Price F = S * exp(r*T) (Assuming q=0)
        self.F = S * np.exp(r * T)
        
        # Calculate Log-Moneyness k = log(K/F)
        self.k = np.log(self.strikes / self.F)
        
        # Target Variance w = sigma^2 * T
        self.target_sigma2 = self.ivs ** 2
        
        # Pre-filtering: Drop NaN/Inf
        mask = np.isfinite(self.target_sigma2)
        self.strikes = self.strikes[mask]
        self.ivs = self.ivs[mask]
        self.volumes = self.volumes[mask] if self.volumes is not None else None
        self.k = self.k[mask]
        self.target_sigma2 = self.target_sigma2[mask]

    def svi_model(self, k, params):
        a, b, rho, m, sigma = params
        val = a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))
        return val

    def huber_loss(self, residuals, delta=0.05):
        """
        Huber Loss: Robust to outliers.
        L = 0.5 * r^2                  if |r| <= delta
        L = delta * (|r| - 0.5*delta)  otherwise
        """
        abs_r = np.abs(residuals)
        quadratic = np.minimum(abs_r, delta)
        linear = abs_r - quadratic
        return 0.5 * quadratic**2 + delta * linear

    def objective(self, params):
        pred_sigma2 = self.svi_model(self.k, params)
        
        # Calculate Weights
        weights = np.ones_like(self.k)
        atm_mask = np.abs(self.k) < 0.1
        weights[atm_mask] = 5.0 # Boost ATM importance
        
        if self.volumes is not None:
             vol_weights = np.log1p(self.volumes)
             vol_weights = vol_weights / (np.mean(vol_weights) + 1e-8)
             weights = weights * vol_weights

        residuals = pred_sigma2 - self.target_sigma2
        
        # Use Huber Loss instead of MSE
        loss = np.mean(weights * self.huber_loss(residuals, delta=0.05)) # delta=0.05 roughly 5% vol error squared
        
        # Soft Constraints
        a, b, rho, m, sigma = params
        penalty = 0.0
        lambda_p = 1000.0
        
        if b < 0: penalty += lambda_p * abs(b)**2 # b >= 0
        if abs(rho) >= 1: penalty += lambda_p * (abs(rho) - 0.999)**2 # |rho| < 1
        if sigma <= 0: penalty += lambda_p * (abs(sigma) + 0.001)**2 # sigma > 0
        if a + b * sigma * np.sqrt(1 - rho**2 + 1e-8) < 0: # a + ... >= 0
             penalty += lambda_p * abs(a + b * sigma * np.sqrt(1 - rho**2))**2
        
        # Check predicted variance non-negative
        min_var = np.min(pred_sigma2)
        if min_var < 0: penalty += lambda_p * abs(min_var)**2
             
        return loss + penalty

    def filter_outliers_ransac(self):
        """
        Simple outlier filtering using quadratic fit residuals.
        """
        if len(self.k) < 5: return
        
        # Simple quadratic fit: y = c2*x^2 + c1*x + c0
        try:
            coeffs = np.polyfit(self.k, self.target_sigma2, 2)
            fitted = np.polyval(coeffs, self.k)
            res = self.target_sigma2 - fitted
            std = np.std(res)
            
            # Keep points within 2 std dev
            mask = np.abs(res) <= 2 * std
            
            # Update data
            self.k = self.k[mask]
            self.target_sigma2 = self.target_sigma2[mask]
            self.strikes = self.strikes[mask]
            self.ivs = self.ivs[mask]
            if self.volumes is not None:
                self.volumes = self.volumes[mask]
        except Exception:
            pass # Fallback if polyfit fails

    def calibrate(self):
        # 1. RANSAC-style Pre-filtering
        self.filter_outliers_ransac()

        # Check sufficiency after filtering
        if len(self.k) < 5:
            return None, float('inf')

        # Smart Initialization
        min_var_idx = np.argmin(self.target_sigma2)
        a_init = self.target_sigma2[min_var_idx]
        m_init = self.k[min_var_idx]
        
        initial_guess = [a_init, 0.1, -0.5, m_init, 0.1]
        
        bounds = [
            (-0.5, 2.0),  # a
            (0.0, 5.0),   # b
            (-0.999, 0.999), # rho
            (-2.0, 2.0),  # m
            (0.001, 2.0)  # sigma
        ]
        
        try:
            result = minimize(
                self.objective, 
                initial_guess, 
                method='L-BFGS-B', 
                bounds=bounds, 
                tol=1e-6
            )
            
            params = result.x
            a, b, rho, m, sigma = params
            
            # Post-Fit Quality Check
            if b < 1e-4: return None, float('inf') # Flat line
            if sigma > 2.0: return None, float('inf') # Unrealistic
            
            return params, result.fun
            
        except Exception:
             return None, float('inf')
