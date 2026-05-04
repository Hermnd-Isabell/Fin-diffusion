
import numpy as np

class GridMapper:
    """
    Transforms fitted SVI parameters/surfaces into a standardized 32x32 time-moneyness grid.
    Handles missing data (None params) by interpolating from valid expiries.
    """
    def __init__(self, m_points=32, t_points=32, m_range=(-0.3, 0.3), t_range=(0.05, 1.0)):
        self.m_grid = np.linspace(m_range[0], m_range[1], m_points)
        self.t_grid = np.linspace(t_range[0], t_range[1], t_points)
        self.m_points = m_points
        self.t_points = t_points

    def interpolate_surface(self, daily_svi_params, current_date):
        """
        daily_svi_params: List of tuples (expiry_date, T_years, params)
                          params can be None if calibration failed.
        Returns: 32x32 Implied Volatility Surface (Sigma).
        """
        # Filter out failed calibrations
        valid_params = [p for p in daily_svi_params if p[2] is not None]
        
        # Sort by tenor T
        sorted_params = sorted(valid_params, key=lambda x: x[1])
        
        # We need to interpolate total variance w(k, t) = sigma^2(k, t) * t
        
        # 1. Compute w(k) for all VALID tenors on the moneyness grid
        available_T = []
        available_w = [] # List of arrays of shape (m_points,)
        
        def svi_func(k, p):
            a, b, rho, m_param, sigma = p
            return a + b * (rho * (k - m_param) + np.sqrt((k - m_param)**2 + sigma**2))
        
        for unused_expiry, T, params in sorted_params:
            if T <= 0: continue
            available_T.append(T)
            # Calculate sigma^2 on the grid
            sigma2_grid = svi_func(self.m_grid, params)
            # Total variance w = sigma^2 * T
            w_grid = sigma2_grid * T
            available_w.append(w_grid)
            
        available_T = np.array(available_T)
        available_w = np.array(available_w) # Shape (N_valid, m_points)
        
        if len(available_T) < 1:
            return np.zeros((self.t_points, self.m_points))
            
        # 2. Interpolate w for target t_grid
        final_surface_sigma = np.zeros((self.t_points, self.m_points))
        
        for i, t_target in enumerate(self.t_grid):
            # If only 1 valid slice, use it flat
            if len(available_T) == 1:
                # Flat extrapolation of variance? 
                # Better: assume constant volatility. w_target = w_ref * (t_target / t_ref)
                w_ref = available_w[0]
                t_ref = available_T[0]
                w_target = w_ref * (t_target / t_ref)
            
            else:
                # Linear interpolation of total variance in time
                if t_target <= available_T[0]:
                    # Short end extrapolation using constant vol from earliest point
                    w_ref = available_w[0]
                    t_ref = available_T[0]
                    w_target = w_ref * (t_target / t_ref)
                elif t_target >= available_T[-1]:
                    # Long end extrapolation using constant vol from latest point
                    w_ref = available_w[-1]
                    t_ref = available_T[-1]
                    w_target = w_ref * (t_target / t_ref)
                else:
                    # Linear Interp between two valid slices
                    idx = np.searchsorted(available_T, t_target)
                    T_prev, T_next = available_T[idx-1], available_T[idx]
                    w_prev, w_next = available_w[idx-1], available_w[idx]
                    
                    ratio = (t_target - T_prev) / (T_next - T_prev)
                    w_target = w_prev + ratio * (w_next - w_prev)
            
            # Convert back to Sigma = sqrt(w / t)
            # Handle numerical zeros/negatives
            sigma_target = np.sqrt(np.maximum(w_target, 0) / t_target)
            final_surface_sigma[i, :] = sigma_target
            
        return final_surface_sigma
