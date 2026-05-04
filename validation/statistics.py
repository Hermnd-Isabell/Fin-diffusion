
import numpy as np
import torch
import scipy.stats as stats
from scipy.stats import wasserstein_distance

def compute_distribution_metrics(real_data, synth_data):
    """
    Compare distributions of key statistics between Real and Synthetic data.
    Args:
        real_data: (N, 1, 32, 32) numpy array or tensor
        synth_data: (N, 1, 32, 32) numpy array or tensor
    Returns:
        Dictionary of Wasserstein distances and KL divergences.
    """
    if isinstance(real_data, torch.Tensor): real_data = real_data.cpu().numpy()
    if isinstance(synth_data, torch.Tensor): synth_data = synth_data.cpu().numpy()
    
    # Flatten features for distribution comparison
    # 1. ATM Volatility (Assuming center of grid is ATM approx)
    atm_idx = 16
    real_atm = real_data[:, 0, :, atm_idx].flatten()
    synth_atm = synth_data[:, 0, :, atm_idx].flatten()
    
    wd_atm = wasserstein_distance(real_atm, synth_atm)
    
    # 2. Skewness (Slope across strikes)
    # Simple proxy: Vol(K_high) - Vol(K_low)
    real_skew = (real_data[:, 0, :, -1] - real_data[:, 0, :, 0]).flatten()
    synth_skew = (synth_data[:, 0, :, -1] - synth_data[:, 0, :, 0]).flatten()
    
    wd_skew = wasserstein_distance(real_skew, synth_skew)
    
    # 3. Global Distribution
    wd_global = wasserstein_distance(real_data.flatten(), synth_data.flatten())
    
    return {
        "WD_ATM_Vol": wd_atm,
        "WD_Skew": wd_skew,
        "WD_Global": wd_global
    }

def compute_acf(series, lags=10):
    """
    Compute Autocorrelation Function for a time series.
    """
    n = len(series)
    mean = np.mean(series)
    c0 = np.sum((series - mean) ** 2) / n
    def r(h):
        return np.sum((series[:n-h] - mean) * (series[h:] - mean)) / n / c0
    return np.array([r(h) for h in range(lags)])

def compute_time_series_metrics(real_data, synth_data):
    """
    Compare temporal dynamics: ACF and Volatility Clustering.
    """
    if isinstance(real_data, torch.Tensor): real_data = real_data.cpu().numpy()
    if isinstance(synth_data, torch.Tensor): synth_data = synth_data.cpu().numpy()
    
    # Take ATM Vol time series
    real_ts = real_data[:, 0, 16, 16] # Mid-term ATM
    synth_ts = synth_data[:, 0, 16, 16] 
    
    real_acf = compute_acf(real_ts)
    synth_acf = compute_acf(synth_ts)
    
    # Volatility Clustering (ACF of squared returns/diffs)
    real_diff = np.diff(real_ts)
    synth_diff = np.diff(synth_ts)
    
    real_vol_clust = compute_acf(real_diff**2)
    synth_vol_clust = compute_acf(synth_diff**2)
    
    return {
        "ACF_Error": np.mean((real_acf - synth_acf)**2),
        "Vol_Clustering_Error": np.mean((real_vol_clust - synth_vol_clust)**2)
    }

def compute_leverage_effect(iv_series, spot_series):
    """
    Calculate Leverage Effect: Correlation between Spot Returns and IV Changes.
    Should be negative (typically -0.4 to -0.6).
    """
    if isinstance(iv_series, torch.Tensor): iv_series = iv_series.cpu().numpy()
    if isinstance(spot_series, torch.Tensor): spot_series = spot_series.cpu().numpy()
    
    spot_returns = np.diff(np.log(spot_series))
    iv_changes = np.diff(iv_series)
    
    min_len = min(len(spot_returns), len(iv_changes))
    corr = np.corrcoef(spot_returns[:min_len], iv_changes[:min_len])[0, 1]
    
    return corr
