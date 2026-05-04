
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from raw_loader import RawLoader
from svi_calibrator import SVICalibrator
from grid_mapper import GridMapper

def run_pipeline(excel_path="50etf_options.xlsx", sample_date=None, plot=True):
    print("--- 1. Raw Loading ---")
    loader = RawLoader(excel_path)
    loader.load_data()
    
    # Pick a sample date from raw data
    if loader.df is not None:
        unique_dates = sorted(loader.df['trade_date'].unique())
        if sample_date is None:
            # Pick a date around 2021 (mid-sample)
            mid_idx = len(unique_dates)//2
            sample_date = unique_dates[mid_idx]
    else:
        print("Failed to load data")
        return

    print(f"\n--- Processing Date: {sample_date} (Cleaning only this date) ---")
    # Clean only this date
    df = loader.clean_data(target_date=sample_date)
    
    # day_df is just df now since we filtered inside clean_data
    day_df = df
    
    if len(day_df) == 0:
        print(f"No data for date {sample_date}")
        return

    S = day_df['fund_close'].iloc[0]
    r = day_df['r'].iloc[0]
    print(f"Spot: {S:.4f}, Rate: {r:.4f}")
    
    # Group by expiry
    expiries = sorted(day_df['last_edate'].unique())
    daily_svi_params = []
    
    print(f"\n--- 2. SVI Calibration ({len(expiries)} expiries) ---")
    
    if plot:
        plt.figure(figsize=(15, 5))
    
    for i, expiry in enumerate(expiries):
        sub_df = day_df[day_df['last_edate'] == expiry]
        T = sub_df['T'].iloc[0]
        
        # We need unique strikes and their IVs.
        # Average IV if multiple (Call/Put should have same IV theoretically, 
        # but in practice diff. We take average or just Call/Put based on liquidity).
        # Simple approach: Group by strike, take mean IV.
        
        grp = sub_df.groupby('exercise_price')['iv_calculated'].mean().reset_index()
        strikes = grp['exercise_price'].values
        ivs = grp['iv_calculated'].values
        
        if len(strikes) < 5:
            print(f"Expiry {expiry}: Not enough points ({len(strikes)}) - Skipping")
            continue
            
        calibrator = SVICalibrator(strikes, ivs, T, S, r, volumes=day_df[day_df['last_edate'] == expiry].groupby('exercise_price')['volume'].mean().values)
        params, mse = calibrator.calibrate()
        
        if params is None:
             print(f"Expiry {expiry}: Calibration Failed (likely due to constraints/sparsity)")
             continue
        
        print(f"Expiry: {expiry} (T={T:.3f}) | MSE: {mse:.6f} | Params: {np.round(params, 4)}")
        daily_svi_params.append((expiry, T, params))
        
        # Plotting
        if plot:
            plt.subplot(1, len(expiries), i+1)
            # Use calibrator.k and calibrator.target_sigma2 because they might have been filtered
            plt.scatter(calibrator.k, calibrator.target_sigma2, label='Market Var', color='blue', s=10)
            
            k_plot = np.linspace(min(calibrator.k)-0.1, max(calibrator.k)+0.1, 100)
            model_var = calibrator.svi_model(k_plot, params)
            plt.plot(k_plot, model_var, label='SVI Fit', color='red')
            plt.title(f"Exp: {expiry}\nT={T:.2f}, MSE={mse:.5f}")
            plt.xlabel("Log-Moneyness k")
            plt.ylabel("Total Variance w")
            plt.legend()
            
    if plot:
        plt.tight_layout()
        plt.savefig(f"svi_fit_{sample_date}.png")
        print(f"Plot saved to svi_fit_{sample_date}.png")
    
    print("\n--- 3. Grid Transformation ---")
    mapper = GridMapper()
    surface = mapper.interpolate_surface(daily_svi_params, sample_date)
    
    print(f"Generated Surface Shape: {surface.shape}")
    print(f"Surface Stats: Min={surface.min():.4f}, Max={surface.max():.4f}, Mean={surface.mean():.4f}")
    
    if plot:
        plt.figure(figsize=(8, 6))
        plt.imshow(surface, aspect='auto', origin='lower', extent=[-0.3, 0.3, 0.05, 1.0])
        plt.colorbar(label='Implied Volatility')
        plt.xlabel("Log-Moneyness")
        plt.ylabel("Tenor (Years)")
        plt.title(f"Interpolated IV Surface - {sample_date}")
        plt.savefig(f"ivs_surface_{sample_date}.png")
        print(f"Surface plot saved to ivs_surface_{sample_date}.png")

if __name__ == "__main__":
    import sys
    import argparse
    
    sys.path.append('.')
    
    parser = argparse.ArgumentParser(description='Run Data Factory Pipeline')
    parser.add_argument('--date', type=int, help='Date in YYYYMMDD format', default=None)
    parser.add_argument('--file', type=str, help='Path to Excel file', default="50etf_options.xlsx")
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    run_pipeline(excel_path=args.file, sample_date=args.date, plot=not args.no_plot)
