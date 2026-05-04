
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

# ----------------- Black-Scholes Formula -----------------
def bs_price(S, K, T, r, sigma, option_type='C'):
    """
    Calculate Black-Scholes option price.
    """
    # Safe handling for small T
    if T <= 0:
        return max(S - K, 0) if option_type == 'C' else max(K - S, 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'C':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    return price

def implied_volatility(price, S, K, T, r, option_type='C'):
    """
    Calculate Implied Volatility using Brent's method (safer than Newton for bounded).
    """
    target = price
    
    # Check intrinsic value lower bound
    intrinsic = max(S - K * np.exp(-r*T), 0) if option_type == 'C' else max(K * np.exp(-r*T) - S, 0)
    if target <= intrinsic + 1e-6:
        return 1e-6 # Minimum IV
        
    def objective(sigma):
        return bs_price(S, K, T, r, sigma, option_type) - target
    
    try:
        # Search in range [0.1%, 500%]
        return brentq(objective, 1e-4, 5.0, xtol=1e-5)
    except Exception:
        return np.nan

# ----------------- Raw Loader -----------------
class RawLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None

    def load_data(self):
        """
        Load data from Excel.
        """
        print(f"Loading data from {self.file_path}...")
        try:
            self.df = pd.read_excel(self.file_path)
            print(f"Loaded {len(self.df)} rows from {self.file_path}")
            print(f"Columns: {self.df.columns.tolist()}")
            if 'trade_date' in self.df.columns:
                print(f"Sample trade_dates: {self.df['trade_date'].head().tolist()}")
                print(f"trade_date dtype: {self.df['trade_date'].dtype}")
        except FileNotFoundError:
            print("File not found. Creating mock data for testing.")
            self.df = self._create_mock_data()
        
        return self.df

    def _create_mock_data(self):
        """
        Create structured mock data if file is missing.
        """
        # Mocking 50ETF data structure
        dates = pd.date_range('2023-01-01', '2023-01-05')
        data = []
        for d in dates:
            S = 2.5 + np.random.normal(0, 0.05)
            # Create options for this date
            expiries = [d + pd.Timedelta(days=30), d + pd.Timedelta(days=90)]
            for T_date in expiries:
                if T_date <= d: continue
                days = (T_date - d).days
                T = days / 365.0
                # Strikes
                for K in np.linspace(S*0.8, S*1.2, 5):
                    # Call
                    sigma = 0.2 + 0.1 * (K/S - 1)**2 # Smile
                    C = bs_price(S, K, T, 0.03, sigma, 'C')
                    data.append({
                        'trade_date': int(d.strftime('%Y%m%d')),
                        'exercise_price': K,
                        'close': C,
                        'fund_close': S,
                        'last_edate': int(T_date.strftime('%Y%m%d')),
                        'remaining_time': days,
                        'call_put': 'C',
                        'volume': 1000 + int(np.random.normal(0, 100)),
                        'ten_year': 3.0,
                        'implc_volatlty': sigma * 100 # Mock exchange IV
                    })
                    # Put
                    P = bs_price(S, K, T, 0.03, sigma, 'P')
                    data.append({
                        'trade_date': int(d.strftime('%Y%m%d')),
                        'exercise_price': K,
                        'close': P,
                        'fund_close': S,
                        'last_edate': int(T_date.strftime('%Y%m%d')),
                        'remaining_time': days,
                        'call_put': 'P',
                        'volume': 1000,
                        'ten_year': 3.0,
                        'implc_volatlty': sigma * 100
                    })
        return pd.DataFrame(data)

    def clean_data(self, target_date=None):
        """
        Apply cleaning logic defined in requirements.
        target_date: If provided (int YYYYMMDD), clean only this date's data.
        """
        if self.df is None:
            self.load_data()
            
        # Optimization: If target_date is provided, we can filter from the main DF without reloading
        # However, self.df accumulates? No, self.df is the full dataset.
        
        # print("Values before cleaning:", len(self.df))
        df = self.df # Reference
        
        # 0. Fast Filter by Date if requested
        if target_date is not None:
             # Ensure format match
             target_date_int = int(str(target_date).replace('-', ''))
             print(f"Filtering for date: {target_date_int} (Type: {type(target_date_int)})")
             
             # Clean input column just in case?
             # df['trade_date'] = pd.to_numeric(df['trade_date'], errors='coerce')
             
             filtered_df = df[df['trade_date'] == target_date_int]
             print(f"Rows for {target_date_int}: {len(filtered_df)}")
             
             if len(filtered_df) == 0:
                 print(f"DEBUG: Data found for date {target_date_int} is 0. Data trade_date dtype: {df['trade_date'].dtype}")
                 return None # Return None to signal skip
             df = filtered_df

        # 1. Filter Garbage
        mask_garbage = (df['volume'] >= 50) & (df['close'] >= 0.001)
        df = df[mask_garbage]
        # print(f"After garbage filter: {len(df)}")

        # 2. Time Calculation
        df['T'] = df['remaining_time'] / 365.0
        df = df[df['T'] >= 0.02]
        # print(f"After T >= 0.02 filter: {len(df)}")

        # Normalize Rate
        # If 'ten_year' exists, use it (percentage -> decimal). Else 0.03
        if 'ten_year' in df.columns:
            df['r'] = df['ten_year'] / 100.0
        else:
            df['r'] = 0.03

        # 3. Arbitrage Filter
        # Call: C >= S - K*e^{-rT} -> C - (S - K*e^{-rT}) >= -epsilon
        # Put:  P >= K*e^{-rT} - S -> P - (K*e^{-rT} - S) >= -epsilon
        # Using a small epsilon for numerical stability
        epsilon = -1e-4 
        
        K = df['exercise_price']
        S = df['fund_close']
        T = df['T']
        r = df['r']
        
        discount_factor = np.exp(-r * T)
        
        # Calculate Lower Bounds
        call_lower_bound = S - K * discount_factor
        put_lower_bound = K * discount_factor - S
        
        # Filter Logic
        mask_call = (df['call_put'] == 'C') & (df['close'] >= call_lower_bound + epsilon)
        mask_put  = (df['call_put'] == 'P') & (df['close'] >= put_lower_bound + epsilon)
        
        # Combine
        # Keep if (Call AND Valid) OR (Put AND Valid)
        # But wait, original dataframe has both.
        # We want to keep rows where:
        # if C: check call arb
        # if P: check put arb
        
        # Vectorized check
        is_call = df['call_put'] == 'C'
        is_put = df['call_put'] == 'P'
        
        valid_call = is_call & (df['close'] >= call_lower_bound + epsilon)
        valid_put = is_put & (df['close'] >= put_lower_bound + epsilon)
        
        # Also need to handle cases where lower bound is negative (always valid for option price >= 0)
        # But option price >= 0 is implicit in garbage filter (close >= 0.001)
        
        df = df[valid_call | valid_put]
        # print(f"After arbitrage filter: {len(df)}")

        # 4. Target Calculation (IV)
        # This is expensive, so we use apply with the optimized scalar function
        print("Calculating Implied Volatility (this may take a moment)...")
        
        # Define wrapper for apply
        def get_iv(row):
            return implied_volatility(
                row['close'], 
                row['fund_close'], 
                row['exercise_price'], 
                row['T'], 
                row['r'], 
                row['call_put']
            )
        
        df['iv_calculated'] = df.apply(get_iv, axis=1)
        
        # Drop NaNs where IV Calculation failed
        df = df.dropna(subset=['iv_calculated'])
        # print(f"After IV calculation (final count): {len(df)}")
        
        # Rename columns to match calibrator expectations
        df = df.rename(columns={'exercise_price': 'strike', 'iv_calculated': 'iv', 'last_edate': 'expiry', 'fund_close': 'S'})
        
        return df

if __name__ == "__main__":
    loader = RawLoader("50etf_options.xlsx")
    clean_df = loader.clean_data()
    print(clean_df[['trade_date', 'call_put', 'exercise_price', 'close', 'iv_calculated']].head())
