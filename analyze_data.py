
import pandas as pd
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

file_path = "50etf_options.xlsx"

try:
    print("Loading data...")
    df = pd.read_excel(file_path)
    print("Data loaded successfully.")
    
    print("\n--- Basic Information ---")
    print(df.info())
    
    print("\n--- First 5 Rows ---")
    print(df.head())
    
    print("\n--- Summary Statistics ---")
    print(df.describe())
    
    print("\n--- Unique Values in Key Columns ---")
    # Identify potential key columns based on common names if present, 
    # but since I don't know exact names, I'll print unique counts for object/categorical columns
    for col in df.select_dtypes(include=['object', 'category']).columns:
        unique_vals = df[col].unique()
        print(f"{col}: {len(unique_vals)} unique values")
        if len(unique_vals) < 20:
             print(f"   Values: {unique_vals}")

    # Check for date columns if any
    date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols:
        print(f"\nPotential Date Columns: {date_cols}")
        for col in date_cols:
            try:
                print(f"Range for {col}: {df[col].min()} to {df[col].max()}")
            except:
                pass

except Exception as e:
    print(f"Error analyzing file: {e}")
