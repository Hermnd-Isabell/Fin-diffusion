
import torch
try:
    data = torch.load("data/processed_tensors.pt")
except FileNotFoundError:
    print("File not found.")
    exit()

iv = data['iv_surface']
cond = data['conditions']

print(f"IV Shape: {iv.shape}")
print(f"Cond Shape: {cond.shape}")

if torch.isnan(iv).any():
    print("FATAL: IV Surface contains NaNs!")
    print(f"Count: {torch.isnan(iv).sum()}")
else:
    print("IV Surface is clean.")

if torch.isnan(cond).any():
    print("FATAL: Conditions contain NaNs!")
else:
    print("Conditions are clean.")

if torch.isinf(iv).any():
    print("FATAL: IV Surface contains Infs!")
else:
    print("IV Surface no Infs.")
