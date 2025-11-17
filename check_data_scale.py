import pickle
import numpy as np

print("Checking KIBA and DAVIS data ranges...")

# Load KIBA
with open('data/kiba/Y', 'rb') as f:
    kiba_y = pickle.load(f, encoding='latin1')
print(f"\nKIBA:")
print(f"  Type: {type(kiba_y)}")
print(f"  Shape: {kiba_y.shape}")
print(f"  Range: {kiba_y.min():.2f} to {kiba_y.max():.2f}")
print(f"  Mean: {kiba_y.mean():.2f}")
print(f"  Std: {kiba_y.std():.2f}")

# Load DAVIS
with open('data/davis/Y', 'rb') as f:
    davis_y = pickle.load(f, encoding='latin1')
print(f"\nDAVIS:")
print(f"  Type: {type(davis_y)}")
print(f"  Shape: {davis_y.shape}")
print(f"  Range: {davis_y.min():.2f} to {davis_y.max():.2f}")
print(f"  Mean: {davis_y.mean():.2f}")
print(f"  Std: {davis_y.std():.2f}")

print(f"\nScale difference: DAVIS/KIBA = {davis_y.mean() / kiba_y.mean():.2f}x")
