"""
Quick script to explore the contents of the .mat file
Run this in your project root directory
"""

import scipy.io as sio
import numpy as np

# Load the .mat file
mat_file = 'D:/Projects/stochastic-spike-generator/c1p8.mat'

print("="*70)
print("EXPLORING .mat FILE")
print("="*70)

# Load the file
try:
    data = sio.loadmat(mat_file)
    print(f"\n✓ File loaded successfully!")
except Exception as e:
    print(f"\n✗ Error loading file: {e}")
    exit()

# Show all variables in the file
print(f"\nVariables in the .mat file:")
print("-" * 70)
for key in data.keys():
    if not key.startswith('__'):  # Skip MATLAB metadata
        value = data[key]
        print(f"\nKey: '{key}'")
        print(f"  Type: {type(value)}")
        print(f"  Shape: {value.shape}")
        
        # Try to show first few values
        try:
            if isinstance(value, np.ndarray):
                if value.size > 0:
                    if value.dtype in [np.float64, np.float32, np.int64, np.int32]:
                        print(f"  Data type: {value.dtype}")
                        print(f"  Min: {np.min(value):.6f}")
                        print(f"  Max: {np.max(value):.6f}")
                        print(f"  Mean: {np.mean(value):.6f}")
                        
                        # Show first 10 values
                        if value.size > 10:
                            if len(value.shape) == 1:
                                print(f"  First 10 values: {value[:10]}")
                            elif len(value.shape) == 2:
                                print(f"  First row: {value[0, :10] if value.shape[1] >= 10 else value[0, :]}")
                        else:
                            print(f"  All values: {value}")
        except Exception as e:
            print(f"  (Could not display values: {e})")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

# Try to guess which variable contains spike times
print("\nLooking for spike time data...")
for key in data.keys():
    if not key.startswith('__'):
        value = data[key]
        if isinstance(value, np.ndarray) and value.size > 0:
            # Check if it looks like spike times (values between 0 and some duration)
            if value.dtype in [np.float64, np.float32]:
                min_val = np.min(value)
                max_val = np.max(value)
                
                # Reasonable range for spike times (0 to 1000 seconds)
                if 0 <= min_val < max_val <= 1000:
                    print(f"\n  ✓ '{key}' looks like spike times!")
                    print(f"    Range: {min_val:.6f} to {max_val:.6f} seconds")
                    print(f"    Count: {value.size} spikes")
                    if value.size > 0:
                        duration = max_val - min_val
                        rate = value.size / duration
                        print(f"    Duration: {duration:.2f} seconds")
                        print(f"    Firing rate: {rate:.1f} Hz")

print("\n" + "="*70)