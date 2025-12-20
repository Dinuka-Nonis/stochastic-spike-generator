"""
Deep exploration of the .mat file to understand data format
"""

import scipy.io as sio
import numpy as np

# Load the .mat file
mat_file = 'c1p8.mat'

print("="*70)
print("DEEP DATA EXPLORATION")
print("="*70)

data = sio.loadmat(mat_file)

# Examine 'rho' (neural response)
print("\n'rho' (Neural Response):")
print("-" * 70)
rho = data['rho'].flatten()  # Flatten to 1D
print(f"Shape: {rho.shape}")
print(f"Data type: {rho.dtype}")
print(f"Min: {np.min(rho)}")
print(f"Max: {np.max(rho)}")
print(f"Mean: {np.mean(rho)}")
print(f"Std: {np.std(rho)}")
print(f"\nUnique values: {np.unique(rho)[:20]}")  # First 20 unique values
print(f"Total unique values: {len(np.unique(rho))}")

# Check if binary (spikes)
if len(np.unique(rho)) == 2:
    print(f"\n✓ BINARY DATA - This is a spike train!")
    print(f"  0s (no spike): {np.sum(rho == 0)}")
    print(f"  1s (spike): {np.sum(rho == 1)}")
    num_spikes = np.sum(rho == 1)
    print(f"  Total spikes: {num_spikes}")
    
elif np.all(rho == rho.astype(int)):
    print(f"\n✓ INTEGER DATA - Likely spike counts per bin")
    print(f"  Total events: {np.sum(rho)}")
    
else:
    print(f"\n? CONTINUOUS DATA - Possible spike rate or membrane potential")

# Examine 'stim' (stimulus)
print("\n\n'stim' (Visual Stimulus):")
print("-" * 70)
stim = data['stim'].flatten()
print(f"Shape: {stim.shape}")
print(f"Data type: {stim.dtype}")
print(f"Min: {np.min(stim):.6f}")
print(f"Max: {np.max(stim):.6f}")
print(f"Mean: {np.mean(stim):.6f}")
print(f"Std: {np.std(stim):.6f}")

# Determine sampling rate and duration
print("\n\n" + "="*70)
print("INFERRED PARAMETERS")
print("="*70)

# Assume 1000 Hz sampling (common for neural recordings)
sampling_rates = [1000, 500, 100, 10]

for sr in sampling_rates:
    duration = len(rho) / sr
    if 100 < duration < 10000:  # Reasonable recording duration
        print(f"\nIf sampling rate = {sr} Hz:")
        print(f"  Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        if len(np.unique(rho)) == 2:
            firing_rate = np.sum(rho == 1) / duration
            print(f"  Firing rate: {firing_rate:.1f} Hz")
            print(f"  Total spikes: {int(np.sum(rho == 1))}")
            
            # Extract spike times
            spike_indices = np.where(rho == 1)[0]
            spike_times = spike_indices / sr
            print(f"  Spike times range: {spike_times[0]:.4f} to {spike_times[-1]:.4f} s")
            print(f"  First 10 spike times: {spike_times[:10]}")

print("\n" + "="*70)
