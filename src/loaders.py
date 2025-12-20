import numpy as np
import scipy.io as sio
from typing import Tuple, Optional
import os

def load_mat_spike_data(
        filepath: str,
        spike_var: str = 'rho',
        stimulus_var: Optional[str] = 'stim',
        sampling_rate: float = 1000.0
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Load spike data from MATLAB .mat file

    Assumes binary spike train (0s and 1s) where 1 = spike event.

   """
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        data = sio.loadmat(filepath)
    except Exception as e:
        raise ValueError(f"Error loading  .mat file: {e}")
    
    if spike_var not in data:
        raise ValueError(f"Variable '{spike_var}' not found in .mat file")
    
    rho = data[spike_var].flatten()  #flatten to 1D

    unique_vals= np.unique(rho)
    if not np.ndarray(unique_vals, [0, 1]) and not np.array_equal(unique_vals, [1]):
        raise ValueError(f"Spike variable not binary. unique values: {unique_vals}")
    
    # Convert binary array to spike times
    spike_indices = np.where(rho == 1)[0]
    spike_times = spike_indices / sampling_rate

    stimulus = None
    if stimulus_var is not None:
        if stimulus_var in data:
            stimulus = data[stimulus_var].flatten()
        else:
            print(f"Warning: Variable '{stimulus_var}' not found, skipping stimulus")
    
    # Create metadata
    duration = len(rho) / sampling_rate
    num_spikes = len(spike_times)
    firing_rate = num_spikes / duration
    
    metadata = {
        'sampling_rate': sampling_rate,
        'duration': duration,
        'num_spikes': num_spikes,
        'firing_rate': firing_rate,
        'file': filepath,
        'total_samples': len(rho)
    }
    
    return spike_times, stimulus, metadata

def print_data_summary(
    spike_times: np.ndarray,
    metadata: dict
) -> None:
    """
    Print a summary of loaded spike data.
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times
    metadata : dict
        Metadata dictionary from load function
    """
    
    print("="*70)
    print("NEURAL RECORDING SUMMARY")
    print("="*70)
    
    print(f"\nFile: {metadata['file']}")
    print(f"Duration: {metadata['duration']:.1f} seconds ({metadata['duration']/60:.2f} minutes)")
    print(f"Sampling rate: {metadata['sampling_rate']:.0f} Hz")
    print(f"Total samples: {metadata['total_samples']}")
    
    print(f"\nSpike Statistics:")
    print(f"  Total spikes: {metadata['num_spikes']}")
    print(f"  Firing rate: {metadata['firing_rate']:.1f} Hz")
    print(f"  Spike time range: {spike_times[0]:.4f} to {spike_times[-1]:.4f} s")
    
    if len(spike_times) > 1:
        isis = np.diff(spike_times)
        print(f"\nInterspike Interval Statistics:")
        print(f"  Mean ISI: {np.mean(isis)*1000:.3f} ms")
        print(f"  Min ISI: {np.min(isis)*1000:.3f} ms")
        print(f"  Max ISI: {np.max(isis)*1000:.3f} ms")
        print(f"  Median ISI: {np.median(isis)*1000:.3f} ms")
    
    print("="*70)