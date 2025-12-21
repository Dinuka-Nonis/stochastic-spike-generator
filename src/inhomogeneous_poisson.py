"""
Key concept:
  Instead of: r(t) = constant
  Use:        r(t) = f(stimulus(t))
  
Where f is the stimulus-response transfer function learned from data.
"""
import numpy as np
from typing import Tuple


def estimate_transfer_function(spike_times: np.ndarray, 
                               stimulus: np.ndarray,
                               sampling_rate: float = 1000.0,
                               n_bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Learn the stimulus-response transfer function from real data.
    
    For each stimulus value, compute average firing rate when that
    stimulus is presented. This gives us r(s) = firing rate given stimulus s.
    
    """
    
    # Create binary spike array (1 if spike, 0 otherwise)
    spike_array = np.zeros(len(stimulus), dtype=int)
    spike_indices = np.round(spike_times * sampling_rate).astype(int)
    spike_indices = spike_indices[spike_indices < len(stimulus)]
    spike_array[spike_indices] = 1
    
    # Bin stimulus values
    stimulus_min = np.min(stimulus)
    stimulus_max = np.max(stimulus)
    bin_edges = np.linspace(stimulus_min, stimulus_max, n_bins + 1)
    
    # For each stimulus bin, compute average firing rate
    firing_rates = np.zeros(n_bins)
    stimulus_values = np.zeros(n_bins)
    
    for i in range(n_bins):
        bin_center = (bin_edges[i] + bin_edges[i+1]) / 2
        stimulus_values[i] = bin_center
        
        # Find times when stimulus was in this bin
        mask = (stimulus >= bin_edges[i]) & (stimulus < bin_edges[i+1])
        
        if np.sum(mask) > 0:
            # Average firing rate in this bin (spike probability × sampling rate)
            firing_rates[i] = np.mean(spike_array[mask]) * sampling_rate
    
    return stimulus_values, firing_rates


def inhomogeneous_poisson_generator(stimulus: np.ndarray,
                                    transfer_function: Tuple[np.ndarray, np.ndarray],
                                    sampling_rate: float = 1000.0,
                                    random_seed: int = None) -> np.ndarray:
    """
    Generate spike train using inhomogeneous Poisson process.
    
    At each time step, the firing rate depends on the current stimulus:
      r(t) = transfer_function(stimulus(t))
    
    For each time step:
      - Get stimulus value s(t)
      - Interpolate to find r(s(t)) from transfer function
      - Generate spike with probability r(s(t)) × dt
    
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    stimulus_values, firing_rates = transfer_function
    dt = 1.0 / sampling_rate
    
    # Interpolate transfer function across entire stimulus range
    # For each time step, get firing rate based on current stimulus
    firing_rate_trajectory = np.interp(stimulus, stimulus_values, firing_rates)
    
    # Generate spikes: at each time step, spike with probability r(t) × dt
    spike_array = np.zeros(len(stimulus), dtype=int)
    
    for i in range(len(stimulus)):
        spike_prob = firing_rate_trajectory[i] * dt
        
        # Ensure probability is valid
        spike_prob = np.clip(spike_prob, 0, 1)
        
        # Generate spike with this probability
        if np.random.uniform(0, 1) < spike_prob:
            spike_array[i] = 1
    
    # Convert binary array to spike times
    spike_indices = np.where(spike_array == 1)[0]
    spike_times = spike_indices / sampling_rate
    
    return spike_times


def inhomogeneous_poisson_batch(stimulus: np.ndarray,
                                transfer_function: Tuple[np.ndarray, np.ndarray],
                                sampling_rate: float = 1000.0,
                                n_trials: int = 100,
                                random_seed: int = None) -> list:
    """
    Generate multiple independent inhomogeneous Poisson spike trains.
    
    """
    
    if random_seed is not None:
        np.random.seed(random_seed)
    
    spike_trains = []
    
    for _ in range(n_trials):
        spikes = inhomogeneous_poisson_generator(
            stimulus,
            transfer_function,
            sampling_rate=sampling_rate,
            random_seed=None  # Each trial different
        )
        spike_trains.append(spikes)
    
    return spike_trains