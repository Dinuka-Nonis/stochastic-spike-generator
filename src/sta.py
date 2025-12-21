"""
Spike-Triggered Average (STA) and Stimulus-Response Analysis.

Based on:
- Dayan & Abbott "Theoretical Neuroscience" Chapter 1.4
- Exercises 8, 9, 10 on H1 neuron stimulus-response analysis

Theoretical Framework:
======================
Inhomogeneous Poisson Process:
  π[t₁,t₂,...,tₙ] = exp(-∫r(t)dt) × ∏r(tᵢ)
  
Where r(t) is the stimulus-dependent firing rate.
This allows us to understand WHAT STIMULUS PATTERNS trigger spikes.
"""

import numpy as np
from typing import Tuple, List


def compute_sta(spike_times: np.ndarray, stimulus: np.ndarray, 
                sampling_rate: float = 1000.0, duration_before_ms: float = 300) -> Tuple[np.ndarray, np.ndarray]:
    """
    Exercise 8: Single-Spike-Triggered Average Stimulus.
    
    Computes: For each spike, what was the stimulus in the window BEFORE it?
    Then averages across all spikes.
    
    Theoretical basis: Spike-triggered stimulus reveals the stimulus pattern
    that the neuron is tuned to detect.
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times in seconds
    stimulus : np.ndarray
        Stimulus values (e.g., motion direction, orientation)
    sampling_rate : float
        Sampling rate in Hz
    duration_before_ms : float
        How far back to look before spike (milliseconds)
    
    Returns
    -------
    sta : np.ndarray
        Average stimulus pattern before spikes
    time_axis : np.ndarray
        Time axis in milliseconds (going backwards)
    
    Formula (Dayan & Abbott):
    c(τ) = <s(t - τ)> for each spike at time t
    where τ ranges from 0 to duration_before_ms
    """
    
    # Convert parameters to samples
    dt = 1.0 / sampling_rate
    duration_before_samples = int(duration_before_ms / (dt * 1000))
    
    # Convert spike times to sample indices
    spike_indices = np.round(spike_times * sampling_rate).astype(int)
    
    # Filter spikes that are too close to the beginning
    spike_indices = spike_indices[spike_indices >= duration_before_samples]
    
    num_spikes = len(spike_indices)
    
    if num_spikes == 0:
        raise ValueError("No spikes far enough from start to compute STA")
    
    # Accumulate stimulus patterns before each spike
    sta = np.zeros(duration_before_samples)
    
    for spike_idx in spike_indices:
        start_idx = spike_idx - duration_before_samples
        end_idx = spike_idx
        sta += stimulus[start_idx:end_idx]
    
    # Average across all spikes
    sta = sta / num_spikes
    
    # Time axis (backwards from spike, in ms)
    time_axis = np.linspace(duration_before_ms, 0, duration_before_samples)
    
    return sta, time_axis


def compute_two_spike_sta(spike_times: np.ndarray, stimulus: np.ndarray,
                          sampling_rate: float = 1000.0,
                          duration_before_ms: float = 300,
                          max_separation_ms: float = 100,
                          min_separation_ms: float = 2) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Exercise 9: Two-Spike-Triggered Average Stimulus.
    
    For spike PAIRS separated by different time intervals,
    compute the average stimulus pattern before the SECOND spike.
    
    Theoretical basis: Examines how stimulus requirements change
    depending on the time interval between two spikes.
    Reveals temporal dependencies in spiking.
    
    Key insight: If the stimulus needed for the second spike changes
    based on the interval since the first spike, this indicates
    temporal structure (like refractory effects or adaptation).
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times in seconds
    stimulus : np.ndarray
        Stimulus values over time
    sampling_rate : float
        Sampling rate in Hz
    duration_before_ms : float
        How far back to look before second spike
    max_separation_ms : float
        Maximum separation between spikes in pair (ms)
    min_separation_ms : float
        Minimum separation between spikes in pair (ms)
    
    Returns
    -------
    sta_list : list of np.ndarray
        Average stimulus patterns for each separation
    separations : np.ndarray
        Separation values in milliseconds
    time_axis : np.ndarray
        Time axis for plotting
    
    Formula (Dayan & Abbott):
    c₂(τ,Δt) = <s(t₂ - τ)> for spike pairs (t₁, t₂) where t₂ - t₁ = Δt
    """
    
    # Convert parameters
    dt = 1.0 / sampling_rate
    duration_before_samples = int(duration_before_ms / (dt * 1000))
    
    # Create separation range
    min_sep_samples = int(min_separation_ms / (dt * 1000))
    max_sep_samples = int(max_separation_ms / (dt * 1000))
    
    separations = np.linspace(min_separation_ms, max_separation_ms,
                              max_sep_samples - min_sep_samples + 1)
    
    # Convert spike times to indices
    spike_indices = np.round(spike_times * sampling_rate).astype(int)
    
    sta_list = []
    
    # For each separation
    for sep_samples in range(min_sep_samples, max_sep_samples + 1):
        sta = np.zeros(duration_before_samples)
        count = 0
        
        # Find all spike pairs with this separation
        for second_spike_idx in spike_indices:
            first_spike_idx = second_spike_idx - sep_samples
            
            # Check if first spike exists and second spike has full history
            if (first_spike_idx >= 0 and 
                first_spike_idx in spike_indices and 
                second_spike_idx >= duration_before_samples):
                
                # Grab stimulus before second spike
                start_idx = second_spike_idx - duration_before_samples
                end_idx = second_spike_idx
                sta += stimulus[start_idx:end_idx]
                count += 1
        
        if count > 0:
            sta = sta / count
        
        sta_list.append(sta)
    
    time_axis = np.linspace(duration_before_ms, 0, duration_before_samples)
    
    return sta_list, separations, time_axis


def compute_exclusive_two_spike_sta(spike_times: np.ndarray, stimulus: np.ndarray,
                                     sampling_rate: float = 1000.0,
                                     duration_before_ms: float = 300,
                                     max_separation_ms: float = 100,
                                     min_separation_ms: float = 2) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Exercise 10: Exclusive Two-Spike-Triggered Average Stimulus.
    
    Same as Exercise 9, but ONLY counts spike pairs where there
    are NO OTHER spikes in between.
    
    Theoretical basis: This is a more stringent test of temporal structure.
    If a neuron requires STRONGER stimulus to fire a second spike
    when it has to be truly consecutive (no intervening spikes),
    this reveals the strength of temporal constraints.
    
    Key insight: Comparing regular pairs vs. exclusive pairs reveals
    whether the temporal pattern is due to:
    - Hard constraints (absolute refractory period)
    - Soft constraints (adaptation/fatigue)
    
    Parameters
    ----------
    (same as compute_two_spike_sta)
    
    Returns
    -------
    (same as compute_two_spike_sta)
    
    Formula (Dayan & Abbott - Exercise 10):
    c₂_exclusive(τ,Δt) = <s(t₂ - τ)> for spike pairs (t₁, t₂)
    where t₂ - t₁ = Δt AND no spikes between t₁ and t₂
    """
    
    # Convert parameters
    dt = 1.0 / sampling_rate
    duration_before_samples = int(duration_before_ms / (dt * 1000))
    
    # Create separation range
    min_sep_samples = int(min_separation_ms / (dt * 1000))
    max_sep_samples = int(max_separation_ms / (dt * 1000))
    
    separations = np.linspace(min_separation_ms, max_separation_ms,
                              max_sep_samples - min_sep_samples + 1)
    
    # Convert spike times to indices
    spike_indices = np.round(spike_times * sampling_rate).astype(int)
    spike_indices_set = set(spike_indices)
    
    sta_list = []
    
    # For each separation
    for sep_samples in range(min_sep_samples, max_sep_samples + 1):
        sta = np.zeros(duration_before_samples)
        count = 0
        
        # Find spike pairs with this separation AND no spikes between
        for second_spike_idx in spike_indices:
            first_spike_idx = second_spike_idx - sep_samples
            
            # Check if first spike exists
            if first_spike_idx not in spike_indices_set or second_spike_idx < duration_before_samples:
                continue
            
            # Check if there are spikes between first and second
            has_spike_between = False
            for i in range(first_spike_idx + 1, second_spike_idx):
                if i in spike_indices_set:
                    has_spike_between = True
                    break
            
            # If no spikes between, use this pair
            if not has_spike_between:
                start_idx = second_spike_idx - duration_before_samples
                end_idx = second_spike_idx
                sta += stimulus[start_idx:end_idx]
                count += 1
        
        if count > 0:
            sta = sta / count
        
        sta_list.append(sta)
    
    time_axis = np.linspace(duration_before_ms, 0, duration_before_samples)
    
    return sta_list, separations, time_axis


def compute_stimulus_response_curve(spike_times: np.ndarray, stimulus: np.ndarray,
                                     sampling_rate: float = 1000.0,
                                     n_bins: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bonus: Compute stimulus-response (transfer) function.
    
    For each stimulus value, compute the average firing rate
    when that stimulus is present.
    
    This estimates r(s) = average firing rate given stimulus = s
    which is the basis for the inhomogeneous Poisson model.
    
    Theoretical basis (Dayan & Abbott):
    The firing rate r(t) that depends on stimulus s(t) is:
    r(s) = estimated firing rate at stimulus level s
    
    This can then be used to generate spikes via:
    π[t₁,t₂,...,tₙ] = exp(-∫r(s(t))dt) × ∏r(s(tᵢ))
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times in seconds
    stimulus : np.ndarray
        Stimulus values over time
    sampling_rate : float
        Sampling rate in Hz
    n_bins : int
        Number of stimulus bins
    
    Returns
    -------
    stimulus_values : np.ndarray
        Stimulus values (bin centers)
    firing_rates : np.ndarray
        Average firing rate at each stimulus level
    """
    
    # Create binary spike array
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
            # Average firing rate in this bin
            firing_rates[i] = np.mean(spike_array[mask]) * sampling_rate
    
    return stimulus_values, firing_rates