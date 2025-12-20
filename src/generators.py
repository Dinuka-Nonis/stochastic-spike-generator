"""
Spike train generation algorithms for computational neuroscience.

This module implements stochastic models for generating neural spike trains,
starting with the homogeneous Poisson process.
"""

import numpy as np
from typing import Tuple


def homogeneous_poisson(
        firing_rate: float,
        duration: float, 
        random_seed: int = None
) -> np.ndarray:
    """
    Generate a homogeneous Poisson spike train.
    
    In a Poisson process, spikes occur randomly at a constant average rate.
    The interspike intervals (ISIs) follow an exponential distribution:
    
        ISI ~ Exponential(rate=firing_rate)
    
    This can be generated from a uniform random variable U via:
        ISI = -ln(U) / firing_rate

    Args:
        firing_rate (float): Average firing rate in Hz  
        duration (float): Total duration of spike train in seconds  
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of spike times in seconds
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Validate inputs
    if firing_rate <= 0:
        raise ValueError("firing_rate must be positive")
    if duration <= 0:
        raise ValueError("duration must be positive")
    
    spike_times = []
    current_time = 0.0

    while current_time < duration:
        # Generate exponentially distributed interspike interval
        # ISI = -ln(U) / rate, where U ~ Uniform(0, 1)
        uniform_sample = np.random.uniform(0, 1)
        isi = -np.log(uniform_sample) / firing_rate

        current_time += isi

        if current_time < duration:
            spike_times.append(current_time)

    # FIX: Changed np.ndarray to np.array
    return np.array(spike_times)


def homogeneous_poisson_batch(
        firing_rate: float,
        duration: float,
        n_trials: int = 100, 
        random_seed: int = None
) -> list:
    """
    Generate multiple independent Poisson spike trains.
    
    Useful for studying variability across trials and computing
    statistics that require multiple samples.

    Args:
        firing_rate (float): Average firing rate in Hz
        duration (float): Total duration of spike train in seconds
        n_trials (int): Number of independent spike trains to generate
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        list: List of np.ndarray, each containing spike times for one trial
    """

    if random_seed is not None: 
        np.random.seed(random_seed)

    spike_trains = []
    for _ in range(n_trials):
        # FIX: Added missing 'duration' parameter
        spikes = homogeneous_poisson(firing_rate, duration, random_seed=None)
        spike_trains.append(spikes)

    return spike_trains


def interspike_intervals(spike_times: np.ndarray) -> np.ndarray:
    """
    Calculate interspike intervals (ISIs) from spike times.
    
    The ISI is the time between consecutive spikes.
    For a Poisson process, ISIs follow an exponential distribution.

    Args:
        spike_times (np.ndarray): Array of spike times (must be sorted)

    Returns:
        np.ndarray: Array of interspike intervals
    """
    if len(spike_times) < 2:
        return np.array([])
    
    # np.diff() subtracts each element from the next one
    # Example: [0.1, 0.15, 0.3] -> [0.05, 0.15]
    return np.diff(spike_times)


def homogeneous_poisson_vectorized(
        firing_rate: float,
        duration: float,
        random_seed: int = None
) -> np.ndarray:
    """
    Vectorized implementation of Poisson spike generation.
    
    More efficient for very large spike trains or high firing rates.
    Uses the relationship between Poisson point processes and uniform
    distributions on intervals.

    Args:
        firing_rate (float): Average firing rate in Hz
        duration (float): Total duration in seconds
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of spike times in seconds
        
    Notes:
        This generates a fixed number of spikes based on expected count,
        then redistributes them uniformly. For most neuroscience applications,
        the sequential method (homogeneous_poisson) is preferred because it
        naturally handles variable spike counts.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Expected number of spikes (Poisson parameter λ)
    expected_spikes = firing_rate * duration
    
    # Sample from Poisson distribution for spike count
    n_spikes = np.random.poisson(expected_spikes)

    if n_spikes == 0:
        # FIX: Changed np.ndarray([]) to np.array([])
        return np.array([])
    
    # Generate spike times uniformly distributed over duration
    spike_times = np.sort(np.random.uniform(0, duration, n_spikes))

    return spike_times