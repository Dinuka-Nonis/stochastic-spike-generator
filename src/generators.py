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
    Generate a homogeneous poisson spike train
    In a Poisson process, spikes occur randomly at a constant average rate.
    The interspike intervals (ISIs) follow an exponential distribution:
    

    Args:
        firing_rate (float): Average firing rate in Hz  
        duration (float): Total duration of spike train in seconds  
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        np.ndarray: _description_
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    #validate inputs
    if firing_rate <= 0:
        raise ValueError("firing_rate must be positive")
    if duration <=0:
        raise ValueError("duration must be positive")
    
    spike_times = []
    current_time =0.0

    while current_time < duration :
        # Generate exponentially distributed interspike interval
        # ISI = -ln(U) / rate, where U ~ Uniform(0, 1)
        uniform_sample = np.random.uniform(0,1)
        isi = -np.log(uniform_sample) / firing_rate

        current_time += isi

        if current_time < duration:
            spike_times.append(current_time)

    return np.ndarray(spike_times)

def homogeneous_poisson_batch(
        firing_rate: float,
        duration: float,
        n_trials: int = 100, 
        random_seed: int = None
) -> np.ndarray:
    
    """
    generate multiple independent poisson spike trains

    """

    if random_seed is not None: 
        np.random.seed(random_seed)

    spike_trains = []
    for _ in range(n_trials):
        spikes = homogeneous_poisson(firing_rate, random_seed=None)
        spike_trains.append(spikes)

    return spike_trains

def interspike_intervals(spike_times: np.ndarray) -> np.ndarray:
    """
    calculate interspike intervals (ISIs) from spike times.

    Args:
        spike_times (np.ndarray): Array of spike times (must be sorted)

    Returns:
        np.ndarray: _description_
    """
    if len(spike_times) < 2:
        return np.array([])
    
    return np.diff(spike_times)
#np.diff() is NumPy's "difference" function. It takes each element and subtracts the previous one.

def homogeneous_poisson_vectorized(
        firing_rate: float,
        duration:float,
        random_seed:int = None
) ->np.ndarray:
    """
    Vectorized method for implementation of spike train - more efficent
    for large spike trains

    Args:
        firing_rate (float): avg firing rate in Hz
        duration (float): total duration in seconds
        random_seed (int, optional): Random seed. Defaults to None.

    Returns:
        np.ndarray: _description_
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    expected_spikes = firing_rate* duration
    n_spikes = np.random.poisson(expected_spikes)

    if n_spikes == 0:
        return np.ndarray([])
    spike_times = np.sort(np.random.uniform(0, duration, n_spikes))

    return spike_times