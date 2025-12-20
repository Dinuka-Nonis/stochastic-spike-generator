import numpy as np
from typing import Tuple
from scipy import stats


def coefficient_of_variation(spike_times: np.ndarray) -> float:
    """
    Calculate the Coefficient of Variation (CV) of interspike intervals.
    
    CV quantifies the regularity of spike timing:
    - CV = 1: Pure randomness (Poisson process)
    - CV < 1: More regular than random (bursting suppressed, e.g., by refractoriness)
    - CV > 1: More variable than random (bursting, rate modulation)

    Args:
        spike_times (np.ndarray): Array of spike times in seconds

    Returns:
        float: Coefficient of variation of interspike intervals
    """
    if len(spike_times) < 2:
        return np.nan
    
    isis = np.diff(spike_times)

    mean_isi = np.mean(isis)
    if mean_isi == 0:
        return np.nan
    
    cv = np.std(isis) / mean_isi

    return cv


def fano_factor(
        spike_times: np.ndarray,
        duration: float,
        window_size: float
) -> float:
    """
    Calculate the Fano Factor for spikes in non-overlapping time windows.
    
    Fano Factor = Variance(spike count) / Mean(spike count)
    
    The Fano factor reveals temporal structure at different timescales:
    - FF = 1: Poisson process (variance = mean)
    - FF < 1: More regular than random (refractoriness suppresses variability)
    - FF > 1: More variable than random (bursting, rate changes)

    Args:
        spike_times (np.ndarray): Array of spike times in seconds
        duration (float): Total duration of recording
        window_size (float): Size of time windows in seconds

    Returns:
        float: Fano factor      
    """
    if window_size >= duration:
        raise ValueError("window_size must be smaller than duration")
    
    n_windows = int(np.floor(duration / window_size))

    if n_windows < 2:
        return np.nan
    
    spike_counts = np.zeros(n_windows)

    # FIX: Changed variable name from 'spike_times' to 'spike_time' to avoid shadowing
    for spike_time in spike_times:
        window_idx = int(np.floor(spike_time / window_size))
        if window_idx < n_windows:
            spike_counts[window_idx] += 1

    mean_count = np.mean(spike_counts)

    if mean_count == 0:
        return np.nan
    
    variance = np.var(spike_counts)
    ff = variance / mean_count

    return ff


def fano_factor_across_timescales(
        spike_times: np.ndarray,
        duration: float,
        window_sizes: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Fano Factor across multiple timescales.
    
    This reveals how temporal structure changes with observation window size,
    helping identify bursting patterns or rate modulation.
    """
    if window_sizes is None:
        # Logarithmically spaced windows from 1ms to duration/10
        window_sizes = np.logspace(-3, np.log10(duration / 10), 15)
    
    fano_factors = np.zeros(len(window_sizes))
    
    for i, ws in enumerate(window_sizes):
        ff = fano_factor(spike_times, duration, ws)
        fano_factors[i] = ff
    
    return window_sizes, fano_factors


def isi_histogram(
        spike_times: np.ndarray,
        n_bins: int = 50,
        log_scale: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate histogram of interspike intervals.
    
    The ISI histogram reveals the temporal structure of spike trains:
    - Poisson: Exponential decay on log scale (straight line)
    - With refractoriness: "Hole" at short intervals (< refractory period)
    - Bursty: Multiple peaks indicating preferred interspike intervals

    Args:
        spike_times (np.ndarray): Array of spike times in seconds
        n_bins (int): Number of histogram bins
        log_scale (bool): If True, use logarithmic bin spacing

    Returns:
        Tuple[np.ndarray, np.ndarray]: Histogram counts and bin edges
    """

    if len(spike_times) < 2:
        return np.array([]), np.array([])  # FIX: Changed np.ndarray([]) to np.array([])
    
    isis = np.diff(spike_times)

    if log_scale:
        min_isi = np.min(isis[isis > 0])
        max_isi = np.max(isis)
        bin_edges = np.logspace(np.log10(min_isi), np.log10(max_isi), n_bins + 1)
    else:
        bin_edges = np.linspace(0, np.max(isis), n_bins + 1)
    
    # FIX: Moved histogram calculation OUTSIDE the else block so it works for both cases
    hist, _ = np.histogram(isis, bins=bin_edges)
    
    return hist, bin_edges


def autocorrelogram(
    spike_times: np.ndarray,
    max_lag: float = 0.1,
    bin_size: float = 0.001
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spike time autocorrelogram.
    
    The autocorrelogram shows correlation between spike times and themselves
    at various delays, revealing refractory periods and burst structure.
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times in seconds
    max_lag : float, default=0.1
        Maximum lag to compute in seconds
    bin_size : float, default=0.001
        Bin size (temporal resolution) in seconds
    
    Returns
    -------
    autocorr : np.ndarray
        Autocorrelogram values
    lags : np.ndarray
        Lag times corresponding to autocorr values
    """
    
    if len(spike_times) < 2:
        return np.array([]), np.array([])
    
    n_bins = int(np.ceil(max_lag / bin_size))
    autocorr = np.zeros(2 * n_bins + 1)
    
    # For each spike pair, increment bin at their time difference
    for i, t1 in enumerate(spike_times):
        for t2 in spike_times:
            lag = t2 - t1
            if abs(lag) <= max_lag:
                bin_idx = int(np.round(lag / bin_size)) + n_bins
                autocorr[bin_idx] += 1
    
    lags = np.arange(-n_bins, n_bins + 1) * bin_size
    
    return autocorr, lags


def firing_rate_estimate(
    spike_times: np.ndarray,
    duration: float,
    method: str = "overall"
) -> float:
    """
    Estimate firing rate from spike times.
    
    Parameters
    ----------
    spike_times : np.ndarray
        Array of spike times in seconds
    duration : float
        Total recording duration in seconds
    method : str, default="overall"
        "overall": Total spikes / duration
        "isi": 1 / mean(ISI)
    
    Returns
    -------
    rate : float
        Firing rate in Hz
    """
    
    if method == "overall":
        return len(spike_times) / duration
    elif method == "isi":
        if len(spike_times) < 2:
            return np.nan
        mean_isi = np.mean(np.diff(spike_times))
        if mean_isi > 0:
            return 1.0 / mean_isi
        else:
            return np.nan
    else:
        raise ValueError(f"Unknown method: {method}")