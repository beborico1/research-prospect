"""
Statistical metrics module for language universals analysis.
Contains implementations of Zipf's law, Heaps' law, Taylor's law,
long-range correlation, and other statistical language properties.
"""

import numpy as np
from scipy import stats

def calculate_long_range_correlation(words, word, max_distance=100):
    """
    Calculate the autocorrelation function for return intervals.
    
    Args:
        words (list): List of words in the text
        word (str): Word to analyze
        max_distance (int): Maximum lag distance to consider
        
    Returns:
        dict: Results including correlation_exponent, r_squared, and std_error
    """
    # Find positions of the word
    positions = [i for i, w in enumerate(words) if w == word]

    if len(positions) < 10:  # Not enough occurrences for meaningful analysis
        return {
            'word': word,
            'correlation_exponent': None,
            'r_squared': None,
            'std_error': None
        }

    # Calculate return intervals
    intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

    # Calculate autocorrelation function
    acf = []
    mean_interval = np.mean(intervals)
    
    for lag in range(1, min(max_distance, len(intervals)//4)):
        # Calculate autocorrelation at this lag
        numerator = np.sum([(intervals[i] - mean_interval) * (intervals[i+lag] - mean_interval)
                           for i in range(len(intervals)-lag)])
        denominator = np.sum([(interval - mean_interval)**2 for interval in intervals])

        if denominator > 0:
            acf.append(numerator / denominator)
        else:
            acf.append(0)

    # Fit power law to autocorrelation function
    lags = np.arange(1, len(acf) + 1)
    valid_points = [(lag, ac) for lag, ac in zip(lags, acf) if ac > 0]

    if len(valid_points) > 5:  # Need enough points for a reasonable fit
        log_lags = np.log([p[0] for p in valid_points])
        log_acf = np.log([p[1] for p in valid_points])

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_acf)
        correlation_exponent = -slope
        r_squared = r_value**2

        return {
            'word': word,
            'correlation_exponent': correlation_exponent,
            'r_squared': r_squared,
            'std_error': std_err
        }
    else:
        return {
            'word': word,
            'correlation_exponent': None,
            'r_squared': None,
            'std_error': None
        }

def calculate_white_noise_fraction(words, word):
    """
    Estimate the white noise fraction from the autocorrelation at lag 1.
    Uses the method from Tanaka-Ishii & Bunde (2016).
    
    Args:
        words (list): List of words in the text
        word (str): Word to analyze
        
    Returns:
        dict: Results including white_noise_fraction and C1
    """
    # Find positions of the word
    positions = [i for i, w in enumerate(words) if w == word]

    if len(positions) < 10:  # Not enough occurrences for meaningful analysis
        return {
            'word': word,
            'white_noise_fraction': None,
            'C1': None
        }

    # Calculate return intervals
    intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

    # Calculate autocorrelation at lag 1
    mean_interval = np.mean(intervals)
    numerator = np.sum([(intervals[i] - mean_interval) * (intervals[i+1] - mean_interval)
                       for i in range(len(intervals)-1)])
    denominator = np.sum([(interval - mean_interval)**2 for interval in intervals])

    C1 = numerator / denominator if denominator > 0 else 0

    # Calculate white noise fraction using the formula from Tanaka-Ishii & Bunde
    # a = 1 / (1 + √[C1/(1-C1)])
    if C1 > 0 and C1 < 1:
        white_noise_fraction = 1 / (1 + np.sqrt(C1 / (1 - C1)))
    else:
        white_noise_fraction = 0.5  # Default value when formula cannot be applied

    return {
        'word': word,
        'white_noise_fraction': white_noise_fraction,
        'C1': C1
    }