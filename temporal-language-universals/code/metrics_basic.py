"""
Statistical metrics module for language universals analysis.
Contains implementations of Zipf's law, Heaps' law, Taylor's law,
long-range correlation, and other statistical language properties.
"""

import numpy as np
import math
from scipy import stats
from nltk.tokenize import word_tokenize
from collections import Counter

def calculate_zipf_exponent(ranks, frequencies):
    """
    Calculate Zipf's law exponent (η) where frequency ∝ rank^(-η).
    
    Args:
        ranks (numpy.ndarray): Word ranks
        frequencies (numpy.ndarray): Word frequencies
        
    Returns:
        dict: Results including zipf_exponent, r_squared, and std_error
    """
    # Take log of ranks and frequencies for linear regression
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)

    # Linear regression to find the exponent
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_frequencies)

    # Zipf's exponent is the negative of the slope
    zipf_exponent = -slope
    r_squared = r_value**2

    return {
        'zipf_exponent': zipf_exponent,
        'r_squared': r_squared,
        'std_error': std_err
    }

def calculate_heaps_exponent(words, window_size=1000):
    """
    Calculate Heaps' law exponent (ξ) where vocabulary_size ∝ text_length^(ξ).
    
    Args:
        words (list): List of words in the text
        window_size (int): Size of windows for vocabulary growth analysis
        
    Returns:
        dict: Results including heaps_exponent, r_squared, and std_error
    """
    # Calculate vocabulary growth
    text_lengths = []
    vocab_sizes = []
    total_words = len(words)

    # Track unique words seen so far
    seen_words = set()

    # Sample at different text lengths
    for i in range(0, total_words, window_size):
        current_length = min(i + window_size, total_words)
        text_lengths.append(current_length)
        
        for word in words[i:current_length]:
            seen_words.add(word)
            
        vocab_sizes.append(len(seen_words))

    # Take log for linear regression
    log_lengths = np.log(text_lengths)
    log_vocab = np.log(vocab_sizes)

    # Linear regression to find the exponent
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lengths, log_vocab)

    # Heaps' exponent is the slope
    heaps_exponent = slope
    r_squared = r_value**2

    return {
        'heaps_exponent': heaps_exponent,
        'r_squared': r_squared,
        'std_error': std_err
    }

def calculate_taylor_exponent(words, word_counts, window_size=1000, overlap=500):
    """
    Calculate Taylor's law exponent (α) where variance ∝ mean^(α).
    
    Args:
        words (list): List of words in the text
        word_counts (Counter): Counter of word frequencies
        window_size (int): Size of windows for variance analysis
        overlap (int): Overlap between consecutive windows
        
    Returns:
        dict: Results including taylor_exponent, r_squared, and std_error
    """
    # Create overlapping windows of text
    windows = []
    total_words = len(words)
    
    for i in range(0, total_words - window_size, overlap):
        windows.append(words[i:i+window_size])

    if not windows:  # Handle case where text is shorter than window_size
        return {
            'taylor_exponent': None,
            'r_squared': None,
            'std_error': None
        }

    # Calculate mean and variance of word frequencies in each window
    word_stats = {}

    # Only analyze words that appear in most windows
    min_presence = max(1, 0.7 * len(windows))

    for word in word_counts:
        if word_counts[word] > min_presence:
            counts = [window.count(word) for window in windows]
            word_stats[word] = {
                'mean': np.mean(counts),
                'variance': np.var(counts)
            }

    # Extract means and variances for regression
    means = np.array([stats['mean'] for stats in word_stats.values() if stats['mean'] > 0])
    variances = np.array([stats['variance'] for stats in word_stats.values() if stats['mean'] > 0])

    if len(means) < 2:  # Not enough data points for regression
        return {
            'taylor_exponent': None,
            'r_squared': None,
            'std_error': None
        }

    # Apply log transformation
    log_means = np.log(means)
    log_variances = np.log(variances)

    # Linear regression to find Taylor exponent
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_means, log_variances)

    # Taylor's exponent is the slope
    taylor_exponent = slope
    r_squared = r_value**2

    return {
        'taylor_exponent': taylor_exponent,
        'r_squared': r_squared,
        'std_error': std_err
    }