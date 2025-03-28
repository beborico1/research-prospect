"""
Statistical metrics module for language universals analysis.
Contains implementations of Zipf's law, Heaps' law, Taylor's law,
long-range correlation, and other statistical language properties.
"""

import numpy as np
import math
from scipy import stats
from scipy.optimize import curve_fit
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

def calculate_entropy_rate(text, sample_sizes):
    """
    Estimate entropy rate using compression method with stretched exponential extrapolation.
    
    Args:
        text (str): The text to analyze
        sample_sizes (list): List of sample sizes for extrapolation
        
    Returns:
        dict: Results including entropy_rate, beta, and r_squared
    """
    if not sample_sizes:
        return {
            'entropy_rate': None,
            'beta': None,
            'r_squared': None
        }

    # Calculate character entropy for different sample sizes
    entropies = []
    for size in sample_sizes:
        sample = text[:size]
        # Calculate character frequencies
        char_freqs = Counter(sample)
        total_chars = len(sample)
        
        # Calculate Shannon entropy
        entropy = -sum((freq/total_chars) * np.log2(freq/total_chars) for freq in char_freqs.values())
        entropies.append(entropy)
        
    # Function for stretched exponential fit: f(n) = h_inf + A*n^(β-1)
    def stretched_exp(n, h_inf, A, beta):
        return h_inf + A * n**(beta-1)
        
    # Fit the function to the data
    try:
        params, _ = curve_fit(stretched_exp, sample_sizes, entropies, 
                            bounds=([0, -np.inf, 0], [np.inf, np.inf, 1]))
        h_inf, A, beta = params
        
        # Calculate R-squared
        residuals = entropies - stretched_exp(np.array(sample_sizes), h_inf, A, beta)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((entropies - np.mean(entropies))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'entropy_rate': h_inf,
            'beta': beta,
            'r_squared': r_squared
        }
    except:
        return {
            'entropy_rate': np.mean(entropies),  # Fallback: use mean entropy
            'beta': None,
            'r_squared': None
        }

def calculate_strahler_number(sentences):
    """
    Simplified Strahler number analysis for sentences.
    This is a basic implementation - a full implementation would require dependency parsing.
    
    Args:
        sentences (list): List of sentences to analyze
        
    Returns:
        dict: Results including average_strahler, logarithmic_coefficient, etc.
    """
    if not sentences:
        return {
            'average_strahler': None,
            'logarithmic_coefficient': None,
            'intercept': None,
            'r_squared': None
        }
    
    # Calculate a simplified Strahler-like metric
    strahler_estimates = []
    sentence_lengths = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        length = len(words)
        sentence_lengths.append(length)  # Fixed: changed from sentence_len to append length

        # Count nesting indicators
        commas = sentence.count(',')
        parentheses = sentence.count('(')
        semicolons = sentence.count(';')
        quotes = sentence.count('"') + sentence.count("'")
        dashes = sentence.count('-')

        # Simplified Strahler estimate based on length and nesting
        # In a real implementation, this would use proper parsing
        nesting_factor = commas + parentheses*2 + semicolons*2 + quotes/2 + dashes/2
        strahler_estimate = math.log2(1 + length/5 + nesting_factor/3)
        strahler_estimates.append(strahler_estimate)

    # Fit logarithmic relationship: S ≈ a log₂(L) + b
    log_lengths = np.log2(np.array(sentence_lengths))

    # Filter out any invalid values
    valid_indices = ~np.isnan(log_lengths) & ~np.isinf(log_lengths) & ~np.isnan(strahler_estimates)
    log_lengths = log_lengths[valid_indices]
    strahler_estimates = np.array(strahler_estimates)[valid_indices]
    
    if len(log_lengths) < 2:  # Not enough valid data points
        return {
            'average_strahler': np.mean(strahler_estimates) if strahler_estimates else None,
            'logarithmic_coefficient': None,
            'intercept': None,
            'r_squared': None
        }

    # Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_lengths, strahler_estimates)
    r_squared = r_value**2

    average_strahler = np.mean(strahler_estimates)
    
    return {
        'average_strahler': average_strahler,
        'logarithmic_coefficient': slope,
        'intercept': intercept,
        'r_squared': r_squared
    }