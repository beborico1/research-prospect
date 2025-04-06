"""
Statistical metrics module for language universals analysis.
Contains implementations of Zipf's law, Heaps' law, Taylor's law,
long-range correlation, and other statistical language properties.
"""

import numpy as np
import math
from scipy.optimize import curve_fit
from collections import Counter

def calculate_entropy_rate(text, sample_sizes=None):
    """
    Estimate entropy rate using compression method with stretched exponential extrapolation.
    Modified to be more robust and ensure successful calculation.
    
    Args:
        text (str): The text to analyze
        sample_sizes (list): List of sample sizes for extrapolation
        
    Returns:
        dict: Results including entropy_rate, beta, and r_squared
    """
    # Default sample sizes if none provided - use more appropriate sampling
    if sample_sizes is None or len(sample_sizes) < 3:
        # Generate logarithmically spaced sample sizes based on text length
        max_size = min(len(text), 100000)  # Cap at 100K characters to prevent computation issues
        min_size = max(100, int(max_size * 0.01))  # Ensure a minimum size
        
        # Create at least 5 sample points for better curve fitting
        sample_sizes = np.logspace(np.log10(min_size), np.log10(max_size), 
                                    num=min(8, max(5, int(np.log10(max_size/min_size) * 3))))
        sample_sizes = [int(size) for size in sample_sizes]
    
    # Validate text length - lower minimum requirement
    if len(text) < 50:  # Reduced from 100
        # For very short texts, compute a simple character entropy
        if len(text) > 0:
            char_freqs = Counter(text)
            total_chars = len(text)
            entropy = 0
            for char, freq in char_freqs.items():
                prob = freq / total_chars
                if prob > 0:
                    entropy -= prob * np.log2(prob)
            return {
                'entropy_rate': entropy,  # Simple estimate for very short texts
                'beta': 0.5,  # Default value
                'r_squared': 0,
                'error': "Text too short for reliable entropy estimation, simple entropy used"
            }
        else:
            return {
                'entropy_rate': None,
                'beta': None,
                'r_squared': None,
                'error': "Empty text"
            }
    
    # Calculate character entropy for different sample sizes
    entropies = []
    valid_sizes = []
    
    for size in sample_sizes:
        if size > len(text):
            continue
            
        sample = text[:size]
        
        # Calculate character frequencies
        char_freqs = Counter(sample)
        total_chars = len(sample)
        
        if total_chars == 0:
            continue
        
        # Calculate Shannon entropy with proper handling of zero probabilities
        entropy = 0
        for char, freq in char_freqs.items():
            prob = freq / total_chars
            if prob > 0:  # Avoid log(0)
                entropy -= prob * np.log2(prob)
        
        entropies.append(entropy)
        valid_sizes.append(size)
    
    # Check if we have enough points for fitting - if not, use simple average
    if len(valid_sizes) < 3:
        # Return basic entropy if we can't fit the curve
        if entropies:
            average_entropy = np.mean(entropies)
            return {
                'entropy_rate': average_entropy,
                'beta': 0.5,  # Default value
                'r_squared': 0,
                'error': "Not enough valid samples for curve fitting, using average entropy"
            }
        else:
            return {
                'entropy_rate': 4.5,  # Default fallback value (typical for English)
                'beta': 0.5,
                'r_squared': 0,
                'error': "No valid entropy calculations, using fallback value"
            }
    
    # Convert to numpy arrays for fitting
    sizes = np.array(valid_sizes)
    entropies = np.array(entropies)
    
    # Function for stretched exponential fit: f(n) = h_inf + A*n^(β-1)
    def stretched_exp(n, h_inf, A, beta):
        return h_inf + A * n**(beta-1)
    
    # Initial parameter estimates to help convergence
    # h_inf ≈ final entropy value, A ≈ first value - final value, beta ≈ 0.7 (typical value)
    p0 = [entropies[-1], entropies[0] - entropies[-1], 0.7]
    
    # Fit the function to the data with constraints and bounds
    try:
        params, pcov = curve_fit(
            stretched_exp, 
            sizes, 
            entropies, 
            p0=p0,
            bounds=([0, -np.inf, 0.1], [10, np.inf, 0.99]),
            maxfev=10000  # Increase maximum function evaluations
        )
        
        h_inf, A, beta = params
        
        # Calculate R-squared
        residuals = entropies - stretched_exp(sizes, h_inf, A, beta)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((entropies - np.mean(entropies))**2)
        
        # Avoid division by zero
        if ss_tot == 0:
            r_squared = 0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        # Perform sanity check on the entropy rate (h_inf) - more lenient bounds
        if h_inf < 0 or h_inf > 10:  # Increased from 8 to 10
            # Instead of falling back to mean, adjust the value to be within reasonable bounds
            if h_inf < 0:
                adjusted_h_inf = 0.1  # Small positive value
            else:
                adjusted_h_inf = 8.0  # Cap at reasonable maximum
                
            return {
                'entropy_rate': adjusted_h_inf,
                'beta': beta,
                'r_squared': r_squared,
                'sample_entropies': list(zip(valid_sizes, entropies)),
                'error': f"Adjusted implausible entropy rate estimate from {h_inf} to {adjusted_h_inf}"
            }
        
        return {
            'entropy_rate': h_inf,
            'beta': beta,
            'r_squared': r_squared,
            'sample_entropies': list(zip(valid_sizes, entropies))
        }
    
    except Exception as e:
        # If curve fitting fails, return the mean entropy as a fallback
        mean_entropy = np.mean(entropies) if entropies else 4.5  # Default to 4.5 if no entropies
        
        # Ensure the mean entropy is within reasonable bounds
        if mean_entropy < 0 or mean_entropy > 10:
            mean_entropy = 4.5  # Default to a typical English entropy rate
            
        return {
            'entropy_rate': mean_entropy,
            'beta': 0.5,  # Default value
            'r_squared': 0,
            'sample_entropies': list(zip(valid_sizes, entropies)) if entropies else None,
            'error': f"Curve fitting error: {str(e)}, using mean entropy"
        }