"""
Statistical metrics module for language universals analysis.
This file serves as a facade to import and expose functions from the split modules.
"""

# Import from submodules
from code.metrics_basic import (
    calculate_zipf_exponent,
    calculate_heaps_exponent,
    calculate_taylor_exponent
)

from code.metrics_correlation import (
    calculate_long_range_correlation,
    calculate_white_noise_fraction
)

from code.metrics_entropy import calculate_entropy_rate
from code.metrics_strahler import calculate_strahler_number

# Re-export all functions to maintain backward compatibility
__all__ = [
    'calculate_zipf_exponent',
    'calculate_heaps_exponent',
    'calculate_taylor_exponent',
    'calculate_long_range_correlation',
    'calculate_white_noise_fraction',
    'calculate_entropy_rate',
    'calculate_strahler_number'
]