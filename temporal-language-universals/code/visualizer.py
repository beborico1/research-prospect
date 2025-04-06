"""
Visualization module for the temporal language universals project.
Creates plots and charts for comparing statistical properties across time periods.
This is the main facade that imports all visualization functions.
"""

# Import all functions from the visualization modules
from .visualizer_basic import (
    create_entropy_rate_chart,
    create_comparative_chart
)

from .visualizer_change import (
    create_percent_change_chart,
    create_language_comparison_chart
)

from .visualizer_specific import (
    create_metric_visualization,
    plot_zipf_distribution
)

# Export all visualization functions
__all__ = [
    'create_entropy_rate_chart',
    'create_comparative_chart',
    'create_percent_change_chart',
    'create_language_comparison_chart',
    'create_metric_visualization',
    'plot_zipf_distribution'
]