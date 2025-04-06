"""
Specific visualization functions for the temporal language universals project.
Contains functions for creating single metric visualizations and Zipf distributions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def create_metric_visualization(metric_data, metric_name, periods, output_dir):
    """
    Create visualization for a specific metric across time periods.
    
    Args:
        metric_data (dict): Dictionary with metric data for each period
        metric_name (str): Name of the metric
        periods (list): List of period names
        output_dir (str): Directory to save output files
        
    Returns:
        str: Path to the saved visualization file
    """
    plt.figure(figsize=(10, 6))
    
    # Extract data for plotting
    values = [metric_data[period]['value'] for period in periods]
    r_squared = [metric_data[period].get('r_squared', None) for period in periods]
    
    # Create bar chart
    bars = plt.bar(periods, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    
    # Add labels
    for i, v in enumerate(values):
        if v is not None:
            r2 = r_squared[i]
            if r2 is not None:
                label = f"{v:.3f}\nR²: {r2:.3f}"
            else:
                label = f"{v:.3f}"
            plt.text(i, v + max(filter(lambda x: x is not None, values)) * 0.02, 
                   label, ha='center')
    
    plt.title(f'{metric_name} Across Time Periods')
    plt.ylabel(metric_name)
    plt.grid(axis='y', alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, f"{metric_name.lower().replace(' ', '_')}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def plot_zipf_distribution(ranks, frequencies, title, period, output_dir):
    """
    Plot Zipf's law distribution for a text.
    
    Args:
        ranks (numpy.ndarray): Word ranks
        frequencies (numpy.ndarray): Word frequencies
        title (str): Title for the plot
        period (str): Time period
        output_dir (str): Directory to save output files
        
    Returns:
        str: Path to the saved visualization file
    """
    # Take log of ranks and frequencies for visualization
    log_ranks = np.log(ranks)
    log_frequencies = np.log(frequencies)

    # Linear regression to find the exponent
    slope, intercept, r_value, p_value, std_err = np.polyfit(log_ranks, log_frequencies, 1, full=True)[0:2]

    # Zipf's exponent is the negative of the slope
    zipf_exponent = -slope
    r_squared = r_value**2

    plt.figure(figsize=(10, 6))
    plt.scatter(log_ranks, log_frequencies, alpha=0.5, s=10)
    plt.plot(log_ranks, intercept + slope * log_ranks, 'r',
            label=f'Zipf\'s law: η = {zipf_exponent:.3f}, R² = {r_squared:.3f}')
    plt.xlabel('Log Rank')
    plt.ylabel('Log Frequency')
    plt.title(f'Zipf\'s Law Analysis for {title} ({period})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the figure
    output_path = os.path.join(output_dir, f"zipf_{period.lower()}_{title.replace(' ', '_').lower()}.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path