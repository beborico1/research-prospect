"""
Change visualization functions for the temporal language universals project.
Contains functions for creating percent change charts and language comparison charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def create_percent_change_chart(comparison_df, output_dir, metric_names=None):
    """
    Create a chart showing percent changes in metrics from one period to another.
    
    Args:
        comparison_df (DataFrame): DataFrame with comparative results
        output_dir (str): Directory to save output files
        metric_names (list): List of metrics to include in the visualization
        
    Returns:
        str: Path to the saved visualization file
    """
    # Define metrics to visualize if not specified
    if metric_names is None:
        metric_names = [
            'Zipf Exponent', 
            'Heaps Exponent', 
            'Taylor Exponent', 
            'Correlation Exponent',
            'White Noise Fraction',
            'Average Strahler', 
            'Strahler Log Coefficient',
            'Entropy Rate'
        ]
    
    # Filter to only include metrics with data
    available_metrics = [m for m in metric_names if m in comparison_df.columns 
                        and not comparison_df[m].isna().all()]
    
    if not available_metrics:
        print("No metrics available for visualization")
        return None
    
    # Check if we have a 'Period' column
    if 'Period' not in comparison_df.columns:
        print("Cannot create percent change chart: 'Period' column not found")
        return None
    
    # Get periods in order
    period_order = {'Early': 0, 'Intermediate': 1, 'Modern': 2}
    periods = comparison_df['Period'].unique()
    
    # Sort periods if possible
    if all(p in period_order for p in periods):
        periods = sorted(periods, key=lambda p: period_order[p])
    
    if len(periods) < 2:
        print("Need at least two periods for percent change visualization")
        return None
    
    # Check if we have language information
    has_language = 'Language' in comparison_df.columns
    
    # Prepare data for visualization
    base_period = periods[0]  # Use first period as base
    target_period = periods[-1]  # Use last period as target
    
    # Create data for visualization
    if has_language and len(comparison_df['Language'].unique()) > 1:
        # Group by language and period, compute averages
        agg_data = comparison_df.groupby(['Language', 'Period']).mean().reset_index()
        
        # Calculate percent changes for each language
        languages = comparison_df['Language'].unique()
        change_data = []
        
        for language in languages:
            lang_data = agg_data[agg_data['Language'] == language]
            
            if len(lang_data) < 2:
                continue
                
            base_values = lang_data[lang_data['Period'] == base_period]
            target_values = lang_data[lang_data['Period'] == target_period]
            
            if len(base_values) == 0 or len(target_values) == 0:
                continue
                
            # Calculate percent changes
            for metric in available_metrics:
                base_val = base_values[metric].values[0]
                target_val = target_values[metric].values[0]
                
                if pd.notna(base_val) and pd.notna(target_val) and base_val != 0:
                    pct_change = (target_val - base_val) / base_val * 100
                    
                    # Limitar valores extremos de cambio porcentual
                    if abs(pct_change) > 1000:
                        print(f"  - Advertencia: Cambio porcentual extremo detectado para {metric} ({pct_change:.2f}%), limitado a ±1000%")
                        pct_change = 1000 if pct_change > 0 else -1000
                    
                    # Tratamiento especial para la tasa de entropía que puede dar cambios extremos
                    if metric == 'Entropy Rate' and abs(pct_change) > 200:
                        print(f"  - Advertencia: Cambio porcentual extremo en Entropy Rate ({pct_change:.2f}%), limitado a ±200%")
                        pct_change = 200 if pct_change > 0 else -200
                    
                    change_data.append({
                        'Language': language,
                        'Metric': metric,
                        'Percent Change': pct_change
                    })
        
        change_df = pd.DataFrame(change_data)
        
        if len(change_df) == 0:
            print("No valid data for percent change visualization")
            return None
            
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot changes by language and metric
        metrics = change_df['Metric'].unique()
        languages = change_df['Language'].unique()
        
        # Set up positions for grouped bars
        x = np.arange(len(metrics))
        width = 0.8 / len(languages)
        
        # Plot each language
        for i, language in enumerate(languages):
            lang_changes = change_df[change_df['Language'] == language]
            
            # Align with metrics
            values = []
            for metric in metrics:
                match = lang_changes[lang_changes['Metric'] == metric]
                if len(match) > 0:
                    values.append(match['Percent Change'].values[0])
                else:
                    values.append(0)
            
            positions = x + (i - len(languages)/2 + 0.5) * width
            plt.bar(positions, values, width, label=language)
        
        # Add labels and formatting
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.xticks(x, [m.replace(' ', '\n') for m in metrics], rotation=0)
        plt.xlabel('Metric')
        plt.ylabel(f'Percent Change ({base_period} to {target_period})')
        plt.title(f'Changes in Language Properties: {base_period} to {target_period}')
        plt.legend(title='Language')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar in plt.gca().patches:
            height = bar.get_height()
            if abs(height) > 1:  # Only label significant changes
                plt.gca().annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height > 0 else -3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom' if height > 0 else 'top',
                    fontsize=8)
    else:
        # Simple percent change chart without language grouping
        base_values = comparison_df[comparison_df['Period'] == base_period]
        target_values = comparison_df[comparison_df['Period'] == target_period]
        
        # Calculate percent changes
        change_data = []
        for metric in available_metrics:
            if metric in base_values.columns and metric in target_values.columns:
                base_val = base_values[metric].mean()
                target_val = target_values[metric].mean()
                
                if pd.notna(base_val) and pd.notna(target_val) and base_val != 0:
                    pct_change = (target_val - base_val) / base_val * 100
                    
                    # Limit extreme percent change values
                    if abs(pct_change) > 1000:
                        print(f"  - Warning: Extreme percent change detected for {metric} ({pct_change:.2f}%), limited to ±1000%")
                        pct_change = 1000 if pct_change > 0 else -1000
                    
                    # Special handling for entropy rate which can have extreme changes
                    if metric == 'Entropy Rate' and abs(pct_change) > 200:
                        print(f"  - Warning: Extreme percent change in Entropy Rate ({pct_change:.2f}%), limited to ±200%")
                        pct_change = 200 if pct_change > 0 else -200
                    
                    change_data.append({
                        'Metric': metric,
                        'Percent Change': pct_change
                    })
        
        change_df = pd.DataFrame(change_data)
        
        if len(change_df) == 0:
            print("No valid data for percent change visualization")
            return None
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Sort by absolute percent change
        change_df['AbsChange'] = change_df['Percent Change'].abs()
        change_df = change_df.sort_values('AbsChange', ascending=False)
        
        # Plot horizontal bars
        metrics = change_df['Metric'].tolist()
        values = change_df['Percent Change'].tolist()
        
        colors = ['#ff9999' if x < 0 else '#66b3ff' for x in values]
        bars = plt.barh(metrics, values, color=colors)
        
        # Add value labels
        for i, bar in enumerate(bars):
            width = bar.get_width()
            if abs(width) > 1:  # Only label significant changes
                plt.text(width + (5 if width > 0 else -5), 
                        bar.get_y() + bar.get_height()/2, 
                        f'{width:.1f}%', 
                        ha='left' if width > 0 else 'right', 
                        va='center')
        
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.xlabel(f'Percent Change ({base_period} to {target_period})')
        plt.title(f'Changes in Language Properties: {base_period} to {target_period}')
        plt.grid(axis='x', alpha=0.3)
    
    # Save the figure
    output_path = os.path.join(output_dir, 'percent_change.png')
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    return output_path

def create_language_comparison_chart(comparison_df, output_dir, metric_names=None):
    """
    Create charts comparing metrics across different languages.
    
    Args:
        comparison_df (DataFrame): DataFrame with comparative results
        output_dir (str): Directory to save output files
        metric_names (list): List of metrics to include in the visualization
        
    Returns:
        list: Paths to the saved visualization files
    """
    # Define metrics to visualize if not specified
    if metric_names is None:
        metric_names = [
            'Zipf Exponent', 
            'Heaps Exponent', 
            'Taylor Exponent', 
            'Correlation Exponent',
            'White Noise Fraction',
            'Average Strahler', 
            'Strahler Log Coefficient',
            'Entropy Rate'
        ]
    
    # Check if we have language information
    if 'Language' not in comparison_df.columns:
        print("Cannot create language comparison: 'Language' column not found")
        return None
    
    # Filter to only include metrics with data
    available_metrics = [m for m in metric_names if m in comparison_df.columns 
                        and not comparison_df[m].isna().all()]
    
    if not available_metrics:
        print("No metrics available for language comparison visualization")
        return None
    
    # Get unique languages and periods
    languages = comparison_df['Language'].unique()
    
    if 'Period' in comparison_df.columns:
        periods = comparison_df['Period'].unique()
        # Sort periods if possible
        period_order = {'Early': 0, 'Intermediate': 1, 'Modern': 2}
        if all(p in period_order for p in periods):
            periods = sorted(periods, key=lambda p: period_order[p])
    else:
        periods = ['All']
    
    # Create a separate visualization for each metric
    output_paths = []
    
    for metric in available_metrics:
        plt.figure(figsize=(12, 6))
        
        # Set up markers and colors for different languages
        markers = ['o', 's', '^', 'd', 'v', '<', '>', 'p', '*']
        colors = plt.cm.tab10(np.linspace(0, 1, len(languages)))
        
        # For each language, plot the metric across periods
        for i, language in enumerate(languages):
            lang_data = comparison_df[comparison_df['Language'] == language]
            
            if 'Period' in lang_data.columns:
                # Sort by period
                if all(p in period_order for p in lang_data['Period']):
                    lang_data = lang_data.sort_values('Period', key=lambda x: x.map(period_order))
                
                # Plot line
                if len(lang_data) > 1:
                    plt.plot(lang_data['Period'], lang_data[metric], 
                             marker=markers[i % len(markers)],
                             color=colors[i],
                             label=language,
                             linewidth=2,
                             markersize=8)
                # Plot single point
                elif len(lang_data) == 1:
                    plt.scatter(lang_data['Period'], lang_data[metric],
                               marker=markers[i % len(markers)],
                               color=colors[i],
                               label=language,
                               s=100)
            else:
                # Just plot the average value without period information
                avg_value = lang_data[metric].mean()
                plt.bar(language, avg_value, color=colors[i])
        
        plt.title(f'{metric} Comparison Across Languages')
        plt.ylabel(metric)
        if 'Period' in comparison_df.columns:
            plt.xlabel('Time Period')
            
        # Adjust y-axis limits to avoid distortion from extreme values
        if metric == 'Entropy Rate':
            # Limit entropy rate visualization to a reasonable range (0-8)
            y_min, y_max = plt.ylim()
            if y_max > 8:
                plt.ylim(top=8)
            if y_min < 0:
                plt.ylim(bottom=0)
        
        plt.grid(True, alpha=0.3)
        plt.legend(title='Language')
        
        # Add data labels
        for i, language in enumerate(languages):
            lang_data = comparison_df[comparison_df['Language'] == language]
            if 'Period' in lang_data.columns:
                for _, row in lang_data.iterrows():
                    if pd.notna(row[metric]):
                        plt.text(row['Period'], row[metric] * 1.03, 
                                f'{row[metric]:.2f}', 
                                ha='center', va='bottom',
                                fontsize=8)
        
        plt.tight_layout()
        
        # Save the figure
        output_path = os.path.join(output_dir, f'{metric.lower().replace(" ", "_")}_by_language.png')
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        output_paths.append(output_path)
    
    return output_paths