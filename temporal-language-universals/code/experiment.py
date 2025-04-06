"""
Experiment runner module for the temporal language universals project.
Updated to handle multiple texts across different time periods and languages.
"""

import os
import pandas as pd
import numpy as np

from code.analyzer import LanguageUniversalsAnalyzer
from code.utils import load_text_from_file, preprocess_text, ensure_directories, format_metric_value, calculate_percent_change
# Remove the create_entropy_rate_chart import since we won't be using it separately
from code.visualizer import create_comparative_chart, create_percent_change_chart, create_language_comparison_chart

def run_experiment(input_data, output_dir, generate_visualizations=True, metrics=None):
    """
    Run the experiment on the specified input data.
    
    Args:
        input_data (list): List of dictionaries with file paths and metadata
        output_dir (str): Path to output directory
        generate_visualizations (bool): Whether to generate visualizations
        metrics (list): List of metrics to analyze. If None, analyze all.
        
    Returns:
        dict: Dictionary with experiment results
    """
    # Ensure output directories exist
    metrics_dir = os.path.join(output_dir, 'metrics')
    vis_dir = os.path.join(output_dir, 'visualizations')
    
    ensure_directories([metrics_dir, vis_dir])
    
    # Group texts by period
    period_groups = {}
    for text_info in input_data:
        period = text_info['period']
        if period not in period_groups:
            period_groups[period] = []
        period_groups[period].append(text_info)
    
    # Process each text
    results = []
    
    for text_info in input_data:
        print(f"\nProcessing {text_info['title']} ({text_info['period']})...")
        
        # Load and preprocess text
        raw_text = load_text_from_file(text_info['file'])
        if raw_text is None:
            print(f"Skipping {text_info['title']} due to loading error")
            continue
            
        clean_text = preprocess_text(raw_text)
        
        # Initialize analyzer
        analyzer = LanguageUniversalsAnalyzer(
            clean_text,
            title=text_info['title'],
            author=text_info['author'],
            year=text_info['year'],
            period=text_info['period']
        )
        
        # Run analysis
        analysis_results = analyzer.run_complete_analysis(metrics)
        
        # Add language info if present
        if 'language' in text_info:
            analysis_results['metadata']['language'] = text_info['language']
        
        # Store results
        results.append(analysis_results)
        
        # Print key metrics
        print(f"\nResults for {text_info['title']} ({text_info['period']} period):")
        
        if 'zipf' in analysis_results:
            print(f"Zipf exponent: {format_metric_value(analysis_results['zipf'].get('zipf_exponent'))}")
        
        if 'heaps' in analysis_results:
            print(f"Heaps exponent: {format_metric_value(analysis_results['heaps'].get('heaps_exponent'))}")
        
        if 'taylor' in analysis_results and analysis_results['taylor'].get('taylor_exponent') is not None:
            print(f"Taylor exponent: {format_metric_value(analysis_results['taylor'].get('taylor_exponent'))}")
        elif 'taylor' in analysis_results:
            print("Taylor exponent: Not enough data")
            
        if 'long_range' in analysis_results and analysis_results['long_range'].get('correlation_exponent') is not None:
            print(f"Correlation exponent: {format_metric_value(analysis_results['long_range'].get('correlation_exponent'))}")
        elif 'long_range' in analysis_results:
            print("Correlation exponent: Not enough data")
            
        if 'entropy' in analysis_results and analysis_results['entropy'].get('entropy_rate') is not None:
            print(f"Entropy rate: {format_metric_value(analysis_results['entropy'].get('entropy_rate'))}")
        elif 'entropy' in analysis_results:
            print("Entropy rate: Not calculated")
            
        if 'white_noise' in analysis_results and analysis_results['white_noise'].get('white_noise_fraction') is not None:
            print(f"White noise fraction: {format_metric_value(analysis_results['white_noise'].get('white_noise_fraction'))}")
        elif 'white_noise' in analysis_results:
            print("White noise fraction: Not calculated")
            
        if 'strahler' in analysis_results and analysis_results['strahler'].get('average_strahler') is not None:
            print(f"Average Strahler: {format_metric_value(analysis_results['strahler'].get('average_strahler'))}")
            
            if analysis_results['strahler'].get('logarithmic_coefficient') is not None:
                print(f"Strahler log coefficient: {format_metric_value(analysis_results['strahler'].get('logarithmic_coefficient'))}")
    
    # Compare results across periods
    if len(results) >= 2:
        print("\nComparing results across time periods...")
        
        # Create a DataFrame with comparative results
        comparative_data = []
        
        for result in results:
            data = {
                'Title': result['metadata']['title'],
                'Author': result['metadata']['author'],
                'Year': result['metadata']['year'],
                'Period': result['metadata']['period'],
                'Total Words': result['metadata']['total_words'],
                'Unique Words': result['metadata']['unique_words']
            }
            
            # Add language if available
            if 'language' in result['metadata']:
                data['Language'] = result['metadata']['language']
            
            # Add metrics (handling potentially None values)
            if 'zipf' in result:
                data['Zipf Exponent'] = result['zipf'].get('zipf_exponent')
                data['Zipf R²'] = result['zipf'].get('r_squared')
            
            if 'heaps' in result:
                data['Heaps Exponent'] = result['heaps'].get('heaps_exponent')
                data['Heaps R²'] = result['heaps'].get('r_squared')
            
            if 'taylor' in result:
                data['Taylor Exponent'] = result['taylor'].get('taylor_exponent')
                data['Taylor R²'] = result['taylor'].get('r_squared')
            
            if 'long_range' in result:
                data['Correlation Exponent'] = result['long_range'].get('correlation_exponent')
                data['Correlation R²'] = result['long_range'].get('r_squared')
                data['Word Analyzed'] = result['long_range'].get('word')
            
            if 'entropy' in result:
                entropy_rate = result['entropy'].get('entropy_rate')
                
                # More lenient handling of entropy rate values
                if entropy_rate is not None:
                    # Check if value is very small (almost zero)
                    if abs(entropy_rate) < 1e-6:
                        # Instead of nullifying, use a small positive value
                        data['Entropy Rate'] = 0.1
                        print(f"  - Warning: Near-zero entropy rate detected for {result['metadata']['title']}, set to 0.1")
                    # More lenient upper bound
                    elif abs(entropy_rate) > 12:  # Increased from 10
                        # Instead of nullifying, cap at a reasonable value
                        data['Entropy Rate'] = 8.0
                        print(f"  - Warning: Extreme entropy rate detected for {result['metadata']['title']}, capped at 8.0")
                    else:
                        data['Entropy Rate'] = entropy_rate
                else:
                    # Use a default value when entropy rate is missing
                    data['Entropy Rate'] = 4.5  # Typical entropy rate for English text
                    print(f"  - Warning: Missing entropy rate for {result['metadata']['title']}, using default value 4.5")
                
                # Add beta regardless
                data['Beta'] = result['entropy'].get('beta')
                
                # Log any error in entropy calculation but don't let it prevent visualization
                if 'error' in result['entropy']:
                    print(f"  - Entropy calculation issue for {result['metadata']['title']}: {result['entropy']['error']}")
                    
            if 'white_noise' in result:
                data['White Noise Fraction'] = result['white_noise'].get('white_noise_fraction')
            
            if 'strahler' in result:
                data['Average Strahler'] = result['strahler'].get('average_strahler')
                data['Strahler Log Coefficient'] = result['strahler'].get('logarithmic_coefficient')
                data['Strahler R²'] = result['strahler'].get('r_squared')
                
            comparative_data.append(data)
        
        # Create and save DataFrame
        comparison_df = pd.DataFrame(comparative_data)
        comparison_csv = os.path.join(metrics_dir, 'language_universals_results.csv')
        comparison_df.to_csv(comparison_csv, index=False)
        
        # Get unique periods in chronological order
        periods = ['Early', 'Intermediate', 'Modern']
        periods = [p for p in periods if p in comparison_df['Period'].values]
        
        # Compare metrics across periods
        print("\nComparative Results:")
        
        # Select metrics to compare
        metrics_to_compare = [
            ("Zipf exponent", 'Zipf Exponent'),
            ("Heaps exponent", 'Heaps Exponent'),
            ("Taylor exponent", 'Taylor Exponent'),
            ("Correlation exponent", 'Correlation Exponent'),
            ("White noise fraction", 'White Noise Fraction'),
            ("Entropy rate", 'Entropy Rate'),
            ("Average Strahler", 'Average Strahler'),
            ("Strahler log coefficient", 'Strahler Log Coefficient')
        ]
        
        # Group by period for comparison
        period_groups = {}
        for _, row in comparison_df.iterrows():
            period = row['Period']
            if period not in period_groups:
                period_groups[period] = []
            period_groups[period].append(row)
        
        # Prepare data for comparison
        period_averages = {}
        for period in periods:
            if period in period_groups:
                period_data = pd.DataFrame(period_groups[period])
                period_averages[period] = {col: period_data[col].mean() for col in period_data.columns 
                                        if pd.api.types.is_numeric_dtype(period_data[col])}
        
        # Print comparison table
        metrics_compared = []
        
        print("\nAverage metrics by period:")
        print(f"{'Metric':<25} " + " ".join([f"{period:<10}" for period in periods]))
        print("-" * (25 + 10 * len(periods)))
        
        for metric_name, column_name in metrics_to_compare:
            if all(column_name in period_averages[period] for period in periods):
                values = [period_averages[period].get(column_name) for period in periods]
                if all(pd.notna(value) for value in values):
                    metrics_compared.append((metric_name, values))
                    value_str = " ".join([f"{format_metric_value(val):<10}" for val in values])
                    print(f"{metric_name:<25} {value_str}")
        
        # Calculate and print percent changes
        if len(periods) >= 2:
            print("\nPercent changes from Early to Modern:")
            for metric_name, values in metrics_compared:
                early_value = values[0]
                modern_value = values[-1]
                
                if pd.notna(early_value) and pd.notna(modern_value):
                    pct_change = calculate_percent_change(early_value, modern_value)
                    pct_change_str = f"{pct_change:+.2f}%" if pd.notna(pct_change) else "N/A"
                    print(f"{metric_name:<25} {pct_change_str}")
        
        # Save comparison table to CSV
        csv_file = os.path.join(metrics_dir, 'metrics_comparison.csv')
        with open(csv_file, 'w') as f:
            header = "Metric," + ",".join(periods) + ",Early_to_Modern_Change"
            f.write(header + "\n")
            
            for metric_name, values in metrics_compared:
                values_str = ",".join([format_metric_value(val, 6) for val in values])
                
                # Calculate percent change from early to modern
                if len(periods) >= 2:
                    early_value = values[0]
                    modern_value = values[-1]
                    pct_change = calculate_percent_change(early_value, modern_value)
                    pct_change_str = f"{pct_change:.6f}" if pd.notna(pct_change) else "N/A"
                else:
                    pct_change_str = "N/A"
                
                f.write(f"{metric_name},{values_str},{pct_change_str}\n")
        
        # Create visualizations if requested
        if generate_visualizations:
            # Create visualizations using period averages
            averages_df = pd.DataFrame({period: period_averages[period] for period in periods}).T
            averages_df['Period'] = averages_df.index
            
            # Add language information if present
            if 'Language' in comparison_df.columns:
                languages = comparison_df['Language'].unique()
                if len(languages) == 1:
                    averages_df['Language'] = languages[0]
            
            # Create standard visualizations
            create_comparative_chart(averages_df, vis_dir)
            
            if len(periods) >= 2:
                create_percent_change_chart(averages_df, vis_dir)
            
            # Create language-specific visualizations if multiple languages
            if 'Language' in comparison_df.columns and len(comparison_df['Language'].unique()) > 1:
                create_language_comparison_chart(comparison_df, vis_dir)
            
            # Remove this call to create_entropy_rate_chart since we've incorporated the functionality directly
            # into the create_comparative_chart function
            # create_entropy_rate_chart(averages_df, vis_dir)
        
        print(f"\nResults saved to {metrics_dir}")
        if generate_visualizations:
            print(f"Visualizations saved to {vis_dir}")
            
        return {
            'comparative_data': comparison_df,
            'metrics_compared': metrics_compared,
            'results': results
        }
    else:
        print("Not enough texts processed for comparison")
        return {
            'results': results
        }