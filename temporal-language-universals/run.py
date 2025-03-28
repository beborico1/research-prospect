"""
Main entry point for the Temporal Stability of Statistical Language Universals experiment.
This script now handles multiple languages and time periods from a structured data directory.
"""

import os
import sys
import argparse
from code.experiment import run_experiment

def parse_arguments():
    """Parse command line arguments for the experiment."""
    parser = argparse.ArgumentParser(
        description='Analyze temporal stability of statistical language universals across languages')
    
    parser.add_argument('--data-dir', type=str, 
                        default='data',
                        help='Path to data directory containing time period and language subdirectories')
    
    parser.add_argument('--output', type=str, 
                        default='results',
                        help='Path to output directory')
    
    parser.add_argument('--language', type=str, 
                        default='all',
                        help='Language to analyze (en, es, jp) or "all"')
    
    parser.add_argument('--visualize', action='store_true',
                        default=True,
                        help='Generate visualizations')
    
    parser.add_argument('--metrics', type=str, 
                        default='all',
                        help='Comma-separated list of metrics to analyze (zipf,heaps,taylor,correlation,entropy,strahler,white_noise) or "all"')
    
    return parser.parse_args()

def validate_paths(args):
    """Validate data directory structure and create output directories."""
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        return False
    
    # Check if period directories exist
    periods = ['early', 'intermediate', 'modern']
    for period in periods:
        period_path = os.path.join(args.data_dir, period)
        if not os.path.exists(period_path):
            print(f"ERROR: Period directory not found: {period_path}")
            return False
    
    # Create output directories
    try:
        os.makedirs(args.output, exist_ok=True)
    except PermissionError:
        print(f"ERROR: No permission to create output directory: {args.output}")
        return False
    
    return True

def get_language_dirs(data_dir, language):
    """Get language directories to process based on user selection."""
    if language.lower() == 'all':
        # Get all language directories from the early period folder as reference
        early_dir = os.path.join(data_dir, 'early')
        return [d for d in os.listdir(early_dir) if os.path.isdir(os.path.join(early_dir, d))]
    else:
        # Process only the specified language
        return [language]

def get_language_files(data_dir, language):
    """Get files for a specific language across all time periods."""
    periods = ['early', 'intermediate', 'modern']
    files = []
    
    for period in periods:
        # Construct the path to the language folder for this period
        lang_dir = os.path.join(data_dir, period, language)
        
        # Skip if the directory doesn't exist
        if not os.path.exists(lang_dir):
            print(f"WARNING: No {language} texts found for {period} period. Skipping.")
            continue
        
        # Get all text files in this directory
        for filename in os.listdir(lang_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(lang_dir, filename)
                
                # Get base name without extension for title
                title = os.path.splitext(filename)[0].capitalize()
                
                # Add metadata
                files.append({
                    "file": file_path,
                    "title": f"{title}",
                    "author": "Unknown",  # This could be extracted from filename or a metadata file
                    "year": "",  # This could be extracted from filename or a metadata file
                    "period": period.capitalize(),
                    "language": language
                })
    
    return files

def main():
    """Main function to set up and run the experiment for all languages."""
    print("=== Temporal Stability of Statistical Language Universals ===")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Validate paths
    if not validate_paths(args):
        return 1
    
    # Determine which metrics to run
    if args.metrics.lower() == 'all':
        metrics = None  # Run all metrics
    else:
        metrics = [m.strip() for m in args.metrics.split(',')]
    
    # Get languages to process
    languages = get_language_dirs(args.data_dir, args.language)
    
    if not languages:
        print(f"ERROR: No languages found to analyze")
        return 1
    
    print(f"Analyzing languages: {', '.join(languages)}")
    
    # Process each language
    for language in languages:
        print(f"\n=== Processing {language.upper()} texts ===")
        
        # Get files for this language
        language_files = get_language_files(args.data_dir, language)
        
        if not language_files:
            print(f"ERROR: No text files found for language: {language}")
            continue
        
        # Create language-specific output directory
        language_output_dir = os.path.join(args.output, language)
        os.makedirs(language_output_dir, exist_ok=True)
        
        # Run the experiment for this language
        try:
            print(f"Found {len(language_files)} files for language {language}")
            for file_info in language_files:
                print(f"  - {file_info['period']}: {file_info['title']}")
                
            run_experiment(
                input_data=language_files,
                output_dir=language_output_dir,
                generate_visualizations=args.visualize,
                metrics=metrics
            )
            print(f"\nExperiment for {language} completed successfully!")
        except Exception as e:
            print(f"\nERROR: Experiment for {language} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll experiments completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())