"""
Utility functions for the temporal language universals project.
Includes file loading, text preprocessing, and other helper functions.
"""

import os
import re

def preprocess_text(text):
    """
    Preprocess text to remove Project Gutenberg headers and footers and standardize formatting.
    
    Args:
        text (str): Raw text
        
    Returns:
        str: Preprocessed text
    """
    lines = text.split('\n')
    start_idx = 0
    end_idx = len(lines)

    # Find Project Gutenberg header
    for i, line in enumerate(lines):
        if "*** START OF" in line:
            start_idx = i + 1
            break
    
    # If no explicit Gutenberg header, try to find the title/chapter start
    if start_idx == 0:
        for i, line in enumerate(lines[:100]):  # Look in first 100 lines
            if "CHAPTER" in line or "Chapter" in line or "PART" in line or "Part" in line:
                start_idx = max(0, i - 5)  # Back up a few lines to include title
                break
    
    # Find Project Gutenberg footer
    for i, line in enumerate(lines):
        if "*** END OF" in line:
            end_idx = i
            break

    # Extract main text
    main_text = '\n'.join(lines[start_idx:end_idx])
    
    # Additional cleaning
    # Standardize whitespace
    main_text = re.sub(r'\s+', ' ', main_text)
    
    # Standardize quotation marks
    main_text = main_text.replace('"', '"').replace('"', '"')
    main_text = main_text.replace(''', "'").replace(''', "'")
    
    return main_text.strip()

def load_text_from_file(filepath, encoding='utf-8', verbose=True):
    """
    Load text from file with error handling for different encodings.
    
    Args:
        filepath (str): Path to the text file
        encoding (str): Encoding to use for reading the file
        verbose (bool): Whether to print status messages
        
    Returns:
        str or None: The file content, or None if loading failed
    """
    if verbose:
        print(f"Loading file: {filepath}")
    
    # Try with specified encoding
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            text = f.read()
        return text
    except UnicodeDecodeError:
        if verbose:
            print(f"  - Failed with {encoding} encoding, trying latin-1...")
        
        # Try with latin-1 encoding
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                text = f.read()
            return text
        except Exception as e:
            if verbose:
                print(f"  - Error loading file with latin-1 encoding: {e}")
            return None
    except FileNotFoundError:
        if verbose:
            print(f"  - File not found: {filepath}")
        return None
    except Exception as e:
        if verbose:
            print(f"  - Error loading file: {e}")
        return None

def ensure_directories(dirs):
    """
    Ensure that directories exist, creating them if necessary.
    
    Args:
        dirs (list): List of directory paths to ensure
        
    Returns:
        bool: True if all directories were created/exist, False otherwise
    """
    for directory in dirs:
        try:
            os.makedirs(directory, exist_ok=True)
        except Exception as e:
            print(f"Error creating directory {directory}: {e}")
            return False
    return True

def format_metric_value(value, precision=3):
    """
    Format a metric value for display, handling None values.
    
    Args:
        value: The value to format
        precision (int): Number of decimal places
        
    Returns:
        str: Formatted value
    """
    if value is None:
        return "N/A"
    elif isinstance(value, (int, float)):
        return f"{value:.{precision}f}"
    else:
        return str(value)

def calculate_percent_change(old_value, new_value):
    """
    Calculate percent change between two values.
    
    Args:
        old_value: The original value
        new_value: The new value
        
    Returns:
        float or None: Percent change, or None if old_value is 0 or None
    """
    if old_value is None or new_value is None:
        return None
    
    try:
        old_value = float(old_value)
        new_value = float(new_value)
        
        if old_value == 0:
            return None
            
        return (new_value - old_value) / old_value * 100
    except (ValueError, TypeError):
        return None