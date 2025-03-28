"""
Core analyzer module for statistical language universals.
Contains the main LanguageUniversalsAnalyzer class that orchestrates the analysis.
"""

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.data import load
from collections import Counter
import re
import numpy as np

# Import metrics functions
from code.metrics import (
    calculate_zipf_exponent,
    calculate_heaps_exponent,
    calculate_taylor_exponent,
    calculate_long_range_correlation,
    calculate_white_noise_fraction,
    calculate_entropy_rate,
    calculate_strahler_number
)

class LanguageUniversalsAnalyzer:
    """
    A class to analyze statistical language universals in texts from different time periods.
    Implements methods for calculating Zipf's law, Heaps' law, Taylor's law, and more.
    """

    def __init__(self, text, title="", author="", year="", period=""):
        """
        Initialize the analyzer with text and metadata.
        
        Args:
            text (str): The text to analyze
            title (str): Title of the text
            author (str): Author of the text
            year (int): Publication year
            period (str): Historical period (Early, Intermediate, Modern)
        """
        self.raw_text = text
        self.title = title
        self.author = author
        self.year = year
        self.period = period

        # Preprocess text
        # Preprocess text
        self.text = self._preprocess_text(text)

        # Use standard NLTK tokenizers directly
        self.words = self.text.lower().split() if self.text else []  # Simple fallback
        self.sentences = []  # Simple fallback

        from nltk.tokenize import word_tokenize, sent_tokenize
        self.words = word_tokenize(self.text.lower())
        self.sentences = sent_tokenize(self.text)
        print(f"Using NLTK tokenizers for {self.title}")
        
        self.word_counts = Counter(self.words)
        self.total_words = len(self.words)
        self.unique_words = len(self.word_counts)
        
        # Track which analyses have been performed
        self._analyzed = {
            'zipf': False,
            'heaps': False,
            'taylor': False,
            'long_range': False,
            'white_noise': False,
            'entropy': False,
            'strahler': False
        }
        
        # Store analysis results
        self.results = {
            'metadata': {
                'title': self.title,
                'author': self.author,
                'year': self.year,
                'period': self.period,
                'total_words': self.total_words,
                'unique_words': self.unique_words,
                'sentences': len(self.sentences)
            }
        }

    def _preprocess_text(self, text):
        """
        Clean and standardize text.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        return text.strip()
        
    def analyze_zipf(self, regenerate=False):
        """
        Perform Zipf's law analysis.
        
        Args:
            regenerate (bool): Whether to regenerate results if already calculated
        
        Returns:
            dict: Results of the analysis
        """
        if not self._analyzed['zipf'] or regenerate:
            frequencies = sorted(self.word_counts.values(), reverse=True)
            ranks = np.arange(1, len(frequencies) + 1)
            
            self.results['zipf'] = calculate_zipf_exponent(ranks, frequencies)
            self._analyzed['zipf'] = True
            
        return self.results['zipf']
        
    def analyze_heaps(self, window_size=1000, regenerate=False):
        """
        Perform Heaps' law analysis.
        
        Args:
            window_size (int): Size of the windows for vocabulary growth analysis
            regenerate (bool): Whether to regenerate results if already calculated
            
        Returns:
            dict: Results of the analysis
        """
        if not self._analyzed['heaps'] or regenerate:
            self.results['heaps'] = calculate_heaps_exponent(self.words, window_size)
            self._analyzed['heaps'] = True
            
        return self.results['heaps']
        
    def analyze_taylor(self, window_size=1000, overlap=500, regenerate=False):
        """
        Perform Taylor's law analysis.
        
        Args:
            window_size (int): Size of the windows for variance analysis
            overlap (int): Overlap between consecutive windows
            regenerate (bool): Whether to regenerate results if already calculated
            
        Returns:
            dict: Results of the analysis
        """
        if not self._analyzed['taylor'] or regenerate:
            self.results['taylor'] = calculate_taylor_exponent(
                self.words, self.word_counts, window_size, overlap)
            self._analyzed['taylor'] = True
            
        return self.results['taylor']
        
    def analyze_long_range_correlation(self, word=None, max_distance=100, regenerate=False):
        """
        Perform long-range correlation analysis.
        
        Args:
            word (str): Word to analyze (if None, use most frequent non-stopword)
            max_distance (int): Maximum lag distance to consider
            regenerate (bool): Whether to regenerate results if already calculated
            
        Returns:
            dict: Results of the analysis
        """
        if not self._analyzed['long_range'] or regenerate:
            if word is None:
                # Find the most frequent content word
                stop_words = set(['the', 'and', 'to', 'a', 'of', 'in', 'is', 'that', 'for', 'it', 
                                'with', 'as', 'was', 'on', 'be', 'at', 'by', 'this', 'have', 'or', 
                                'from', 'an', 'but', 'not', 'his', 'her', 'they', 'we', 'he', 'she'])
                word = max((w for w in self.word_counts.items() if w[0] not in stop_words), 
                          key=lambda x: x[1])[0]
                          
            self.results['long_range'] = calculate_long_range_correlation(
                self.words, word, max_distance)
            self._analyzed['long_range'] = True
            
        return self.results['long_range']
        
    def analyze_white_noise_fraction(self, word=None, regenerate=False):
        """
        Calculate the white noise fraction.
        
        Args:
            word (str): Word to analyze (if None, use most frequent non-stopword)
            regenerate (bool): Whether to regenerate results if already calculated
            
        Returns:
            dict: Results of the analysis
        """
        if not self._analyzed['white_noise'] or regenerate:
            if word is None:
                # Use the same word as in long-range correlation if possible
                if self._analyzed['long_range'] and 'word' in self.results['long_range']:
                    word = self.results['long_range']['word']
                else:
                    # Find the most frequent content word
                    stop_words = set(['the', 'and', 'to', 'a', 'of', 'in', 'is', 'that', 'for', 'it', 
                                    'with', 'as', 'was', 'on', 'be', 'at', 'by', 'this', 'have', 'or', 
                                    'from', 'an', 'but', 'not', 'his', 'her', 'they', 'we', 'he', 'she'])
                    word = max((w for w in self.word_counts.items() if w[0] not in stop_words), 
                              key=lambda x: x[1])[0]
                              
            self.results['white_noise'] = calculate_white_noise_fraction(self.words, word)
            self._analyzed['white_noise'] = True
            
        return self.results['white_noise']
        
    def analyze_entropy_rate(self, sample_sizes=None, regenerate=False):
        """
        Estimate the entropy rate.
        
        Args:
            sample_sizes (list): List of sample sizes for extrapolation
            regenerate (bool): Whether to regenerate results if already calculated
            
        Returns:
            dict: Results of the analysis
        """
        if not self._analyzed['entropy'] or regenerate:
            if sample_sizes is None:
                # Default sample sizes for extrapolation
                sample_sizes = [1000, 2000, 5000, 10000, 20000]
                # Limit to actual text size
                sample_sizes = [size for size in sample_sizes if size <= len(self.text)]
                
            self.results['entropy'] = calculate_entropy_rate(self.text, sample_sizes)
            self._analyzed['entropy'] = True
            
        return self.results['entropy']
        
    def analyze_strahler(self, max_sentences=200, regenerate=False):
        """
        Perform Strahler number analysis for sentences.
        
        Args:
            max_sentences (int): Maximum number of sentences to analyze
            regenerate (bool): Whether to regenerate results if already calculated
            
        Returns:
            dict: Results of the analysis
        """
        if not self._analyzed['strahler'] or regenerate:
            # Sample sentences (limit to prevent long processing time)
            sample_sentences = self.sentences[:max_sentences]
            
            self.results['strahler'] = calculate_strahler_number(sample_sentences)
            self._analyzed['strahler'] = True
            
        return self.results['strahler']
        
    def run_complete_analysis(self, metrics=None):
        """
        Run all analyses and return compiled results.
        
        Args:
            metrics (list): List of metrics to analyze. If None, analyze all.
            
        Returns:
            dict: Complete analysis results
        """
        if metrics is None:
            metrics = ['zipf', 'heaps', 'taylor', 'long_range', 
                      'white_noise', 'entropy', 'strahler']
            
        print(f"Running analysis for {self.title}...")
        
        if 'zipf' in metrics:
            print(f"  - Zipf analysis...")
            self.analyze_zipf()
            
        if 'heaps' in metrics:
            print(f"  - Heaps analysis...")
            self.analyze_heaps()
            
        if 'taylor' in metrics:
            print(f"  - Taylor analysis...")
            self.analyze_taylor()
            
        if 'long_range' in metrics:
            print(f"  - Long-range correlation analysis...")
            self.analyze_long_range_correlation()
            
        if 'white_noise' in metrics:
            print(f"  - White noise fraction analysis...")
            self.analyze_white_noise_fraction()
            
        if 'entropy' in metrics:
            print(f"  - Entropy rate estimation...")
            self.analyze_entropy_rate()
            
        if 'strahler' in metrics:
            print(f"  - Strahler number analysis...")
            self.analyze_strahler()
        
        return self.results