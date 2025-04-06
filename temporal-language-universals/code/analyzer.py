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
import os

# Try to import Japanese tokenizers
try:
    import janome
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    JANOME_AVAILABLE = True
except ImportError:
    JANOME_AVAILABLE = False

try:
    import MeCab
    MECAB_AVAILABLE = True
except ImportError:
    MECAB_AVAILABLE = False

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
    Enhanced with language-specific processing capabilities.
    """

    def __init__(self, text, title="", author="", year="", period="", language="en"):
        """
        Initialize the analyzer with text and metadata.
        
        Args:
            text (str): The text to analyze
            title (str): Title of the text
            author (str): Author of the text
            year (int): Publication year
            period (str): Historical period (Early, Intermediate, Modern)
            language (str): Language code ('en', 'es', 'jp', etc.)
        """
        self.raw_text = text
        self.title = title
        self.author = author
        self.year = year
        self.period = period
        self.language = language.lower() if language else "en"
        
        # Preprocess text
        self.text = self._preprocess_text(text)
        
        # Use language-specific tokenizer if available
        self.words, self.sentences = self._tokenize_text()
        
        self.word_counts = Counter(self.words)
        self.total_words = len(self.words)
        self.unique_words = len(self.word_counts)
        
        # Print tokenization info
        print(f"Tokenized {self.title} with {self.total_words} words, {self.unique_words} unique words, {len(self.sentences)} sentences")
        
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
                'language': self.language,
                'total_words': self.total_words,
                'unique_words': self.unique_words,
                'sentences': len(self.sentences)
            }
        }

    def _preprocess_text(self, text):
        """
        Clean and standardize text with language-specific handling.
        
        Args:
            text (str): Raw text
            
        Returns:
            str: Preprocessed text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # For Japanese, keep all characters intact
        if self.language == 'jp':
            # Just remove non-printable characters
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        else:
            # For other languages, apply standard cleaning
            text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-\(\)]', '', text)
        
        return text.strip()
        
    def _tokenize_text(self):
        """
        Tokenize text using language-specific tokenizers when available.
        
        Returns:
            tuple: (words, sentences)
        """
        # Language-specific tokenization
        if self.language == 'jp':
            return self._tokenize_japanese()
        elif self.language == 'es':
            return self._tokenize_spanish()
        else:
            # Default to NLTK for English and other languages
            try:
                words = word_tokenize(self.text.lower())
                sentences = sent_tokenize(self.text)
                print(f"Using NLTK tokenizers for {self.title}")
                return words, sentences
            except Exception as e:
                print(f"NLTK tokenization failed: {e}")
                # Fallback to simple tokenization
                words = self.text.lower().split()
                sentences = re.split(r'[.!?]+', self.text)
                return words, sentences
    
    def _tokenize_japanese(self):
        """
        Tokenize Japanese text using specialized tokenizers.
        
        Returns:
            tuple: (words, sentences)
        """
        # Try MeCab first (more accurate)
        if MECAB_AVAILABLE:
            try:
                tagger = MeCab.Tagger("-Owakati")  # Output with spaces between words
                words = tagger.parse(self.text).split()
                
                # For sentences, use period and other markers
                sentences = re.split(r'[。！？]', self.text)
                sentences = [s for s in sentences if s.strip()]
                
                print(f"Using MeCab tokenizer for {self.title}")
                return words, sentences
            except Exception as e:
                print(f"MeCab tokenization failed: {e}")
        
        # Try Janome next
        if JANOME_AVAILABLE:
            try:
                tokenizer = JanomeTokenizer()
                words = [token.surface for token in tokenizer.tokenize(self.text)]
                
                # For sentences, use period and other markers
                sentences = re.split(r'[。！？]', self.text)
                sentences = [s for s in sentences if s.strip()]
                
                print(f"Using Janome tokenizer for {self.title}")
                return words, sentences
            except Exception as e:
                print(f"Janome tokenization failed: {e}")
        
        # Last resort: character-based tokenization for Japanese
        print(f"No Japanese tokenizer available, using character-based tokenization for {self.title}")
        words = list(self.text)  # Each character as a token
        sentences = re.split(r'[。！？]', self.text)
        sentences = [s for s in sentences if s.strip()]
        return words, sentences
    
    def _tokenize_spanish(self):
        """
        Tokenize Spanish text with Spanish-specific considerations.
        
        Returns:
            tuple: (words, sentences)
        """
        try:
            # Make sure we have the Spanish tokenizer models
            try:
                nltk.data.find('tokenizers/punkt/spanish.pickle')
            except LookupError:
                print("Spanish tokenizer not found. Using default tokenizer.")
            
            words = word_tokenize(self.text.lower(), language='spanish')
            sentences = sent_tokenize(self.text, language='spanish')
            print(f"Using Spanish NLTK tokenizers for {self.title}")
            return words, sentences
        except Exception as e:
            print(f"Spanish NLTK tokenization failed: {e}")
            # Fallback to simple tokenization
            words = self.text.lower().split()
            sentences = re.split(r'[.!?]+', self.text)
            return words, sentences
    
    def analyze_zipf(self, regenerate=False):
        """Analyze Zipf's law with language-specific adjustments."""
        if not self._analyzed['zipf'] or regenerate:
            frequencies = sorted(self.word_counts.values(), reverse=True)
            ranks = np.arange(1, len(frequencies) + 1)
            
            # Adjust for Japanese if needed
            if self.language == 'jp' and len(frequencies) > 1000:
                # Use only top 1000 frequencies for more stable results
                frequencies = frequencies[:1000]
                ranks = ranks[:1000]
            
            self.results['zipf'] = calculate_zipf_exponent(ranks, frequencies)
            self._analyzed['zipf'] = True
            
        return self.results['zipf']
        
    def analyze_heaps(self, window_size=1000, regenerate=False):
        """Analyze Heaps' law with language-specific adjustments."""
        if not self._analyzed['heaps'] or regenerate:
            # Adjust window size for Japanese (character-based needs smaller windows)
            if self.language == 'jp' and not (MECAB_AVAILABLE or JANOME_AVAILABLE):
                window_size = min(200, window_size)  # Smaller window for character tokenization
            
            self.results['heaps'] = calculate_heaps_exponent(self.words, window_size)
            self._analyzed['heaps'] = True
            
        return self.results['heaps']
        
    def analyze_taylor(self, window_size=1000, overlap=500, regenerate=False):
        """Analyze Taylor's law with language-specific adjustments."""
        if not self._analyzed['taylor'] or regenerate:
            # Adjust window size for Japanese
            if self.language == 'jp':
                window_size = min(500, window_size)  # Smaller window for Japanese
                overlap = min(250, overlap)  # Smaller overlap too
            
            self.results['taylor'] = calculate_taylor_exponent(
                self.words, self.word_counts, window_size, overlap)
            self._analyzed['taylor'] = True
            
        return self.results['taylor']
        
    def analyze_long_range_correlation(self, word=None, max_distance=100, regenerate=False):
        """Analyze long-range correlation with language-specific adjustments."""
        if not self._analyzed['long_range'] or regenerate:
            if word is None:
                # Find the most frequent content word, with language-specific stop words
                stop_words = self._get_stop_words()
                
                # Find most frequent non-stop word
                word = max((w for w in self.word_counts.items() if w[0] not in stop_words), 
                          key=lambda x: x[1], default=('', 0))[0]
                
                # If no suitable word found, use the most frequent word
                if not word and self.word_counts:
                    word = max(self.word_counts.items(), key=lambda x: x[1])[0]
            
            # Adjust max distance for Japanese
            if self.language == 'jp':
                max_distance = min(50, max_distance)  # Smaller max distance for Japanese
                
            self.results['long_range'] = calculate_long_range_correlation(
                self.words, word, max_distance)
            self._analyzed['long_range'] = True
            
        return self.results['long_range']
        
    def analyze_white_noise_fraction(self, word=None, regenerate=False):
        """Analyze white noise fraction with language-specific adjustments."""
        if not self._analyzed['white_noise'] or regenerate:
            if word is None:
                # Use the same word as in long-range correlation if possible
                if self._analyzed['long_range'] and 'word' in self.results['long_range']:
                    word = self.results['long_range']['word']
                else:
                    # Find most frequent non-stop word
                    stop_words = self._get_stop_words()
                    word = max((w for w in self.word_counts.items() if w[0] not in stop_words), 
                              key=lambda x: x[1], default=('', 0))[0]
                    
                    # If no suitable word found, use the most frequent word
                    if not word and self.word_counts:
                        word = max(self.word_counts.items(), key=lambda x: x[1])[0]
            
            self.results['white_noise'] = calculate_white_noise_fraction(self.words, word)
            self._analyzed['white_noise'] = True
            
        return self.results['white_noise']
        
    def analyze_entropy_rate(self, sample_sizes=None, regenerate=False):
        """Analyze entropy rate with language-specific adjustments."""
        if not self._analyzed['entropy'] or regenerate:
            if sample_sizes is None:
                # Default sample sizes for extrapolation, adjusted by language
                if self.language == 'jp':
                    # Smaller samples for Japanese to avoid computation issues
                    sample_sizes = [500, 1000, 2000, 5000, 10000]
                else:
                    sample_sizes = [1000, 2000, 5000, 10000, 20000]
                
                # Limit to actual text size
                sample_sizes = [size for size in sample_sizes if size <= len(self.text)]
            
            self.results['entropy'] = calculate_entropy_rate(self.text, sample_sizes)
            self._analyzed['entropy'] = True
            
        return self.results['entropy']
        
    def analyze_strahler(self, max_sentences=200, regenerate=False):
        """Analyze Strahler number with language-specific adjustments."""
        if not self._analyzed['strahler'] or regenerate:
            # Sample sentences (limit to prevent long processing time)
            sample_sentences = self.sentences[:max_sentences]
            
            # If we have very few sentences for Japanese, generate more by splitting
            if self.language == 'jp' and len(sample_sentences) < 20:
                # Generate more sentence-like chunks by splitting on commas too
                chunks = []
                for sent in self.sentences:
                    chunks.extend(re.split(r'[、，,]', sent))
                sample_sentences = [s for s in chunks if len(s) > 5][:max_sentences]
            
            self.results['strahler'] = calculate_strahler_number(sample_sentences)
            self._analyzed['strahler'] = True
            
        return self.results['strahler']
    
    def _get_stop_words(self):
        """Get appropriate stop words list for the current language."""
        # English stop words
        en_stop_words = set(['the', 'and', 'to', 'a', 'of', 'in', 'is', 'that', 'for', 'it', 
                          'with', 'as', 'was', 'on', 'be', 'at', 'by', 'this', 'have', 'or', 
                          'from', 'an', 'but', 'not', 'his', 'her', 'they', 'we', 'he', 'she'])
        
        # Spanish stop words
        es_stop_words = set(['el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'y', 'o', 
                          'a', 'ante', 'bajo', 'con', 'de', 'desde', 'en', 'entre', 'hacia', 
                          'hasta', 'para', 'por', 'según', 'sin', 'sobre', 'tras', 'que', 'es', 
                          'su', 'sus', 'del', 'al', 'como', 'pero', 'si', 'no', 'más', 'ya', 
                          'le', 'lo', 'me', 'mi', 'tu', 'se', 'te', 'muy'])
        
        # Japanese stop words (common particles and auxiliaries)
        jp_stop_words = set(['の', 'に', 'は', 'を', 'た', 'が', 'で', 'て', 'と', 'し', 'れ', 
                          'さ', 'ある', 'いる', 'う', 'から', 'な', 'こと', 'として', 'い', 
                          'や', 'れる', 'など', 'なる', 'へ', 'か', 'だ', 'よう', 'も', 'ない'])
        
        # Return the appropriate stop words list
        if self.language == 'es':
            return es_stop_words
        elif self.language == 'jp':
            return jp_stop_words
        else:
            return en_stop_words
        
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
        
        # Run analyses in a try-except block to catch and handle errors
        for metric in metrics:
            try:
                if metric == 'zipf':
                    print(f"  - Zipf analysis...")
                    self.analyze_zipf()
                elif metric == 'heaps':
                    print(f"  - Heaps analysis...")
                    self.analyze_heaps()
                elif metric == 'taylor':
                    print(f"  - Taylor analysis...")
                    self.analyze_taylor()
                elif metric == 'long_range':
                    print(f"  - Long-range correlation analysis...")
                    self.analyze_long_range_correlation()
                elif metric == 'white_noise':
                    print(f"  - White noise fraction analysis...")
                    self.analyze_white_noise_fraction()
                elif metric == 'entropy':
                    print(f"  - Entropy rate estimation...")
                    self.analyze_entropy_rate()
                elif metric == 'strahler':
                    print(f"  - Strahler number analysis...")
                    self.analyze_strahler()
            except Exception as e:
                print(f"  - Error in {metric} analysis: {e}")
                # Create a placeholder result to indicate the error
                self.results[metric] = {'error': str(e)}
        
        return self.results