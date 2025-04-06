# Experiment

Lets design and conduct an experiment for the research prospect on the "Temporal Stability of Statistical Language Universals." Let us outline a practical approach to get started with this fascinating diachronic analysis.

## Experimental Design for Temporal Stability Analysis

Based on the research prospect, We'll design a scaled-down experiment that focuses on a few key statistical properties across different time periods. This will help validate the approach before expanding to the full research plan.

### Step 1: Corpus Selection and Preparation

Let's start by selecting texts from three distinct time periods in English literature:

1. **Early Period (16th-17th century)**: Shakespeare's works (e.g., "Hamlet," "Macbeth")
2. **Intermediate Period (19th century)**: Works by Charles Dickens (e.g., "Great Expectations," "Oliver Twist")
3. **Modern Period (20th-21st century)**: Contemporary fiction (e.g., works by Zadie Smith, Colson Whitehead)

We'll need to:

- Ensure comparable lengths (approximately 50,000 words from each author)
- Select similar genres (fictional prose/drama)
- Clean and preprocess the texts (remove metadata, standardize formatting)

### Step 2: Implementation of Core Analytical Methods

Let us create a Python implementation that will calculate the following key statistical properties:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re
import math
from scipy import stats
from scipy.optimize import curve_fit
import networkx as nx
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
nltk.download('punkt')

class LanguageUniversalsAnalyzer:
    """
    A class to analyze statistical language universals in texts from different time periods.
    Implements methods for calculating Zipf's law, Heaps' law, Taylor's law, and more.
    """

    def __init__(self, text, title="", author="", year="", period=""):
        """Initialize with text and metadata."""
        self.raw_text = text
        self.title = title
        self.author = author
        self.year = year
        self.period = period

        # Preprocess text
        self.text = self._preprocess_text(text)
        self.words = word_tokenize(self.text.lower())
        self.sentences = sent_tokenize(self.text)
        self.word_counts = Counter(self.words)
        self.total_words = len(self.words)
        self.unique_words = len(self.word_counts)

    def _preprocess_text(self, text):
        """Clean and standardize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        return text.strip()

    def zipf_analysis(self, plot=False):
        """
        Calculate Zipf's law exponent (η) where frequency ∝ rank^(-η).
        Returns the exponent and goodness of fit.
        """
        # Get word frequencies and sort them
        frequencies = sorted(self.word_counts.values(), reverse=True)
        ranks = np.arange(1, len(frequencies) + 1)

        # Take log of ranks and frequencies for linear regression
        log_ranks = np.log(ranks)
        log_frequencies = np.log(frequencies)

        # Linear regression to find the exponent
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_ranks, log_frequencies)

        # Zipf's exponent is the negative of the slope
        zipf_exponent = -slope
        r_squared = r_value**2

        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(log_ranks, log_frequencies, alpha=0.5, s=10)
            plt.plot(log_ranks, intercept + slope * log_ranks, 'r',
                    label=f'Zipf\'s law: η = {zipf_exponent:.3f}, R² = {r_squared:.3f}')
            plt.xlabel('Log Rank')
            plt.ylabel('Log Frequency')
            plt.title(f'Zipf\'s Law Analysis for {self.title} ({self.period})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        return {
            'zipf_exponent': zipf_exponent,
            'r_squared': r_squared,
            'std_error': std_err
        }

    def heaps_analysis(self, window_size=1000, plot=False):
        """
        Calculate Heaps' law exponent (ξ) where vocabulary_size ∝ text_length^(ξ).
        Returns the exponent and goodness of fit.
        """
        # Calculate vocabulary growth
        text_lengths = []
        vocab_sizes = []

        # Track unique words seen so far
        seen_words = set()

        # Sample at different text lengths
        for i in range(0, self.total_words, window_size):
            text_lengths.append(i + window_size)
            for word in self.words[i:i+window_size]:
                seen_words.add(word)
            vocab_sizes.append(len(seen_words))

        # Take log for linear regression
        log_lengths = np.log(text_lengths)
        log_vocab = np.log(vocab_sizes)

        # Linear regression to find the exponent
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lengths, log_vocab)

        # Heaps' exponent is the slope
        heaps_exponent = slope
        r_squared = r_value**2

        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(log_lengths, log_vocab, alpha=0.7)
            plt.plot(log_lengths, intercept + slope * log_lengths, 'r',
                    label=f'Heaps\' law: ξ = {heaps_exponent:.3f}, R² = {r_squared:.3f}')
            plt.xlabel('Log Text Length')
            plt.ylabel('Log Vocabulary Size')
            plt.title(f'Heaps\' Law Analysis for {self.title} ({self.period})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        return {
            'heaps_exponent': heaps_exponent,
            'r_squared': r_squared,
            'std_error': std_err
        }

    def taylor_analysis(self, window_size=1000, overlap=500, plot=False):
        """
        Calculate Taylor's law exponent (α) where variance ∝ mean^(α).
        Returns the exponent and goodness of fit.
        """
        # Create overlapping windows of text
        windows = [self.words[i:i+window_size] for i in range(0, self.total_words - window_size, overlap)]

        # Calculate mean and variance of word frequencies in each window
        word_stats = {}

        # Only analyze words that appear in most windows
        min_presence = 0.7 * len(windows)

        for word in self.word_counts:
            if self.word_counts[word] > min_presence:
                counts = [window.count(word) for window in windows]
                word_stats[word] = {
                    'mean': np.mean(counts),
                    'variance': np.var(counts)
                }

        # Extract means and variances for regression
        means = np.array([stats['mean'] for stats in word_stats.values() if stats['mean'] > 0])
        variances = np.array([stats['variance'] for stats in word_stats.values() if stats['mean'] > 0])

        # Apply log transformation
        log_means = np.log(means)
        log_variances = np.log(variances)

        # Linear regression to find Taylor exponent
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_means, log_variances)

        # Taylor's exponent is the slope
        taylor_exponent = slope
        r_squared = r_value**2

        if plot:
            plt.figure(figsize=(10, 6))
            plt.scatter(log_means, log_variances, alpha=0.7)
            plt.plot(log_means, intercept + slope * log_means, 'r',
                    label=f'Taylor\'s law: α = {taylor_exponent:.3f}, R² = {r_squared:.3f}')
            plt.xlabel('Log Mean Frequency')
            plt.ylabel('Log Variance')
            plt.title(f'Taylor\'s Law Analysis for {self.title} ({self.period})')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

        return {
            'taylor_exponent': taylor_exponent,
            'r_squared': r_squared,
            'std_error': std_err
        }

    def long_range_correlation(self, word=None, max_distance=100):
        """
        Calculate the autocorrelation function for return intervals.
        If word is None, use the most frequent content word.
        """
        if word is None:
            # Find the most frequent content word
            stop_words = set(['the', 'and', 'to', 'a', 'of', 'in', 'is', 'that', 'for', 'it', 'with', 'as', 'was', 'on'])
            word = max((w for w in self.word_counts.items() if w[0] not in stop_words), key=lambda x: x[1])[0]

        # Find positions of the word
        positions = [i for i, w in enumerate(self.words) if w == word]

        # Calculate return intervals
        intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

        # Calculate autocorrelation function
        acf = []
        mean_interval = np.mean(intervals)
        for lag in range(1, min(max_distance, len(intervals)//4)):
            # Calculate autocorrelation at this lag
            numerator = np.sum([(intervals[i] - mean_interval) * (intervals[i+lag] - mean_interval)
                               for i in range(len(intervals)-lag)])
            denominator = np.sum([(interval - mean_interval)**2 for interval in intervals])

            if denominator > 0:
                acf.append(numerator / denominator)
            else:
                acf.append(0)

        # Fit power law to autocorrelation function
        lags = np.arange(1, len(acf) + 1)
        valid_points = [(lag, ac) for lag, ac in zip(lags, acf) if ac > 0]

        if len(valid_points) > 5:  # Need enough points for a reasonable fit
            log_lags = np.log([p[0] for p in valid_points])
            log_acf = np.log([p[1] for p in valid_points])

            # Linear regression
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_lags, log_acf)
            correlation_exponent = -slope
            r_squared = r_value**2

            return {
                'word': word,
                'correlation_exponent': correlation_exponent,
                'r_squared': r_squared,
                'std_error': std_err
            }
        else:
            return {
                'word': word,
                'correlation_exponent': None,
                'r_squared': None,
                'std_error': None
            }

    def strahler_analysis(self, max_sentences=100):
        """
        Simplified Strahler number analysis for sentences.
        This is a basic implementation - a full implementation would require dependency parsing.
        """
        # For this simplified version, we'll estimate Strahler numbers based on
        # sentence length and basic nested structure indicators (parentheses, commas)

        # Sample sentences (limit to prevent long processing time)
        sample_sentences = self.sentences[:max_sentences]

        # Calculate a simplified Strahler-like metric
        strahler_estimates = []
        sentence_lengths = []

        for sentence in sample_sentences:
            words = word_tokenize(sentence)
            length = len(words)
            sentence_lengths.append(length)

            # Count nesting indicators
            commas = sentence.count(',')
            parentheses = sentence.count('(')
            semicolons = sentence.count(';')

            # Simplified Strahler estimate based on length and nesting
            # In a real implementation, this would use proper parsing
            nesting_factor = commas + parentheses*2 + semicolons*2
            strahler_estimate = math.log2(1 + length/5 + nesting_factor/3)
            strahler_estimates.append(strahler_estimate)

        # Fit logarithmic relationship: S ≈ a log₂(L) + b
        log_lengths = np.log2(np.array(sentence_lengths))

        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_lengths, strahler_estimates)
        r_squared = r_value**2

        average_strahler = np.mean(strahler_estimates)

        return {
            'average_strahler': average_strahler,
            'logarithmic_coefficient': slope,
            'intercept': intercept,
            'r_squared': r_squared
        }

    def run_complete_analysis(self, plot=False):
        """Run all analyses and return compiled results."""
        results = {
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

        # Run analyses
        results['zipf'] = self.zipf_analysis(plot=plot)
        results['heaps'] = self.heaps_analysis(plot=plot)
        results['taylor'] = self.taylor_analysis(plot=plot)
        results['long_range'] = self.long_range_correlation()
        results['strahler'] = self.strahler_analysis()

        return results


# Function to compare texts from different time periods
def compare_statistical_universals(texts_data, output_file=None):
    """
    Analyze and compare statistical universals across different texts/time periods.

    Args:
        texts_data: List of dictionaries with text content and metadata
        output_file: Optional file to save results

    Returns:
        DataFrame with comparative results
    """
    results = []

    for text_data in texts_data:
        analyzer = LanguageUniversalsAnalyzer(
            text_data['text'],
            title=text_data.get('title', ''),
            author=text_data.get('author', ''),
            year=text_data.get('year', ''),
            period=text_data.get('period', '')
        )

        analysis_results = analyzer.run_complete_analysis()

        # Extract key metrics
        result = {
            'Title': text_data.get('title', ''),
            'Author': text_data.get('author', ''),
            'Year': text_data.get('year', ''),
            'Period': text_data.get('period', ''),
            'Total Words': analysis_results['metadata']['total_words'],
            'Unique Words': analysis_results['metadata']['unique_words'],
            'Zipf Exponent': analysis_results['zipf']['zipf_exponent'],
            'Zipf R²': analysis_results['zipf']['r_squared'],
            'Heaps Exponent': analysis_results['heaps']['heaps_exponent'],
            'Heaps R²': analysis_results['heaps']['r_squared'],
            'Taylor Exponent': analysis_results['taylor']['taylor_exponent'],
            'Taylor R²': analysis_results['taylor']['r_squared'],
            'Correlation Exponent': analysis_results['long_range'].get('correlation_exponent'),
            'Average Strahler': analysis_results['strahler']['average_strahler'],
            'Strahler Log Coefficient': analysis_results['strahler']['logarithmic_coefficient']
        }

        results.append(result)

    # Create DataFrame for comparative analysis
    df = pd.DataFrame(results)

    # Save to file if requested
    if output_file:
        df.to_csv(output_file, index=False)

    return df


# Function to visualize comparative results
def visualize_comparison(comparison_df, metrics=None):
    """
    Create visualizations comparing key metrics across time periods.

    Args:
        comparison_df: DataFrame with comparative results
        metrics: List of metrics to visualize (default: visualize all key metrics)
    """
    if metrics is None:
        metrics = [
            'Zipf Exponent',
            'Heaps Exponent',
            'Taylor Exponent',
            'Correlation Exponent',
            'Average Strahler',
            'Strahler Log Coefficient'
        ]

    # Group by period for averaging
    period_groups = comparison_df.groupby('Period')
    period_means = period_groups.mean()
    periods = period_means.index

    # Create visualizations
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4*len(metrics)))

    for i, metric in enumerate(metrics):
        if metric in period_means.columns:
            ax = axes[i] if len(metrics) > 1 else axes

            # Plot individual data points
            for period in periods:
                period_data = comparison_df[comparison_df['Period'] == period]
                ax.scatter(period_data['Period'], period_data[metric], alpha=0.7, label=None)

            # Plot period means
            ax.plot(periods, period_means[metric], 'ro-', linewidth=2, markersize=8)

            # Add error bars if we have multiple texts per period
            if len(period_groups) > 0:
                period_stds = period_groups.std()
                if not period_stds[metric].isna().all():
                    ax.errorbar(periods, period_means[metric], yerr=period_stds[metric],
                               fmt='none', ecolor='r', capsize=5, alpha=0.5)

            ax.set_title(f'Temporal Variation in {metric}')
            ax.set_xlabel('Time Period')
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


# Function to load and preprocess text files
def load_text_from_file(filepath, title=None, author=None, year=None, period=None):
    """Load text from file with metadata."""
    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    return {
        'text': text,
        'title': title or filepath.split('/')[-1],
        'author': author,
        'year': year,
        'period': period
    }


# Example of how to use the full analysis pipeline
def main():
    """Example of full analysis workflow."""

    # Sample data (replace with actual file paths in real implementation)
    texts_data = [
        # Early period (16th-17th century)
        load_text_from_file('hamlet.txt', 'Hamlet', 'Shakespeare', 1600, 'Early'),
        load_text_from_file('macbeth.txt', 'Macbeth', 'Shakespeare', 1606, 'Early'),

        # Intermediate period (19th century)
        load_text_from_file('great_expectations.txt', 'Great Expectations', 'Dickens', 1861, 'Intermediate'),
        load_text_from_file('oliver_twist.txt', 'Oliver Twist', 'Dickens', 1838, 'Intermediate'),

        # Modern period (20th-21st century)
        load_text_from_file('white_teeth.txt', 'White Teeth', 'Smith', 2000, 'Modern'),
        load_text_from_file('the_underground_railroad.txt', 'The Underground Railroad', 'Whitehead', 2016, 'Modern')
    ]

    # Run comparative analysis
    comparison_results = compare_statistical_universals(texts_data, 'language_universals_results.csv')

    # Visualize results
    visualize_comparison(comparison_results)
    plt.savefig('language_universals_comparison.png')

    print("Analysis complete. Results saved to 'language_universals_results.csv' and 'language_universals_comparison.png'")

    return comparison_results


if __name__ == "__main__":
    main()
```

Now, let us create a practical experiment plan that you can follow to implement this research. This will include specific steps for acquiring texts, adjusting the code for your specific needs, and analyzing the results.

## Practical Implementation Plan: Temporal Stability of Statistical Language Universals

This document outlines a step-by-step approach to implement the experiment described in the research prospect, focusing on measuring the temporal stability of statistical language universals across different historical periods.

## 1. Data Collection

### 1.1 Text Selection Guidelines

Select 2-3 texts from each of the following periods:

**Early Period (15th-17th centuries)**:

- Shakespeare's works (e.g., "Hamlet," "Macbeth," "King Lear")
- Works by Christopher Marlowe or Francis Bacon
- Source: [Project Gutenberg](https://www.gutenberg.org/) or [Folger Digital Texts](https://shakespeare.folger.edu/)

**Intermediate Period (18th-19th centuries)**:

- Works by Charles Dickens (e.g., "Great Expectations," "Oliver Twist")
- Works by Jane Austen (e.g., "Pride and Prejudice")
- Source: [Project Gutenberg](https://www.gutenberg.org/)

**Modern Period (20th-21st centuries)**:

- Contemporary fiction by authors like Zadie Smith, Colson Whitehead
- Source: E-books or available digital samples

### 1.2 Text Preparation Guidelines

For each text:

1. Ensure digital format (plain text preferred)
2. Clean metadata and editorial notes
3. Standardize formatting:
   - Remove chapter headings and page numbers
   - Standardize punctuation
   - Ensure UTF-8 encoding
4. Aim for comparable lengths (30,000-50,000 words per text)
5. Create a catalog with metadata:
   - Title
   - Author
   - Year of publication
   - Period classification
   - Word count

## 2. Setting Up the Analysis Environment

### 2.1 Software Requirements

- Python 3.7+ environment
- Required packages:

  ```libraries
  numpy
  pandas
  matplotlib
  scipy
  nltk
  networkx
  ```

### 2.2 Directory Structure

Create the following directory structure:

```structure
temporal-language-universals/
├── data/
│   ├── early/
│   ├── intermediate/
│   └── modern/
├── code/
│   └── language_universals_analyzer.py
├── results/
│   ├── metrics/
│   └── visualizations/
└── README.md
```

## 3. Implementation Steps

### 3.1 Text Processing

1. Place cleaned text files in their respective period directories
2. Create a metadata CSV file with information about each text
3. Run text preprocessing to standardize all texts

### 3.2 Analysis Pipeline

1. **Zipf's Law Analysis**: Calculate the exponent η for each text
2. **Heaps' Law Analysis**: Calculate the exponent ξ for vocabulary growth
3. **Taylor's Law Analysis**: Calculate the exponent α relating mean frequency to variance
4. **Long-Range Correlation Analysis**: Calculate correlation exponent γ
5. **Strahler Number Analysis**: Calculate simplified Strahler numbers and fit to logarithmic model

### 3.3 Comparative Analysis

1. Aggregate results by time period
2. Calculate means and standard deviations for each metric
3. Perform statistical tests to evaluate significance of differences between periods
4. Create visualizations showing temporal changes

## 4. Experimental Protocol

### 4.1 Basic Analysis

For each text:

1. Load and preprocess the text
2. Calculate all statistical measures using the provided code
3. Save individual results to CSV files

```python
# Example code to analyze a single text
from language_universals_analyzer import LanguageUniversalsAnalyzer

# Load text
with open('data/early/hamlet.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize analyzer
analyzer = LanguageUniversalsAnalyzer(
    text, 
    title="Hamlet",
    author="Shakespeare",
    year=1600,
    period="Early"
)

# Run analysis
results = analyzer.run_complete_analysis(plot=True)

# Save plots
plt.savefig('results/visualizations/hamlet_zipf.png')
```

### 4.2 Comparative Analysis

After analyzing all texts:

1. Combine individual results into a master dataset
2. Group by time period
3. Generate comparative visualizations

```python
# Example code to compare results
from language_universals_analyzer import compare_statistical_universals, visualize_comparison

# List of text data
texts_data = [
    # Include all texts with metadata
]

# Run comparative analysis
comparison_results = compare_statistical_universals(texts_data, 'results/metrics/comparison.csv')

# Visualize results
visualize_comparison(comparison_results)
plt.savefig('results/visualizations/temporal_comparison.png')
```

## 5. Advanced Analysis Options

If initial results show promise, consider these extensions:

### 5.1 Enhanced Strahler Number Analysis

For more accurate Strahler number calculation:

1. Install spaCy for dependency parsing
2. Implement proper binary tree conversion from dependency trees
3. Calculate true Strahler numbers based on syntactic structure

### 5.2 White Noise Fraction Estimation

Implement the methodology from Tanaka-Ishii & Bunde (2016):

1. Calculate exceedance probability for return intervals
2. Fit to stretched exponential function
3. Estimate white noise fraction

### 5.3 Entropy Rate Estimation

Implement compression-based methods:

1. Use different compression algorithms (e.g., gzip, bzip2)
2. Calculate entropy rates for texts of different lengths
3. Extrapolate using stretched exponential function

## 6. Result Interpretation Guide

When analyzing your results, consider:

### 6.1 Statistical Significance

- Calculate p-values for differences between time periods
- Use confidence intervals to account for within-period variation
- Consider both parametric (t-tests) and non-parametric (Mann-Whitney) tests

### 6.2 Effect Size

- Beyond statistical significance, measure the magnitude of differences
- Calculate Cohen's d or similar effect size metrics
- Consider what constitutes a meaningful difference for each property

### 6.3 Patterns of Stability/Change

Look for these specific patterns:

- Which properties remain most stable across time?
- Which show systematic changes?
- Do different languages show similar patterns of change?
- Are changes correlated with known historical linguistic shifts?

### 6.4 Cognitive Constraints

For Strahler number analysis:

- Does the 3-4 limit remain constant across periods?
- How does the logarithmic coefficient change over time?
- Are simplification trends visible in modern language?

## 7. Documentation and Sharing

For reproducibility:

1. Document all preprocessing steps in detail
2. Save all raw metrics, not just summaries
3. Create notebooks demonstrating the analysis pipeline
4. Share code, metrics, and selected visualizations

## Next Steps

After completing the initial experiment:

1. Refine methods based on initial findings
2. Expand to more texts and languages
3. Implement more sophisticated analysis techniques
4. Consider correlating changes with specific linguistic features
5. Connect findings to theoretical frameworks about language evolution

## Troubleshooting Common Issues

- **Text length effects**: Ensure texts are comparable in length or use windowing techniques
- **Genre differences**: Control for genre or analyze genre effects separately
- **Translation effects**: Be aware of potential artifacts from translated texts
- **Preprocessing decisions**: Document all cleaning steps and test their impact on results
- **Statistical power**: Use bootstrapping for more reliable confidence intervals with small sample sizes

Now, let us provide more detailed guidance on actually setting up and running the first phase of this experiment:

## Getting Started with the Temporal Stability Experiment

### 1. Initial Setup (Week 1)

First, let's set up the environment and get the data you'll need:

1. **Create your Python environment**:
   - Install Python 3.7+ if you don't have it
   - Install required packages: `pip install numpy pandas matplotlib scipy nltk networkx`

2. **Set up project structure**:
   - Create the directories as outlined in the experiment plan
   - Download the code from the first artifact to `code/language_universals_analyzer.py`

3. **Data acquisition**:
   - Start with a small pilot of just 3 texts (one from each period)
   - Project Gutenberg is an excellent source: <https://www.gutenberg.org/>
   - For early period: "Hamlet" by Shakespeare is a good starting point
   - For intermediate period: "Great Expectations" by Dickens
   - For modern period: Look for a contemporary novel with a free sample or use a public domain contemporary work

### 2. Initial Analysis (Week 2)

1. **Text preprocessing**:

   ```python
   # Sample code for basic preprocessing
   def preprocess_text(text):
       # Remove metadata and headers (typically at beginning/end of Gutenberg texts)
       lines = text.split('\n')
       start_idx = 0
       end_idx = len(lines)

       # Find start of actual text (after Gutenberg header)
       for i, line in enumerate(lines):
           if "*** START OF" in line:
               start_idx = i + 1
               break

       # Find end of actual text (before Gutenberg footer)
       for i, line in enumerate(lines):
           if "*** END OF" in line:
               end_idx = i
               break

       # Extract main text
       main_text = '\n'.join(lines[start_idx:end_idx])

       # Basic cleaning
       import re
       # Standardize whitespace

       main_text = re.sub(r'\s+', ' ', main_text)

       # Remove non-standard characters but keep basic punctuation
       main_text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-]', '',   main_text)

       # Standardize quotes
       main_text = main_text.replace('"', '"').replace('"', '"')
       main_text = main_text.replace(''', "'").replace(''', "'")

       return main_text.strip()
    ```

2. **Run initial analysis**:

   ```python
   from language_universals_analyzer import LanguageUniversalsAnalyzer
   import matplotlib.pyplot as plt

   # Process each text
   texts = [
       {"file": "data/early/hamlet.txt", "title": "Hamlet", "author": "Shakespeare", "year": 1600, "period": "Early"},
       {"file": "data/intermediate/great_expectations.txt", "title": "Great Expectations", "author": "Dickens", "year": 1861, "period": "Intermediate"},
       {"file": "data/modern/contemporary_novel.txt", "title": "Contemporary Novel", "author": "Modern Author", "year": 2010, "period": "Modern"}
   ]

   results = []

   for text_info in texts:
       # Load and preprocess
       with open(text_info["file"], 'r', encoding='utf-8') as f:
           raw_text = f.read()

       clean_text = preprocess_text(raw_text)

       # Initialize analyzer
       analyzer = LanguageUniversalsAnalyzer(
           clean_text,
           title=text_info["title"],
           author=text_info["author"],
           year=text_info["year"],
           period=text_info["period"]
       )

       # Run analysis with visualizations
       analysis_results = analyzer.run_complete_analysis(plot=True)

       # Save visualizations
       plt.savefig(f'results/visualizations/{text_info["title"].lower().replace(" ", "_")}_stats.png')

       # Store results
       results.append(analysis_results)

       # Print key metrics
       print(f"\nResults for {text_info['title']} ({text_info['period']} period):")
       print(f"Zipf exponent: {analysis_results['zipf']['zipf_exponent']:.3f}")
       print(f"Heaps exponent: {analysis_results['heaps']['heaps_exponent']:.3f}")
       print(f"Taylor exponent: {analysis_results['taylor']['taylor_exponent']:.3f}")
       print(f"Correlation exponent: {analysis_results['long_range'].get('correlation_exponent', 'N/A')}")
       print(f"Average Strahler: {analysis_results['strahler']['average_strahler']:.3f}")
   ```

### 3. Expanding the Corpus (Week 3-4)

After getting your initial results, expand to include more texts:

1. **Add more texts from each period**:
   - Early: Add "Macbeth" and/or "King Lear"
   - Intermediate: Add "Oliver Twist" and/or "Pride and Prejudice"
   - Modern: Add 2-3 more contemporary texts

2. **Run comparative analysis**:

   ```python
   from language_universals_analyzer import compare_statistical_universals, visualize_comparison
   import pandas as pd

   # Create a list of text data dictionaries
   texts_data = []
   for text_info in expanded_texts:
       with open(text_info["file"], 'r', encoding='utf-8') as f:
           raw_text = f.read()

       texts_data.append({
           'text': preprocess_text(raw_text),
           'title': text_info["title"],
           'author': text_info["author"],
           'year': text_info["year"],
           'period': text_info["period"]
       })

   # Run comparative analysis
   comparison_df = compare_statistical_universals(texts_data, 'results/metrics/comparison.csv')

   # Create visualizations
   fig = visualize_comparison(comparison_df)
   fig.savefig('results/visualizations/temporal_comparison.png')
   ```

### 4. Analyzing Results (Week 5)

1. **Statistical testing**:

   ```python
   # Group by period
   early_data = comparison_df[comparison_df['Period'] == 'Early']
   intermediate_data = comparison_df[comparison_df['Period'] == 'Intermediate']
   modern_data = comparison_df[comparison_df['Period'] == 'Modern']

   # Run t-tests for key metrics
   from scipy import stats

   metrics = ['Zipf Exponent', 'Heaps Exponent', 'Taylor Exponent', 'Average Strahler']

   print("Statistical comparison between periods:")
   for metric in metrics:
       # Early vs. Intermediate
       t_stat, p_value = stats.ttest_ind(early_data[metric], intermediate_data[metric])
       print(f"{metric} - Early vs. Intermediate: t={t_stat:.3f}, p={p_value:.3f}")

       # Intermediate vs. Modern
       t_stat, p_value = stats.ttest_ind(intermediate_data[metric], modern_data[metric])
       print(f"{metric} - Intermediate vs. Modern: t={t_stat:.3f}, p={p_value:.3f}")

       # Early vs. Modern
       t_stat, p_value = stats.ttest_ind(early_data[metric], modern_data[metric])
       print(f"{metric} - Early vs. Modern: t={t_stat:.3f}, p={p_value:.3f}")
       print()
   ```

2. **Create summary tables**:

   ```python
   # Mean and standard deviation by period
   summary = comparison_df.groupby('Period').agg({
       'Zipf Exponent': ['mean', 'std'],
       'Heaps Exponent': ['mean', 'std'],
       'Taylor Exponent': ['mean', 'std'],
       'Correlation Exponent': ['mean', 'std'],
       'Average Strahler': ['mean', 'std'],
       'Strahler Log Coefficient': ['mean', 'std']
   })

   # Save summary table
   summary.to_csv('results/metrics/period_summary.csv')
   print(summary)
   ```

### 5. Enhancing the Analysis (Week 6+)

If your initial results show interesting patterns, consider these enhancements:

1. **Implement improved Strahler number analysis**:

   ```python
   # Install necessary packages
   # pip install spacy
   # python -m spacy download en_core_web_sm

   import spacy

   # Load spaCy model
   nlp = spacy.load("en_core_web_sm")

   def calculate_strahler_number(sentence):
       """Calculate Strahler number based on dependency parse."""
       # Parse the sentence
       doc = nlp(sentence)

       # Create a dictionary to store Strahler numbers for each token
       strahler_values = {}

       # Process each token bottom-up
       for token in doc:
           # Start with leaf nodes (tokens with no children)
           if len(list(token.children)) == 0:
               strahler_values[token.i] = 1

       # Iteratively calculate Strahler numbers until we reach the root
       # This is a simplified approach - a full implementation would convert
       # the dependency tree to a binary tree first
       changes = True
       while changes:
           changes = False
           for token in doc:
               # Skip tokens we've already processed
               if token.i in strahler_values:
                   continue

               # Get children's Strahler numbers if they're available
               children_strahlers = [strahler_values.get(child.i, 0) for child in token.children]

               # If we have values for all children, calculate this token's Strahler number
               if all(strahler > 0 for strahler in children_strahlers) and children_strahlers:
                   # Sort children's Strahler numbers in descending order
                   sorted_strahlers = sorted(children_strahlers, reverse=True)

                   # Calculate Strahler number based on children
                   if len(sorted_strahlers) >= 2 and sorted_strahlers[0] == sorted_strahlers[1]:
                       strahler_values[token.i] = sorted_strahlers[0] + 1
                   else:
                       strahler_values[token.i] = sorted_strahlers[0]

                   changes = True

       # Return the Strahler number of the sentence (root node)
       root = [token for token in doc if token.head == token][0]
       return strahler_values.get(root.i, 0)
   ```

2. **Implement white noise fraction estimation**:

   ```python
   def estimate_white_noise_fraction(text, word=None):
       """Estimate the white noise fraction using methodology from Tanaka-Ishii & Bunde."""
       analyzer = LanguageUniversalsAnalyzer(text)

       # Find most frequent content word if not specified
       if word is None:
           stop_words = set(['the', 'and', 'to', 'a', 'of', 'in', 'is', 'that', 'for', 'it'])
           word = max((w for w in analyzer.word_counts.items() if w[0] not in stop_words),
                      key=lambda x: x[1])[0]

       # Find positions of the word
       positions = [i for i, w in enumerate(analyzer.words) if w == word]

       # Calculate return intervals
       intervals = [positions[i+1] - positions[i] for i in range(len(positions)-1)]

       # Calculate autocorrelation at lag 1
       mean_interval = np.mean(intervals)
       numerator = np.sum([(intervals[i] - mean_interval) * (intervals[i+1] - mean_interval)
                         for i in range(len(intervals)-1)])
       denominator = np.sum([(interval - mean_interval)**2 for interval in intervals])

       C1 = numerator / denominator if denominator > 0 else 0

       # Calculate white noise fraction
       white_noise = 1 / (1 + np.sqrt(C1 / (1 - C1))) if C1 > 0 and C1 < 1 else 0.5

       return {
           'word': word,
           'white_noise_fraction': white_noise,
           'C1': C1
       }
   ```

### 6. Interpreting Your Results

When analyzing the data, look for these specific patterns:

1. **Stability vs. Change**:
   - Which properties show the least variation across time periods?
   - Do any properties show a clear directional trend from Early → Intermediate → Modern?

2. **Cognitive Constraints**:
   - Does the Strahler number analysis confirm the 3-4 limit across all periods?
   - How does sentence complexity (relative to length) change over time?

3. **Relationships Between Properties**:

   ```python
   # Correlation analysis between different metrics
   from scipy.stats import pearsonr

   # Calculate correlations
   corr_matrix = comparison_df[metrics].corr()
   print("Correlation between metrics:")
   print(corr_matrix)

   # Visualize relationships
   import seaborn as sns

   plt.figure(figsize=(10, 8))
   sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
   plt.title('Correlation Between Statistical Properties')
   plt.savefig('results/visualizations/property_correlations.png')
   ```

4. **Relationship to known linguistic shifts**:
   - Compare your findings with documented historical changes in language
   - Look for connections between statistical properties and specific linguistic features

### Documentation and Sharing

Create a comprehensive report of your findings:

1. **Structure your report**:
   - Introduction: Research questions and background
   - Methods: Corpus details and analytical approach
   - Results: Findings for each statistical property
   - Discussion: Interpretation and implications
   - Conclusion: Key takeaways and future directions

2. **Visualize key findings**:

   ```python
   # Create summary visualization
   plt.figure(figsize=(12, 8))

   # Normalize metrics for comparison
   for metric in metrics:
       normalized_values = []
       for period in ['Early', 'Intermediate', 'Modern']:
           period_data = comparison_df[comparison_df['Period'] == period][metric]
           mean_value = period_data.mean()
           normalized_values.append(mean_value)

       # Normalize to Early period
       normalized_values = [v/normalized_values[0] for v in normalized_values]

       # Plot
       plt.plot(['Early', 'Intermediate', 'Modern'], normalized_values, 'o-', linewidth=2, label=metric)

   plt.title('Relative Change in Statistical Properties Over Time')
   plt.ylabel('Value (Relative to Early Period)')
   plt.grid(True, alpha=0.3)
   plt.legend()
   plt.savefig('results/visualizations/temporal_trends_summary.png')
   ```

## Next Steps After Initial Experiment

Once you've completed this initial experiment, you can:

1. **Expand to other languages**:
   - Repeat the analysis with Spanish texts (e.g., Cervantes → modern authors)
   - Compare temporal trends across languages

2. **Implement more sophisticated analyses**:
   - Entropy rate estimation using compression methods
   - Full Strahler number analysis with proper dependency parsing
   - Rigorous statistical testing with bootstrapping

3. **Connect to practical applications**:
   - Use findings to improve historical language modeling
   - Develop methods for dating texts of unknown origin
   - Create tools for authorship analysis

This practical guide should give you a structured approach to begin implementing the research prospect. Start with the small pilot study, then gradually expand as you refine your methods and understanding.