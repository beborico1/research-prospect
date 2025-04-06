# Temporal Stability of Statistical Language Universals - Experiment

This repository contains code to analyze the statistical properties of language across different time periods, using early and modern versions of Don Quixote as a case study.

## Project Structure

```structure
temporal-language-universals/
├── README.md
├── code/
│   ├── run.py
│   └── run_experiment.py
├── data/
│   ├── early/
│   │   └── donquixote.txt
│   ├── intermediate/
│   └── modern/
│       └── donquixote.txt
├── results/
│   ├── metrics/
│   └── visualizations/
├── experiment.md
└── formulas.md
```

## Prerequisites

The experiment requires the following Python packages:

- numpy
- pandas
- matplotlib
- scipy
- nltk

You can install these with pip:

```bash
pip install numpy pandas matplotlib scipy nltk
```

## Running the Experiment

1. Make sure your project structure is set up correctly with the data files in place.
2. Navigate to the `temporal-language-universals` directory.
3. Create a `code` directory if it doesn't exist:

   ```bash
   mkdir -p code
   ```

4. Copy the experiment script files to the code directory:

   ```bash
   cp run.py code/
   cp run_experiment.py code/
   ```

5. Run the experiment:

   ```bash
   cd code
   python run.py
   ```

## What the Experiment Does

The experiment analyzes the following statistical properties of language:

1. **Zipf's Law**: Measures the exponent η in the power-law distribution of word frequencies.
2. **Heaps' Law**: Calculates the exponent ξ relating vocabulary size to text length.
3. **Taylor's Law**: Finds the exponent α relating mean word frequency to variance.
4. **Long-Range Correlation**: Analyzes the clustering and correlation of word occurrences.
5. **Entropy Rate**: Estimates the complexity of language through information theory.
6. **White Noise Fraction**: Measures the proportion of random to structured patterns in text.
7. **Strahler Number Analysis**: Quantifies sentence complexity using a proxy for syntactic tree complexity.

## Interpreting the Results

The experiment outputs:

1. **CSV files** in `results/metrics/` showing numerical results for each statistical property.
2. **Visualizations** in `results/visualizations/` showing comparisons between the early and modern texts.
3. **Console output** with key metrics and comparisons.

The most important metrics to look at:

- **Zipf Exponent**: Values closer to 1.0 indicate classic Zipfian distribution.
- **Taylor Exponent**: Values around 0.58 are typical for natural language; deviations may indicate stylistic differences.
- **Correlation Exponent**: Indicates the strength of long-range correlations in word usage.
- **White Noise Fraction**: Measures how much of the text behaves like random noise (typically 55-69% for natural language).
- **Entropy Rate**: Lower values indicate more predictable/constrained language.
- **Average Strahler**: Values between 3-4 are typical for natural language, reflecting cognitive constraints.
- **Strahler Log Coefficient**: Measures how sentence complexity grows with length.

## Comparative Analysis

The experiment provides:

1. **Direct Comparison**: Bar charts showing each property side by side for early vs. modern text.
2. **Percent Change Analysis**: Shows how much each property has changed from the early to modern period.

## Extending the Experiment

You can extend this experiment in several ways:

1. **Add More Texts**: Include intermediate period texts or works from other authors in the same periods.
2. **Add More Languages**: Apply the same analysis to Spanish, Japanese, or other languages.
3. **Improve Strahler Analysis**: Implement proper dependency parsing for more accurate sentence complexity analysis.
4. **Advanced Statistical Tests**: Add significance testing to determine if differences are statistically significant.

## References

This experiment is based on research by Professor Kumiko Tanaka-Ishii and colleagues, including:

- Tanaka-Ishii, K. (2021). Statistical Universals of Language.
- Tanaka-Ishii, K., & Bunde, A. (2016). Long-range memory in literary texts.
- Tanaka-Ishii, K., & Kobayashi, T. (2018). Taylor's law for linguistic sequences.
- Tanaka-Ishii, K., & Ishii, Y. (2023). Strahler number of natural language sentences.

## Troubleshooting

If you encounter issues:

1. **File Encoding Problems**: Try manually specifying encoding when opening files (utf-8 or latin-1).
2. **Missing NLTK Data**: Ensure you've downloaded NLTK punkt tokenizer (`nltk.download('punkt')`).
3. **Visualization Errors**: Make sure all directories exist and are writable.
4. **Empty Files**: Check if the data files contain sufficient text for analysis.