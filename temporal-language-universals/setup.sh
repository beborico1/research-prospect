#!/bin/bash
# Setup script for the Temporal Stability of Statistical Language Universals experiment

echo "=== Setting up Temporal Stability of Statistical Language Universals ==="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/early/es data/early/jp data/early/en
mkdir -p data/intermediate/es data/intermediate/jp data/intermediate/en
mkdir -p data/modern/es data/modern/jp data/modern/en
mkdir -p results/metrics results/visualizations

# Check if data files exist
if [ ! -f "data/early/es/don-quixote.txt" ]; then
    echo "Warning: Early Don Quixote data file is missing."
    echo "Please add the text file to data/early/es/don-quixote.txt"
fi

if [ ! -f "data/early/jp/genji.txt" ]; then
    echo "Warning: Early Genji data file is missing."
    echo "Please add the text file to data/early/jp/genji.txt"
fi

if [ ! -f "data/early/en/pride-and-prejudice.txt" ]; then
    echo "Warning: Early Pride and Prejudice data file is missing."
    echo "Please add the text file to data/early/en/pride-and-prejudice.txt"
fi

# Install required packages
echo "Installing required Python packages..."
pip install numpy pandas matplotlib scipy nltk

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

# Try to download punkt_tab (this might fail if it's not a standard resource)
echo "Attempting to download punkt_tab (custom resource)..."
python -c "
import nltk
try:
    nltk.download('punkt_tab')
    print('Successfully downloaded punkt_tab')
except Exception as e:
    print('Error downloading punkt_tab: This may be a custom resource')
    print('Creating empty punkt_tab directory structure as fallback...')
    import os
    nltk_data_path = nltk.data.path[0]
    punkt_tab_dir = os.path.join(nltk_data_path, 'tokenizers', 'punkt_tab')
    os.makedirs(punkt_tab_dir, exist_ok=True)
    for lang in ['english', 'spanish', 'japanese']:
        os.makedirs(os.path.join(punkt_tab_dir, lang), exist_ok=True)
        # Create an empty file to prevent lookup errors
        with open(os.path.join(punkt_tab_dir, lang, 'punkt.pickle'), 'w') as f:
            pass
"

echo "Setup complete!"
echo ""
echo "To run the experiment:"
echo "  python run.py"
echo ""
echo "For more options:"
echo "  python run.py --help"