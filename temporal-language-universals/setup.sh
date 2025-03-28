#!/bin/bash
# Setup script for the Temporal Stability of Statistical Language Universals experiment

echo "=== Setting up Temporal Stability of Statistical Language Universals ==="

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create necessary directories
echo "Creating directory structure..."
mkdir -p data/early data/modern results/metrics results/visualizations

# Check if data files exist
if [ ! -f "data/early/donquixote.txt" ]; then
    echo "Warning: Early Don Quixote data file is missing."
    echo "Please add the text file to data/early/donquixote.txt"
fi

if [ ! -f "data/modern/donquixote.txt" ]; then
    echo "Warning: Modern Don Quixote data file is missing."
    echo "Please add the text file to data/modern/donquixote.txt"
fi

# Install required packages
echo "Installing required Python packages..."
pip install numpy pandas matplotlib scipy nltk

# Download NLTK data
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt')"

echo "Setup complete!"
echo ""
echo "To run the experiment:"
echo "  python run.py"
echo ""
echo "For more options:"
echo "  python run.py --help"