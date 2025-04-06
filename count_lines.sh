#!/bin/bash

# Create a temporary file to store the results
temp_file=$(mktemp)

# Find all .py files, excluding those in venv directories
# Count the lines in each file and store in the format "line_count path"
find . -name "*.py" -not -path "*/venv/*" -type f | while read -r file; do
    line_count=$(wc -l < "$file")
    echo "$line_count $file" >> "$temp_file"
done

# Sort the results numerically in descending order
sort -nr "$temp_file" > line_count.txt

# Remove the temporary file
rm "$temp_file"

echo "Line count complete. Results saved to line_count.txt"