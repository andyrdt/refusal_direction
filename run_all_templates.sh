#!/bin/bash

# Template files to process
TEMPLATES=("template_1k" "template_3k" "template_11k" "template_21k" "template_31k" "template_47k")

echo "Running harmless components calculation for all template lengths"

# Harmless components calculation
echo "=== Harmless Components Calculation ===" 
for template in "${TEMPLATES[@]}"; do
    echo "Running $template..."
    python calculate_stealth_harmful_components.py --template_file $template
done

echo "All harmless components calculation completed!"