#!/bin/bash

# Create the conda environment
conda env create -f environment.yml

# Activate the environment
conda activate molecular-descriptors

# Install additional dependencies if needed
pip install -r requirements.txt

echo "Setup complete. You can now run the benchmarking with 'python main.py'."
