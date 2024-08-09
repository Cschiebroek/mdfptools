# Molecular Descriptor Benchmarking Platform

## Overview
This project is a molecular descriptor benchmarking platform for evaluating the performance of different machine learning models on various molecular descriptors. The platform includes descriptors like RDKit, MDFP, MACCS, ECFP4, and PaDEL, and models like XGBoost, PLS, Lasso, RandomForest, and kNN.

## Installation
To get started with this project, you'll need to set up the environment and install all necessary dependencies.

### Using Conda:
```bash
conda env create -f environment.yml
conda activate molecular-descriptors
```
### Using Pip:
```bash
pip install -r requirements.txt
```
### Usage
1. Prepare Data:
The data is fetched from a PostgreSQL database. You can modify the database connection settings in config.yml.
2. Run the Benchmark:
To run the benchmark with specific descriptors and models:
```bash
python main.py
```
### Configeration
The project is configured using the config.yml file. This file contains settings for descriptors, models, data paths, and more.

### Dependencies
- Python 3.9
- RDKit
- Scikit-learn
- XGBoost
- Pandas
- PostgreSQL (for database connection)

### Directory Structure
```bash
.
├── data/                   # Raw data
├── descriptors/            # Descriptor calculation scripts
├── models/                 # Machine learning models
├── results/                # Results and plots
├── utils/                  # Utility scripts (data preprocessing, visualization)
├── config.yml              # Configuration file
├── environment.yml         # Conda environment configuration
├── requirements.txt        # Pip requirements file
├── README.md               # Project documentation
└── main.py                 # Main script to run the benchmarking
```
### Contributors
Carl Schiebroek
