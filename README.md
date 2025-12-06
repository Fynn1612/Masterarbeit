# Master's Thesis: Uncertainty Quantification Methods for Regression

This repository contains the implementation and evaluation of multiple uncertainty quantification (UQ) methods for regression tasks, developed as part of a Master's thesis.

## Overview

The project compares four different uncertainty quantification approaches:
- **MC Dropout** (Heteroscedastic)
- **Deep Ensembles** (Heteroscedastic)
- **Gaussian Process Regression** (Variational Inference)
- **NGBoost** (Natural Gradient Boosting)

Each method is evaluated across 10 independent runs with different random seeds to ensure robust and reproducible results.

## Project Structure

Masterarbeit/
├── models/ # Jupyter notebooks for each UQ method
│ ├── Deep Ensembles.ipynb
│ ├── Heteroscedastic MC Dropout.ipynb
│ ├── GPR.ipynb
│ ├── NGBoost.ipynb
│ ├── Modelresults/ # Saved predictions and results
│ └── Modelsaves/ # Trained model checkpoints
├── utils/ # Core utility modules
│ ├── data_prep.py # Data loading and preprocessing
│ ├── NN_model.py # Neural network architecture and training
│ └── metrices.py # Custom evaluation metrics
├── Test/ # Experimental notebooks and tests
├── data/ # Production data (not included in repo)
└── requirements.txt # Python dependencies


## Key Features

- **Heteroscedastic Uncertainty**: Decomposition into epistemic and aleatoric uncertainty
- **Hyperparameter Optimization**: Optuna-based search for all methods
- **Comprehensive Evaluation**: RMSE, MAE, R², NLL, CRPS, Coverage, MPIW
- **Reproducibility**: Fixed random seeds and consistent train/val/test splits
- **GPU Acceleration**: CUDA support for PyTorch models

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt

Main libraries:

PyTorch (with CUDA support)
GPyTorch
NGBoost
Optuna
Uncertainty Toolbox
scikit-learn
pandas, numpy, matplotlib
Usage
Data Preparation: Configure the data path in data_prep.py based on your system
Run Notebooks: Execute notebooks in models folder for each UQ method
Results: Predictions and metrics are saved in Modelresults
Each notebook follows the same structure:

Data loading and preprocessing
Hyperparameter optimization with Optuna
Model training and evaluation across 10 runs
Results saving and visualization
Evaluation Metrics
Accuracy: RMSE, MAE, R², Correlation
Probabilistic Scoring: Negative Log-Likelihood (NLL), CRPS
Calibration: Coverage (95% CI), Mean Prediction Interval Width (MPIW)
Random Seeds
All experiments use the same 10 seeds for fair comparison:

Author
Fynn - Master's Thesis in Management and Engineering

License
This project is part of academic research. Please contact the author for usage permissions.
