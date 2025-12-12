# Master's Thesis: "Reliable Flange Length Estimation in Deep Drawing: A Comparative Study of Uncertainty-Aware Machine Learning Models Using Real-World Data"

This repository contains the implementation and evaluation of four uncertainty quantification (UQ) methods for regression tasks on a real-wolrd dataset from a deep drawing process, developed as part of the Master's thesis.


## Overview

The project compares four different uncertainty quantification approaches:
- **MC Dropout** (Heteroscedastic)
- **Deep Ensembles** (Heteroscedastic)
- **Sparse Gaussian Process Regression**
- **NGBoost** (Natural Gradient Boosting)

Each method is evaluated across 10 independent runs with different random seeds to ensure robust and reproducible results.

## Project Structure
```
Masterarbeit/
├── data/Produktionsdaten # Jupyter notebooks for data exploration and preprocessing
| ├── data_analysis_Prod_data.ipynb
├── models/ # Jupyter notebooks for each UQ method
│ ├── Deep Ensembles.ipynb
│ ├── Heteroscedastic MC Dropout.ipynb
│ ├── GPR.ipynb
│ ├── NGBoost.ipynb
│ ├── Modelresults/ # Saved predictions and results
├── utils/ # Core utility modules
│ ├── data_prep.py # Data loading and preprocessing
│ ├── NN_model.py # Neural network architecture and training
│ └── metrices.py # Custom evaluation metrics
├── data/ # Production data (not included in repo)
└── requirements.txt # Python dependencies
```

## Key Features

- **Heteroscedastic Uncertainty**: Decomposition into epistemic and aleatoric uncertainty for MC Dropout and Deep Ensembles
- **Hyperparameter Optimization**: Optuna-based search for all methods
- **Comprehensive Evaluation**: RMSE, MAE, R², Correlation, NLL, CRPS, Coverage, MPIW
- **Reproducibility**: Fixed random seeds and consistent train/val/test splits
- **GPU Acceleration**: CUDA support for PyTorch models

## Requirements

Install dependencies with:
```bash
pip install -r requirements.txt
```

## Main libraries

- PyTorch (with CUDA support)
- GPyTorch
- NGBoost
- Optuna
- Uncertainty Toolbox
- scikit-learn
- pandas, numpy, matplotlib

## Usage:
All models are implemented in Jupyter notebooks.
Each notebook follows the same structure:
- Data loading and preprocessing
- Hyperparameter optimization with Optuna
- Model training and evaluation across 10 runs
- Results saving and visualization

## Evaluation Metrics
    Accuracy: RMSE, MAE, R², Correlation
    Probabilistic Scoring: Negative Log-Likelihood (NLL), CRPS
    Calibration: Coverage (95% CI), Mean Prediction Interval Width (MPIW)

## Random Seeds
All experiments use the same 10 seeds for fair comparison:
```python
[42, 123, 777, 2024, 5250, 8888, 9876, 10001, 31415, 54321]
```

Author
Fynn - Master's Thesis in Management and Engineering

License
This project is part of academic research. Please contact the author for usage permissions.
