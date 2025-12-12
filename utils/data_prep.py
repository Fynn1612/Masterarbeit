"""
Data Preprocessing Utilities for Master's Thesis

This module provides functions for loading, transforming, and splitting
the production data for uncertainty quantification experiments.

Key functions:
- cat_transform: Converts categorical variables to frequency encoding
- load_tranform_and_split_data: Main data loading and preprocessing pipeline
- set_seed: Sets random seeds for reproducibility across all libraries
"""
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd
import torch
import numpy as np
import random

# function that transforms the categorical variables in the dataframes to their frequency
# delete the original columns from the dataframes
def cat_transform(train_df, val_df, test_df, cat_vars):
    """
    Transform categorical variables to their frequency in the dataframe.
    The original columns are removed from the dataframes.

    Args:
        train_set, val_set, test_set:    dataframes with the data
        cat_vars:                       list categorical variables to transform

    Returns:
        train_df, val_df, test_df
        original dataframes with the frequency of each category and removed original columns
    """
    # loop through each categorical variable
    if not isinstance(cat_vars, list):
        raise ValueError("cat_vars should be a list of categorical variables")
    
    for cat_var in cat_vars:
        # count the frequency of the category in the first dataframe
        # just use the training set to calculate the frequency to prevent data leakage
        freq = train_df[cat_var].value_counts()
        # map the frequency to each dataframe in the list
        for df in [train_df, val_df, test_df]:
            df[f"{cat_var}_freq"] = df[cat_var].map(freq).fillna(0)
            # drop the cat_var column
            df.drop(columns=[cat_var], inplace=True)
    return train_df, val_df, test_df


# function that loads, transforms and splits the data it into train, val and test sets, and returns the feature names
def load_tranform_and_split_data(DATA_PATH, target, split_ratio=(0.6, 0.2, 0.2)):
    """
    Load, transform and split the data into train, val and test sets.
    Default: 60% training, 20% validation, 20% test set.

    Args:
        target : target variable to predict
        split_ratio : tuple with the split ratio for train, val and test sets, default is (0.6, 0.2, 0.2)

    Returns: 
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names 
        splitted and transformed dataframes with the features and target variable, and the feature names
    """
    
    # check the computer name to determine the path to the data
    # if os.environ['COMPUTERNAME'] == 'FYNN':            # name of surface PC
    #     path = r"C:\Users\Surface\Masterarbeit\data\Produktionsdaten\WZ_2_Feature_Engineered_Fynn6.xlsx"
    # elif os.environ['COMPUTERNAME'] == 'FYNNS-PC':  # desktop name
    #     path = r"C:\Users\test\Masterarbeit\data\WZ_2_Feature_Engineered_Fynn6.xlsx"
        
    # else:
    #     raise ValueError("Unbekannter Computername: " + os.environ['COMPUTERNAME'])
    
    
    #load the data from the excel file
    df = pd.read_excel(DATA_PATH)
    
    # get the numerical features
    data_num = df.drop(target, axis = 1, inplace=False)
    # get the target values
    data_labels = df[target].to_numpy()

    #calculate the validation size relative to the remaining data after test split
    test_size = split_ratio[2]
    val_size = split_ratio[1] / (1 - test_size) 
    # split the data into training, validation and test sets
    # 60% training, 20%, validation, 20% test
    X_temp, X_test_prep, y_temp, y_test = train_test_split(data_num, data_labels, test_size= test_size, random_state=42)
    X_train_prep, X_val_prep, y_train, y_val = train_test_split(X_temp, y_temp, test_size= val_size, random_state=42)

    # use custom function "cat_transform" to map the categorical features with their frequencies
    X_train_prep, X_val_prep, X_test_prep = cat_transform(X_train_prep, X_val_prep, X_test_prep, ['BT_NR', 'STP_NR'])
    # print(X_train_prep.columns)

    # pipeline for preprocessing the data
    # Standard Scaler for distribution with 0 mean and 1 std., normal distributed data
    data_pipeline = Pipeline([
        ('std_scaler', StandardScaler())
    ])

    # get the feature names after preprocessing for the feature importance
    feature_names = X_train_prep.columns

    # fit the pipeline to the data and transform it
    X_train = data_pipeline.fit_transform(X_train_prep)
    X_val = data_pipeline.transform(X_val_prep)
    X_test = data_pipeline.transform(X_test_prep)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

# function to set the seed for reproducibility
def set_seed(seed: int):
    """
    Set the seed for reproducibility across all random number generators.
    
    Sets seeds for Python's random, NumPy, and PyTorch (CPU and GPU).
    Also configures PyTorch for deterministic behavior.

    Args:
        seed : Integer seed for the random number generator
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

