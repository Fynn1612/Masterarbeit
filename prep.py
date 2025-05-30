# fuction that takes as input a dataframe and a categorial variables
# determine the frequency of each category in the dataframe
# return a dataframe with the frequency of each category
# delete the original columns from the dataframe
import pandas as pd
def prep(df: pd.DataFrame, cat_var: str) -> pd.DataFrame:
    """
    Prepares the DataFrame by calculating the frequency of each category in the specified categorical variable.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    cat_var (str): The name of the categorical variable to analyze.
    
    Returns:
    pd.DataFrame: A DataFrame containing the frequency of each category.
    """
    # Calculate the frequency of each category
    freq_df = df[cat_var].value_counts()
    df[f"{cat_var}_freq"] = df[cat_var].map(freq_df).fillna(0)
    # drop the original categorical variable from the DataFrame  
    freq_df.columns = [cat_var, 'frequency']
    
    # Drop the original categorical variable from the DataFrame
    df = df.drop(columns=[cat_var], axis=1, inplace=True)
    
    return freq_df