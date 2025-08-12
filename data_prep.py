# fuction that takes as input train_set, val_set, test_set as dataframes and a list of categorial variables
# determine the frequency of each category in the dataframes
# return train_set, val_set, test_set dataframes with the frequency of each category
# delete the original columns from the dataframes
def cat_transform(train_df, val_df, test_df, cat_vars):
    """
    train_set, val_set, test_set
    cat_ vars: list categorical variables to transform
    return: original dataframe with the frequency of each category 
    and removed original columns
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