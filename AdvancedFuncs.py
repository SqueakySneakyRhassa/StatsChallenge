import numpy as np
import pandas as pd



def generate_train_val_sets(X, Y, y_columns=None):
    if y_columns is None:
        y_columns = ['ClaimAmount', 'PurePremium', 'Frequency', 'AvgClaimAmount', 'Exposure', 'ClaimNb']
    
    # Ensure X and Y have the same index
    if not X.index.equals(Y.index):
        raise ValueError("Indices of X and Y DataFrames do not match.")
    
    # Merge the feature and predictor DataFrames
    df_merged = pd.concat([X, Y], axis=1)
    
    # Perform bootstrapping using the internal function
    train_df, val_df = bootstrap_sample(df_merged)
    
    # Separate the predictors and outcomes for the training and validation sets
    X_train = train_df.drop(columns=y_columns)
    Y_train = train_df[y_columns]
    X_val = val_df.drop(columns=y_columns)
    Y_val = val_df[y_columns]
    
    return X_train, Y_train, X_val, Y_val


def bootstrap_sample(df):
    # Step 1: Determine the length of the DataFrame
    n = len(df)

    # Step 2: Draw random indices with replacement for the training set
    train_indices = np.random.choice(a=n, size=n, replace=True)
    
    # Step 3: Identify indices for the validation set (those not included in the bootstrap sample)
    all_indices = set(range(n))
    train_indices_set = set(train_indices)
    val_indices = np.array(list(all_indices - train_indices_set))
    
    # Step 4 & 5: Extract the training and validation sets using the indices
    train_df = df.iloc[train_indices].reset_index(drop=True)
    val_df = df.iloc[val_indices].reset_index(drop=True)
    
    return train_df, val_df