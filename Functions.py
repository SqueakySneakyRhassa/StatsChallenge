from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.datasets import fetch_openml
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_tweedie_deviance,
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml


# DATA PREPARATION STEPS
# ----------------------

# Primary functions to run the entire process of insurance data processing
def load_and_basic_process(n_samples=None):
    """
    Process the insurance data and return the necessary components for analysis.
    """
    df = load_mtpl2(n_samples)
    df = clean_and_clip_data(df)
    df = calculate_derived_variables(df)
    
    return df

# This enables our data processing steps to use on the dataframe after loading it in
def df_data_process(df):
    df_X = df.copy()

    # Define the log transformation for 'Density'
    def log_transform(x):
        return np.log1p(x)  # Using log1p for log(x+1)

    # Apply log transformation and standard scaling to 'Density'
    df_X['Density'] = df_X[['Density']].apply(log_transform)
    df_X['Density'] = StandardScaler().fit_transform(df_X[['Density']])


    # One-hot encode categorical variables
    categorical_cols = ['VehBrand', 'VehPower', 'VehGas', 'Region', 'Area']  # Specify actual categorical columns
    df_X = pd.get_dummies(df_X, columns=categorical_cols)

    # 'ClaimAmount', 'PurePremium', 'Frequency', 'AvgClaimAmount', 'Exposure', 'ClaimNb' - targets/weights
    derived_variables = ['ClaimAmount', 'PurePremium', 'Frequency', 'AvgClaimAmount', 'Exposure', 'ClaimNb']
    df_Y = df_X[derived_variables]
    df_X = df_X.drop(columns=derived_variables)  # Drop original 'VehAge' and 'DrivAge' after binning

    return df_X, df_Y



# Secondary Functions for load_and_basic_process

# Function to load the dataset
def load_mtpl2(n_samples=None):
    """
    Fetch the French Motor Third-Party Liability Claims dataset.
    """
    parser_option = 'auto'

    # Load frequency data
    df_freq = fetch_openml(data_id=41214, as_frame=True, parser=parser_option).data
    df_freq["IDpol"] = df_freq["IDpol"].astype(int)
    df_freq.set_index("IDpol", inplace=True)

    # Load severity data
    df_sev = fetch_openml(data_id=41215, as_frame=True, parser=parser_option).data
    df_sev = df_sev.groupby("IDpol").sum()

    # Merge datasets
    df = df_freq.join(df_sev, how="left")
    df["ClaimAmount"].fillna(0, inplace=True)

    # Unquote string fields
    for column_name in df.columns[df.dtypes.values == object]:
        df[column_name] = df[column_name].str.strip("'")

    return df.iloc[:n_samples]

# Function for data cleaning and clipping
def clean_and_clip_data(df):
    """
    Clean and clip data for reasonable observation ranges.
    """
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    df["Exposure"] = df["Exposure"].clip(upper=1)
    df["ClaimAmount"] = df["ClaimAmount"].clip(upper=200000)
    df.loc[(df["ClaimAmount"] == 0) & (df["ClaimNb"] >= 1), "ClaimNb"] = 0
    return df


# Function to calculate derived variables
def calculate_derived_variables(df):
    """
    Calculate derived variables like pure premium, frequency, and average claim amount.
    """
    df["PurePremium"] = df["ClaimAmount"] / df["Exposure"]
    df["Frequency"] = df["ClaimNb"] / df["Exposure"]
    df["AvgClaimAmount"] = df["ClaimAmount"] / np.fmax(df["ClaimNb"], 1)
    return df



# MODEL APPLICATION STEPS
# ------------------------

# Primary Score / Evaluation Function

def score_estimator(estimator, X_train, X_test, df_train, df_test, target, weights, tweedie_powers=None):
    """
    Evaluate an estimator (/ML Model) on train and test sets with different metrics.
    """
    metrics = define_metrics(tweedie_powers)
    scores = evaluate_on_datasets(estimator, X_train, X_test, df_train, df_test, target, weights, metrics)
    return format_scores(scores)

# Secondary functions for score_estimator

def define_metrics(tweedie_powers):
    """
    Define metrics for evaluation.
    """
    base_metrics = [("D² explained", None), ("mean abs. error", mean_absolute_error), ("mean squared error", mean_squared_error)]
    tweedie_metrics = [("mean Tweedie dev p={:.4f}".format(p), partial(mean_tweedie_deviance, power=p)) for p in tweedie_powers or []]
    return base_metrics + tweedie_metrics

def evaluate_on_datasets(estimator, X_train, X_test, df_train, df_test, target, weights, metrics):
    """
    Evaluate the estimator on training and test datasets.
    """
    results = []
    for subset_label, X, df in [("train", X_train, df_train), ("test", X_test, df_test)]:
        y, _weights = df[target], df[weights]
        results.extend(evaluate_metrics(estimator, X, y, _weights, subset_label, metrics))
    return results

def format_scores(scores):
    """
    Format the evaluation scores into a DataFrame.
    """
    df_scores = pd.DataFrame(scores).set_index(["metric", "subset"])
    return df_scores.score.unstack(-1).round(4).loc[:, ["train", "test"]]

# Tertiary Functions

# Secondary function of evaluate_on_datasets

def evaluate_metrics(estimator, X, y, weights, subset_label, metrics):
    """
    Evaluate specified metrics for the estimator.
    """
    scores = []
    for score_label, metric in metrics:
        y_pred = predict_values(estimator, X)
        score = calculate_score(metric, estimator, X, y, y_pred, weights)
        scores.append({"subset": subset_label, "metric": score_label, "score": score})
    return scores

# Quarternary Functions

# Secondary functions of evaluate_metrics

def predict_values(estimator, X):
    """
    Predict values using the estimator.
    """
    if isinstance(estimator, tuple) and len(estimator) == 2:  # For combined models
        est_freq, est_sev = estimator
        return est_freq.predict(X) * est_sev.predict(X)
    else:
        return estimator.predict(X)
    
def calculate_score(metric, estimator, X, y, y_pred, weights):
    """
    Calculate the score for a given metric.
    """
    if metric is None:
        return estimator.score(X, y, sample_weight=weights) if hasattr(estimator, "score") else None
    else:
        return metric(y, y_pred, sample_weight=weights)