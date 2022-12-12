import pandas as pd
from scipy.stats import pointbiserialr, chi2_contingency
import numpy as np


def compute_pb_correlation(input_df: pd.DataFrame, features: list, target_column: str) -> pd.DataFrame:
    results = []
    for feature_a in features:
        corr = pointbiserialr(input_df[target_column], input_df[feature_a])
        results.append({"feature": feature_a, "corr": corr[0], "abs_corr": abs(corr[0])})
    return pd.DataFrame(results)


def cramers_v(values_a: pd.Series, values_b: pd.Series) -> float:
    confusion_matrix = pd.crosstab(values_a, values_b).to_numpy()
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n_pop = confusion_matrix.sum()
    minimum_dimension = min(confusion_matrix.shape) - 1

    # Calculate Cramer's V
    result = np.sqrt((chi2 / n_pop) / minimum_dimension)
    return result


def compute_cramers_v(input_df: pd.DataFrame, features: list, target_column: str) -> pd.DataFrame:
    results = []
    for feature_a in features:
        cra_v = cramers_v(input_df[feature_a], input_df[target_column])
        results.append({"feature": feature_a, "cra_v": cra_v})

    return pd.DataFrame(results)
