"""
Tools and utils for machine learning.
"""

import warnings

import pandas as pd

from automatminer.utils.pkg import AutomatminerError

AMM_REG_NAME = "regression"
AMM_CLF_NAME = "classification"

def is_greater_better(scoring_function) -> bool:
    """
    Determines whether scoring_function being greater is more favorable/better.
    Args:
        scoring_function (str): the name of the scoring function supported by
            TPOT and sklearn. Please see below for more information.
    Returns (bool): Whether the scoring metric should be considered better if
        it is larger or better if it is smaller
    """
    desired_high_metrics = {
        'accuracy', 'adjusted_rand_score', 'average_precision',
        'balanced_accuracy', 'f1', 'f1_macro', 'f1_micro', 'f1_samples',
        'f1_weighted', 'precision', 'precision_macro', 'precision_micro',
        'precision_samples', 'precision_weighted', 'recall',
        'recall_macro', 'recall_micro', 'recall_samples',
        'recall_weighted', 'roc_auc', 'r2', 'r2_score',
        'neg_median_absolute_error', 'neg_mean_absolute_error',
        'neg_mean_squared_error'
    }

    desired_low_metrics = {
        'median_absolute_error',
        'mean_absolute_error',
        'mean_squared_error'
    }

    # Check to ensure no metrics are accidentally placed in both sets
    if desired_high_metrics.intersection(desired_low_metrics):
        raise AutomatminerError("Error, there is a metric in both desired"
                                " high and desired low metrics")

    if scoring_function not in desired_high_metrics \
            and scoring_function not in desired_low_metrics:
        warnings.warn(
            'The scoring_function: "{}" not found; continuing assuming'
            ' greater score is better'.format(scoring_function))

    # True if not in either set or only in desired_high,
    # False if in desired_low or both sets
    return scoring_function not in desired_low_metrics


def regression_or_classification(series) -> str:
    """
    Determine if a series (target column) is numeric or categorical, to
    decide on the problem as regression or classification.

    Args:
        series (pandas.Series): The target column.

    Returns:
        (str): "regression" or "classification"
    """
    if series.dtypes.name == "bool":
        return AMM_CLF_NAME
    else:
        unique = series.unique().tolist()
        if len(unique) == 2 and all([un in [0, 1] for un in unique]):
            return AMM_CLF_NAME
        else:
            try:
                pd.to_numeric(series, errors="raise")
                return AMM_REG_NAME
            except (ValueError, TypeError):
                return AMM_CLF_NAME
