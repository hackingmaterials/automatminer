import logging
import os
import sys
import warnings

import pandas as pd
from sklearn.exceptions import NotFittedError


class MatbenchError(BaseException):
    """
    Exception specific to matbench methods.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "MatbenchError : " + self.msg


def initialize_logger(name, filepath='.', filename=None, level=None):
    """Initialize the default logger with stdout and file handlers.

    Args:
        name (str): The package name.
        filepath (str): Path to the folder where the log file will be written.
        filename (str): The log filename.
        level (int): The log level. For example logging.DEBUG.
    Returns:
        (Logger): A logging instance with customized formatter and handlers.
    """
    level = level or logging.INFO
    filename = filename or name + ".log"

    logger = logging.getLogger(name)
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler = logging.FileHandler(os.path.join(filepath, filename), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(screen_handler)
    logger.addHandler(handler)
    return logger


def initialize_null_logger(name):
    """Initialize the a dummy logger which will swallow all logging commands.
    Returns:
        (Logger): The package name.
        (Logger): A dummy logging instance with no output.
    """
    logger = logging.getLogger(name + "_null")
    logger.addHandler(logging.NullHandler())
    return logger


def is_greater_better(scoring_function):
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
        'recall_weighted', 'roc_auc' 'r2', 'neg_median_absolute_error',
        'neg_mean_absolute_error', 'neg_mean_squared_error'
    }

    desired_low_metrics = {
        'median_absolute_error',
        'mean_absolute_error',
        'mean_squared_error'
    }

    # Check to ensure no metrics are accidentally placed in both sets
    if desired_high_metrics.intersection(desired_low_metrics):
        raise MatbenchError("Error, there is a metric in both desired"
                            " high and desired low metrics")

    if scoring_function not in desired_high_metrics \
            and scoring_function not in desired_low_metrics:

        warnings.warn(
            'The scoring_function: "{}" not found; continuing assuming'
            ' greater score is better'.format(scoring_function))

    # True if not in either set or only in desired_high,
    # False if in desired_low or both sets
    return scoring_function not in desired_low_metrics


def compare_columns(df1, df2, ignore=None):
    """
    Compare the columns of a dataframe.

    Args:
        df1 (pandas.DataFrame): The first dataframe.
        df2 (pandas.DataFrame): The second dataframe.
        ignore ([str]): The feature labels to ignore in the analyis.

    Returns:
        (dict): {"df1_not_in_df2": [The columns in df1 not in df2],
                 "df2_not_in_df1": [The columns in df2 not in df1],
                 "mismatch": (bool)}
    """
    ignore = () if ignore is None else ignore
    df2_not_in_df1 = [f for f in df2.columns if f not in df1.columns and f not in ignore]
    df1_not_in_df2 = [f for f in df1.columns if f not in df2.columns and f not in ignore]
    matched = not (df2_not_in_df1 + df1_not_in_df2)
    return {"df2_not_in_df1": df2_not_in_df1,
            "df1_not_in_df2": df1_not_in_df2,
            "mismatch": not matched}


def regression_or_classification(series):
    """
    Determine if a series (target column) is numeric or categorical, to
    decide on the problem as regression or classification.

    Args:
        series (pandas.Series): The target column.

    Returns:
        (str): "regression" or "classification"
    """
    try:
        pd.to_numeric(series, errors="raise")
        return "regression"
    except (ValueError, TypeError):
        return "classification"


def check_fitted(func):
    """
    Decorator to check if a transformer has been fitted.
    Args:
        func: A function or method.

    Returns:
        A wrapper function for the input function/method.
    """
    def wrapper(*args, **kwargs):
        if not hasattr(args[0], "is_fit"):
            raise AttributeError("Method using check_fitted has no is_fit attr"
                                 " to check if fitted!")
        if not args[0].is_fit:
            raise NotFittedError("{} has not been fit!"
                                 "".format(args[0].__class__.__name__))
        else:
            return func(*args, **kwargs)
    return wrapper


def set_fitted(func):
    """
    Decorator to ensure a transformer is fitted properly.
    Args:
        func: A function or method.

    Returns:
        A wrapper function for the input function/method.
    """
    def wrapper(*args, **kwargs):
        args[0].is_fit = False
        result = func(*args, **kwargs)
        args[0].is_fit = True
        return result
    return wrapper


if __name__ == "__main__":
    s = pd.Series(data=["4", "5", "6"])
    print(regression_or_classification(s))
