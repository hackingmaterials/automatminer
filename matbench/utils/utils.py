import importlib
import logging
import os
import sys
import warnings


class MatbenchError(BaseException):
    """
    Exception specific to matbench methods.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "MatbenchError : " + self.msg


def setup_custom_logger(name='matbench_logger', filepath='.',
                        filename='matbench.log', level=None):
    """
    Custom logger with both screen and file handlers. This is particularly
    useful if there are other programs that call on logging
    in which case the log results and their levels are distinct and clear.

    Args:
        name (str): logger name to distinguish between different codes.
        filepath (str): path to the folder where the logfile is meant to be
        filename (str): log file filename
        level (int): log level in logging package; example: logging.DEBUG

    Returns: a logging instance with customized formatter and handlers
    """
    level = level or logging.INFO
    logger = logging.getLogger(name)
    importlib.reload(logging)
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
