import logging
import os
import sys


class MatbenchError(Exception):
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
    in which case the log results and their levels are distict and clear.

    Args:
        name (str): logger name to distinguish between different codes.
        filepath (str): path to the folder where the logfile is meant to be
        filename (str): log file filename
        level (int): log level in logging package; example: logging.DEBUG

    Returns: a logging instance with customized formatter and handlers
    """
    level = level or logging.DEBUG
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler(os.path.join(filepath, filename), mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
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

    Returns (bool):
    """
    if scoring_function in [
        'accuracy', 'adjusted_rand_score', 'average_precision',
        'balanced_accuracy','f1', 'f1_macro', 'f1_micro', 'f1_samples',
        'f1_weighted', 'precision', 'precision_macro', 'precision_micro',
        'precision_samples','precision_weighted', 'recall',
        'recall_macro', 'recall_micro','recall_samples',
        'recall_weighted', 'roc_auc'] + \
            ['r2', 'neg_median_absolute_error', 'neg_mean_absolute_error',
            'neg_mean_squared_error']:
        return True
    elif scoring_function in ['median_absolute_error',
                              'mean_absolute_error',
                              'mean_squared_error']:
        return False
    else:
        raise MatbenchError('Unsupported scoring_function: "{}"'.format(
            scoring_function))