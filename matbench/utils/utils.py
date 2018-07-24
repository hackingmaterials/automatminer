

class MatbenchError(Exception):
    """
    Exception specific to matbench methods.
    """

    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return "AmsetError : " + self.msg


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