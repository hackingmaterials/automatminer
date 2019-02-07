"""
Utils for logging.
"""

import logging
import os
import sys

LOG_FIT_STR = "fitting"
LOG_TRANSFORM_STR = "transforming"


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


def log_progress(operation):
    """
    Decorator to auto-log progress before and after executing a method, such
    as fit and transform. Should only be applied to DataFrameTransformers.

    For example,

    INFO: Beginning AutoFeaturizer fitting.
    ... autofeaturizer logs ...
    INFO: Finished AutoFeaturizer fitting.

    Args:
        operation (str): Some info about the operation you want to log.

    Returns:
        A wrapper for the input method.
    """

    def decorator_wrapper(meth):
        def wrapper(*args, **kwargs):
            """
            Wrapper for a method to log.

            Args:
                operation (str): The operation to be logging.

            Return:
                result: The method result.
            """
            self = args[0]
            name = self.__class__.__name__
            self.logger.info("{} starting {}.".format(name, operation))
            result = meth(*args, **kwargs)
            self.logger.info("{} finished {}.".format(name, operation))
            return result

        return wrapper

    return decorator_wrapper
