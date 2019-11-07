"""
Utils for logging.
"""

import os
import sys
import logging
import datetime

AMM_LOGGER_BASENAME = "automatminer"
AMM_LOG_FIT_STR = "fitting"
AMM_LOG_TRANSFORM_STR = "transforming"
AMM_LOG_PREDICT_STR = "predicting"

AMM_DEFAULT_LOGGER = True


def initialize_logger(logger_name, log_dir=".", level=None) -> logging.Logger:
    """Initialize the default logger with stdout and file handlers.

    Args:
        logger_name (str): The package name.
        log_dir (str): Path to the folder where the log file will be written.
        level (int): The log level. For example logging.DEBUG.
    Returns:
        (Logger): A logging instance with customized formatter and handlers.
    """
    level = level or logging.INFO

    logger = logging.getLogger(logger_name)
    logger.handlers = []  # reset logging handlers if they already exist

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-8s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    logpath = os.path.join(log_dir, logger_name)
    if os.path.exists(logpath + ".log"):
        logpath += "_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logpath += ".log"

    handler = logging.FileHandler(logpath, mode="w")
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger.setLevel(level)
    logger.addHandler(screen_handler)
    logger.addHandler(handler)
    return logger


def initialize_null_logger(name) -> logging.Logger:
    """Initialize the a dummy logger which will swallow all logging commands.
    Returns:
        (Logger): The package name.
        (Logger): A dummy logging instance with no output.
    """
    logger = logging.getLogger(name + "_null")
    logger.addHandler(logging.NullHandler())
    return logger


def log_progress(logger, operation):
    """
    Decorator to auto-log progress before and after executing a method, such
    as fit and transform. Should only be applied to DataFrameTransformers.

    For example,

    INFO: Beginning AutoFeaturizer fitting.
    ... autofeaturizer logs ...
    INFO: Finished AutoFeaturizer fitting.

    Args:
        logger (logging.Logger): A logger object to help log progress.
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
            logger.info("{}Starting {}.".format(self._log_prefix, operation))
            result = meth(*args, **kwargs)
            logger.info("{}Finished {}.".format(self._log_prefix, operation))
            return result

        return wrapper

    return decorator_wrapper
