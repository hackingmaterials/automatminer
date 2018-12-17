"""
Utils for logging.
"""

import logging
import os
import sys


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
