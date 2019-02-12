"""
Test logging related utils.
"""

import unittest
import os

import logging

from automatminer.utils.log import initialize_logger, \
    initialize_null_logger

run_dir = os.getcwd()


class TestLogTools(unittest.TestCase):

    def test_logger_initialization(self):
        logger_base_name = "TESTING"
        log = initialize_logger(logger_base_name, level=logging.DEBUG)
        log.info("Test logging.")
        log.debug("Test debug.")
        log.warning("Test warning.")

        # test the log is written to run dir (e.g. where the script was called
        # from and not the location of this test file
        log_file = os.path.join(run_dir, logger_base_name + ".log")
        self.assertTrue(os.path.isfile(log_file))

        with open(log_file, 'r') as f:
            lines = f.readlines()

        self.assertTrue("logging" in lines[0])
        self.assertTrue("debug" in lines[1])
        self.assertTrue("warning" in lines[2])

        null = initialize_null_logger("matbench_null")
        null.info("Test null log 1.")
        null.debug("Test null log 2.")
        null.warning("Test null log 3.")

        null_log_file = os.path.join(run_dir, logger_base_name + "_null.log")
        self.assertFalse(os.path.isfile(null_log_file))


if __name__ == "__main__":
    unittest.main()
