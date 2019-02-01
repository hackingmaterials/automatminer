"""
Testing the preset configurations for MatPipe.

Mainly ensuring all args can be passed to matpipe constituent parts correctly.
"""
import unittest

from automatminer.presets import get_preset_config


class TestMatPipe(unittest.TestCase):
    def test_production(self):
        prod = get_preset_config("production")

    def test_debug(self):
        debug = get_preset_config("debug")

    def test_fast(self):
        fast = get_preset_config("fast")

    def test_default(self):
        default = get_preset_config("default")
