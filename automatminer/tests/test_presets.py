"""
Testing the preset configurations for MatPipe.

Mainly ensuring all args can be passed to matpipe constituent parts correctly.
"""
import unittest

from automatminer.presets import get_preset_config

AF_KEY = "autofeaturizer"
DC_KEY = "cleaner"
FR_KEY = "reducer"
ML_KEY = "learner"
KEYSET = [AF_KEY, DC_KEY, FR_KEY, ML_KEY]


class TestMatPipe(unittest.TestCase):
    def test_production(self):
        prod = get_preset_config("production")
        for k in KEYSET:
            self.assertTrue(k in prod.keys())

    def test_debug(self):
        debug = get_preset_config("debug")
        for k in KEYSET:
            self.assertTrue(k in debug.keys())

    def test_debug_single(self):
        debug_single = get_preset_config("debug_single")
        for k in KEYSET:
            self.assertTrue(k in debug_single.keys())

    def test_fast(self):
        fast = get_preset_config("fast")
        for k in KEYSET:
            self.assertTrue(k in fast.keys())

    def test_default(self):
        default = get_preset_config("default")
        for k in KEYSET:
            self.assertTrue(k in default.keys())
