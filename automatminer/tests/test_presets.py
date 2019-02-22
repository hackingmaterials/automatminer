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

    def test_express(self):
        express = get_preset_config("express")
        for k in KEYSET:
            self.assertTrue(k in express.keys())

    def test_heavy(self):
        heavy = get_preset_config("heavy")
        for k in KEYSET:
            self.assertTrue(k in heavy.keys())

    def test_caching_powerup(self):
        cache_src = "./somefile.json"
        prod = get_preset_config("production", cache_src=cache_src)
        self.assertEqual(prod[AF_KEY].cache_src, cache_src)

    def test_missing(self):
        with self.assertRaises(ValueError):
            _ = get_preset_config("QWERTYUIOP1234567890")
