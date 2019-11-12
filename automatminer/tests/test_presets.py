"""
Testing the preset configurations for MatPipe.

Mainly ensuring all args can be passed to matpipe constituent parts correctly.
"""
import unittest

from automatminer import MatPipe
from automatminer.presets import get_preset_config

AF_KEY = "autofeaturizer"
DC_KEY = "cleaner"
FR_KEY = "reducer"
ML_KEY = "learner"
KEYSET = [AF_KEY, DC_KEY, FR_KEY, ML_KEY]


class TestMatPipePresets(unittest.TestCase):
    def test_production(self):
        prod = get_preset_config("production")
        for k in KEYSET:
            self.assertTrue(k in prod.keys())
        MatPipe(**prod)

    def test_debug(self):
        debug = get_preset_config("debug")
        for k in KEYSET:
            self.assertTrue(k in debug.keys())
        MatPipe(**debug)

    def test_debug_single(self):
        debug_single = get_preset_config("debug_single")
        for k in KEYSET:
            self.assertTrue(k in debug_single.keys())
        MatPipe(**debug_single)

    def test_express(self):
        express = get_preset_config("express")
        for k in KEYSET:
            self.assertTrue(k in express.keys())
        MatPipe(**express)

    def test_express_single(self):
        express_single = get_preset_config("express_single")
        for k in KEYSET:
            self.assertTrue(k in express_single.keys())
        MatPipe(**express_single)

    def test_heavy(self):
        heavy = get_preset_config("heavy")
        for k in KEYSET:
            self.assertTrue(k in heavy.keys())
        MatPipe(**heavy)

    def test_caching_powerup(self):
        cache_src = "./somefile.json"
        prod = get_preset_config("production", cache_src=cache_src)
        self.assertEqual(prod[AF_KEY].cache_src, cache_src)
        MatPipe(**prod)

    def test_n_jobs_powerup(self):
        n_jobs = 1
        prod = get_preset_config("production", n_jobs=n_jobs)
        self.assertEqual(prod[AF_KEY].n_jobs, n_jobs)
        self.assertEqual(prod[ML_KEY].tpot_kwargs["n_jobs"], n_jobs)
        MatPipe(**prod)

    def test_missing(self):
        with self.assertRaises(ValueError):
            get_preset_config("QWERTYUIOP1234567890")
