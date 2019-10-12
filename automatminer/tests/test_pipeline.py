"""
Tests for the top level interface.
"""

import unittest
import os.path

import pandas as pd
from matminer.datasets.dataset_retrieval import load_dataset
from sklearn.metrics import r2_score
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

from automatminer.pipeline import MatPipe
from automatminer.presets import get_preset_config, get_available_presets
from automatminer.utils.pkg import AutomatminerError

test_dir = os.path.dirname(__file__)
CACHE_SRC = os.path.join(test_dir, "cache.json")
DIGEST_PATH = os.path.join(test_dir, "matdigest.")
DIGEST_EXTS = ['txt', 'json', 'yml', 'yaml']
PIPE_PATH = os.path.join(test_dir, "test_pipe.p")
VERSION_PIPE_PATH = os.path.join(test_dir, "version_test.p")


class TestMatPipeSetup(unittest.TestCase):
    def setUp(self):
        self.config = get_preset_config('debug')

    def test_instantiation(self):
        learner = self.config['learner']
        autofeaturizer = self.config['autofeaturizer']
        with self.assertRaises(AutomatminerError):
            MatPipe(learner=learner)
        with self.assertRaises(AutomatminerError):
            MatPipe(autofeaturizer=autofeaturizer)
        with self.assertRaises(AutomatminerError):
            MatPipe(autofeaturizer=autofeaturizer, learner=learner)
        MatPipe()
        MatPipe(**self.config)

    def test_from_preset(self):
        for preset in get_available_presets():
            MatPipe.from_preset(preset)

        MatPipe.from_preset("debug", cache_src="some_file.json")
        MatPipe.from_preset("debug", logger=False)
        MatPipe.from_preset("debug", log_level=1)


def make_matpipe_test(config_preset):
    """
    Create a full matpipe test suite for a particular preset.

    Args:
        config_preset (str): A preset to test.

    Returns:
        TestMatPipe (unittest): A unittest for MatPipe as specifically
            implemented by a preset.

    """

    class TestMatPipe(unittest.TestCase):
        def setUp(self):
            df = load_dataset("elastic_tensor_2015").rename(
                columns={"formula": "composition"})
            self.df = df[["composition", "K_VRH"]]
            self.df_struc = df[["composition", "structure", "K_VRH"]]
            self.extra_features = df["G_VRH"]
            self.target = "K_VRH"
            self.config = get_preset_config(config_preset)
            self.config_cached = get_preset_config(config_preset,
                                                   cache_src=CACHE_SRC)
            self.pipe = MatPipe(**self.config)
            self.pipe_cached = MatPipe(**self.config_cached)

        def test_transferability(self):
            df_train = self.df.iloc[:200]
            df_test = self.df.iloc[201:250]
            self.pipe.fit(df_train, self.target)
            df_test = self.pipe.predict(df_test)
            true = df_test[self.target]
            test = df_test[self.target + " predicted"]
            self.assertTrue("composition" not in df_test.columns)
            self.assertTrue(r2_score(true, test) > 0.5)

            # Use the same pipe object by refitting and reusing
            df_train2 = self.df.iloc[250:450]
            df_test2 = self.df.iloc[451:500]
            self.pipe.fit(df_train2, self.target)
            df_test2 = self.pipe.predict(df_test2)
            true2 = df_test2[self.target]
            test2 = df_test2[self.target + " predicted"]
            self.assertTrue("composition" not in df_test2.columns)
            self.assertTrue(r2_score(true2, test2) > 0.5)

        def test_user_features(self):
            df = self.df
            df["G_VRH"] = self.extra_features
            self.assertTrue("G_VRH" in df.columns)
            self.assertTrue("K_VRH" in df.columns)
            df_train = df.iloc[:200]
            df_test = df.iloc[201:250]
            self.pipe.fit(df_train, self.target)

            # If shear modulus is included as a feature it should probably show up
            # in the final pipeline
            self.assertTrue("G_VRH" in self.pipe.learner.features)
            df_test = self.pipe.predict(df_test)
            true = df_test[self.target]
            test = df_test[self.target + " predicted"]
            self.assertTrue(r2_score(true, test) > 0.75)

        def test_benchmarking_no_cache(self):
            pipe = self.pipe
            # Make sure we can't run a cached run with no cache AF and cache pipe
            with self.assertRaises(AutomatminerError):
                self._run_benchmark(cache=True, pipe=pipe)

            self._run_benchmark(cache=False, pipe=pipe)

        def test_benchmarking_cache(self):
            pipe = self.pipe_cached

            # Make sure we can't run a cached run with no cache AF and cache pipe
            with self.assertRaises(AutomatminerError):
                self._run_benchmark(cache=False, pipe=pipe)
            self._run_benchmark(cache=True, pipe=pipe)

        def test_persistence_and_digest(self):
            with self.assertRaises(NotFittedError):
                self.pipe.save()
            df = self.df[-200:]
            self.pipe.fit(df, self.target)

            self.pipe.save(filename=PIPE_PATH)
            self.pipe = MatPipe.load(PIPE_PATH, logger=False)
            df_test = self.pipe.predict(self.df[-220:-201])
            self.assertTrue(self.target in df_test.columns)
            self.assertTrue(self.target + " predicted" in df_test.columns)

            for ext in DIGEST_EXTS:
                digest = self.pipe.digest(filename=DIGEST_PATH + ext)
                self.assertTrue(os.path.isfile(DIGEST_PATH + ext))
                self.assertTrue(isinstance(digest, str))

                digest = self.pipe.digest(output_format=ext)
                self.assertTrue(isinstance(digest, str))

            # Version test
            self.pipe.version = "not a real version"
            self.pipe.save(VERSION_PIPE_PATH)
            with self.assertRaises(AutomatminerError):
                MatPipe.load(VERSION_PIPE_PATH)

        def _run_benchmark(self, cache, pipe):
            # Test static, regular benchmark (no fittable featurizers)
            df = self.df.iloc[500:600]
            kfold = KFold(n_splits=5)
            df_tests = pipe.benchmark(df, self.target, kfold, cache=cache)
            self.assertEqual(len(df_tests), kfold.n_splits)

            # Make sure we retain a good amount of test samples...
            df_tests_all = pd.concat(df_tests)
            self.assertGreaterEqual(len(df_tests_all), 0.95 * len(df))

            # Test static subset of kfold
            df2 = self.df.iloc[500:550]
            df_tests2 = pipe.benchmark(df2, self.target, kfold,
                                       fold_subset=[0, 3], cache=cache)
            self.assertEqual(len(df_tests2), 2)

        def tearDown(self):
            digests = [DIGEST_PATH + ext for ext in DIGEST_EXTS]
            for remnant in [CACHE_SRC, PIPE_PATH, *digests]:
                if os.path.exists(remnant):
                    os.remove(remnant)

    return TestMatPipe


class MatPipeDebugTest(make_matpipe_test("debug")):
    pass


class MatPipeDebugSingleTest(make_matpipe_test("debug_single")):
    pass
