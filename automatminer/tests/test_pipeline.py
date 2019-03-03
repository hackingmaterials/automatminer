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
from automatminer.presets import get_preset_config
from automatminer.utils.pkg import AutomatminerError

test_dir = os.path.dirname(__file__)
CACHE_SRC = os.path.join(test_dir, "cache.json")
DIGEST_PATH = os.path.join(test_dir, "matdigest.txt")
PIPE_PATH = os.path.join(test_dir, "test_pipe.p")


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


# todo: figure out a way to run these tests with TPOTAdaptor!
class TestMatPipe(unittest.TestCase):
    def setUp(self):
        df = load_dataset("elastic_tensor_2015").rename(
            columns={"formula": "composition"})
        self.df = df[["composition", "K_VRH"]]
        self.df_struc = df[["composition", "structure", "K_VRH"]]
        self.extra_features = df["G_VRH"]
        self.target = "K_VRH"
        self.config = get_preset_config("debug_single")
        self.config_cached = get_preset_config("debug_single",
                                               cache_src=CACHE_SRC)
        self.pipe = MatPipe(**self.config)
        self.pipe_cached = MatPipe(**self.config_cached)

    def test_transferability(self):
        df_train = self.df.iloc[:200]
        df_test = self.df.iloc[201:250]
        self.pipe.fit(df_train, self.target)
        df_test = self.pipe.predict(df_test, self.target)
        true = df_test[self.target]
        test = df_test[self.target + " predicted"]
        self.assertTrue("composition" not in df_test.columns)
        self.assertTrue(r2_score(true, test) > 0.5)

        # Use the same pipe object by refitting and reusing
        df_train2 = self.df.iloc[250:450]
        df_test2 = self.df.iloc[451:500]
        self.pipe.fit(df_train2, self.target)
        df_test2 = self.pipe.predict(df_test2, self.target)
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
        df_test = self.pipe.predict(df_test, self.target)
        true = df_test[self.target]
        test = df_test[self.target + " predicted"]
        self.assertTrue(r2_score(true, test) > 0.75)

    @unittest.skipIf(int(os.environ.get("SKIP_INTENSIVE", 0)),
                     "Test too intensive for CircleCI commit builds.")
    def test_benchmarking_no_cache(self):
        pipe = self.pipe
        # Make sure we can't run a cached run with no cache AF and cache pipe
        with self.assertRaises(AutomatminerError):
            self._run_benchmark(cache=True, pipe=pipe)

        self._run_benchmark(cache=False, pipe=pipe)

    @unittest.skipIf(int(os.environ.get("SKIP_INTENSIVE", 0)),
                     "Test too intensive for CircleCI commit builds.")
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

        filename = os.path.join(test_dir, PIPE_PATH)
        self.pipe.save(filename=filename)
        self.pipe = MatPipe.load(filename, logger=False)
        df_test = self.pipe.predict(self.df[-220:-201], self.target)
        self.assertTrue(self.target in df_test.columns)
        self.assertTrue(self.target + " predicted" in df_test.columns)

        digest_file = os.path.join(test_dir, DIGEST_PATH)
        digest = self.pipe.digest(filename=digest_file)
        self.assertTrue(os.path.isfile(digest_file))
        self.assertTrue(isinstance(digest, str))

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
        for remnant in [CACHE_SRC, PIPE_PATH, DIGEST_PATH]:
            if os.path.exists(remnant):
                os.remove(remnant)
