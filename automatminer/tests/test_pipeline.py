"""
Tests for the top level interface.
"""

import os.path
import unittest

import pandas as pd
from automatminer.pipeline import MatPipe
from automatminer.presets import get_available_presets, get_preset_config
from automatminer.utils.pkg import AMM_SUPPORTED_EXTS, AutomatminerError
from matminer.datasets.dataset_retrieval import load_dataset
from sklearn.exceptions import NotFittedError
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

test_dir = os.path.dirname(__file__)
CACHE_SRC = os.path.join(test_dir, "cache.json")
PIPE_PATH = os.path.join(test_dir, "test_pipe.p")
VERSION_PIPE_PATH = os.path.join(test_dir, "version_test.p")
DIGEST_PATH = os.path.join(test_dir, "matdigest")


class TestMatPipeSetup(unittest.TestCase):
    def setUp(self):
        self.config = get_preset_config("debug")

    def test_instantiation(self):
        learner = self.config["learner"]
        autofeaturizer = self.config["autofeaturizer"]
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
        MatPipe.from_preset("debug")
        MatPipe.from_preset("debug", log_level=1)


def make_matpipe_test(config_preset, skip=None):
    """
    Create a full matpipe test suite for a particular preset.

    Args:
        config_preset (str): A preset to test.
        skip ([str]): Names of skippable tests. Check skippables for current
            lists of tests to skip. Useful for debugging where you only want to
            run a certain test.

    Returns:
        TestMatPipe (unittest): A unittest for MatPipe as specifically
            implemented by a preset.

    """

    n_jobs = 2
    skippables = [
        "transferability",
        "user_features",
        "predict_kwargs",
        "benchmarking",
        "persistence",
        "digests",
    ]
    if not skip:
        skip = []
    for s in skip:
        if s not in skippables:
            raise ValueError(
                f"{s} is not a skippable test. Choose from {skippables}"
            )
    reason = "Skip was requested."

    class TestMatPipe(unittest.TestCase):
        def setUp(self):
            df = load_dataset("elastic_tensor_2015").rename(
                columns={"formula": "composition"}
            )
            self.df = df[["composition", "K_VRH"]]
            self.df_struc = df[["composition", "structure", "K_VRH"]]
            self.extra_features = df["G_VRH"]
            self.target = "K_VRH"
            self.config = get_preset_config(config_preset, n_jobs=n_jobs)
            self.config_cached = get_preset_config(
                config_preset, cache_src=CACHE_SRC, n_jobs=n_jobs
            )
            self.pipe = MatPipe(**self.config)
            self.pipe_cached = MatPipe(**self.config_cached)

        @unittest.skipIf("transferability" in skip, reason)
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

        @unittest.skipIf("user_features" in skip, reason)
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

        @unittest.skipIf("predict_kwargs" in skip, reason)
        def test_predict_kwargs(self):
            # Test mat_pipe.predict()'s ignore and output_col kwargs.
            df_train = self.df.iloc[:200]
            df_test = self.df.iloc[201:250]
            ef = "ExtraFeature"
            df_test[ef] = [i + 100 for i in range(df_test.shape[0])]
            self.pipe.fit(df_train, self.target)

            self.assertTrue(ef in df_test.columns)
            self.assertTrue("composition" in df_test.columns)

            ignore = [ef, "composition"]
            predicted_ignored = self.pipe.predict(df_test, ignore=ignore)
            self.assertTrue(ef in predicted_ignored.columns)
            self.assertTrue("composition" in predicted_ignored.columns)

            predicted_none = self.pipe.predict(df_test, ignore=None)
            self.assertFalse(ef in predicted_none.columns)
            self.assertFalse("composition" in predicted_none.columns)

            some = ["composition"]
            predicted_some = self.pipe.predict(df_test, ignore=some)
            self.assertFalse(ef in predicted_some.columns)
            self.assertTrue("composition" in predicted_some.columns)

            output_col_name = self.target + "_pred"
            predicted_custom_col = self.pipe.predict(
                df_test, output_col=output_col_name
            )
            self.assertTrue(output_col_name in predicted_custom_col)

        @unittest.skipIf("benchmarking" in skip, reason)
        def test_benchmarking_no_cache(self):
            pipe = self.pipe
            # Make sure we can't run a cached run with no cache AF and cache pipe
            with self.assertRaises(AutomatminerError):
                self._run_benchmark(cache=True, pipe=pipe)

            self._run_benchmark(cache=False, pipe=pipe)

        @unittest.skipIf("benchmarking" in skip, reason)
        def test_benchmarking_cache(self):
            pipe = self.pipe_cached

            # Make sure we can't run a cached run with no cache AF and cache pipe
            with self.assertRaises(AutomatminerError):
                self._run_benchmark(cache=False, pipe=pipe)
            self._run_benchmark(cache=True, pipe=pipe)

        @unittest.skipIf("persistence" in skip, reason)
        def test_persistence(self):
            with self.assertRaises(NotFittedError):
                self.pipe.save()
            df = self.df[-200:]
            self.pipe.fit(df, self.target)

            # Load test
            self.pipe.save(filename=PIPE_PATH)
            self.pipe = MatPipe.load(PIPE_PATH)
            df_test = self.pipe.predict(self.df[-220:-201])
            self.assertTrue(self.target in df_test.columns)
            self.assertTrue(self.target + " predicted" in df_test.columns)

            # Version test
            self.pipe.version = "not a real version"
            self.pipe.save(VERSION_PIPE_PATH)
            with self.assertRaises(AutomatminerError):
                MatPipe.load(VERSION_PIPE_PATH)

        @unittest.skipIf("digests" in skip, reason)
        def test_summarize_and_inspect(self):
            df = self.df[-200:]
            self.pipe.fit(df, self.target)

            for ext in AMM_SUPPORTED_EXTS:
                digest = self.pipe.inspect(filename=DIGEST_PATH + ext)
                self.assertTrue(os.path.isfile(DIGEST_PATH + ext))
                self.assertTrue(isinstance(digest, dict))

            for ext in AMM_SUPPORTED_EXTS:
                digest = self.pipe.summarize(filename=DIGEST_PATH + ext)
                self.assertTrue(os.path.isfile(DIGEST_PATH + ext))
                self.assertTrue(isinstance(digest, dict))

        def _run_benchmark(self, cache, pipe):
            # Test static, regular benchmark (no fittable featurizers)
            df = self.df.iloc[500:600]
            kfold = KFold(n_splits=2)
            df_tests = pipe.benchmark(df, self.target, kfold, cache=cache)
            self.assertEqual(len(df_tests), kfold.n_splits)

            # Make sure we retain a good amount of test samples...
            df_tests_all = pd.concat(df_tests)
            self.assertGreaterEqual(len(df_tests_all), 0.95 * len(df))

            # Test static subset of kfold
            df2 = self.df.iloc[500:550]
            df_tests2 = pipe.benchmark(
                df2, self.target, kfold, fold_subset=[0], cache=cache
            )
            self.assertEqual(len(df_tests2), 1)

        def tearDown(self) -> None:
            digests = [DIGEST_PATH + ext for ext in AMM_SUPPORTED_EXTS]
            for remnant in [CACHE_SRC, PIPE_PATH, VERSION_PIPE_PATH, *digests]:
                if os.path.exists(remnant):
                    os.remove(remnant)

    return TestMatPipe


@unittest.skipIf(
    int(os.environ.get("SKIP_INTENSIVE", 0)),
    "Test too intensive for CircleCI commit builds.",
)
class MatPipeDebugTest(make_matpipe_test("debug")):
    pass


class MatPipeDebugSingleTest(make_matpipe_test("debug_single")):
    pass


if __name__ == "__main__":
    unittest.main()
