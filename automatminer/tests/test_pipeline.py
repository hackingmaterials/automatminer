"""
Tests for the top level interface.
"""

import unittest
import os.path

from matminer.datasets.dataset_retrieval import load_dataset
from sklearn.metrics import r2_score
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import KFold

from automatminer.pipeline import MatPipe
from automatminer.presets import get_preset_config

test_dir = os.path.dirname(__file__)


@unittest.skipIf("CI" in os.environ.keys(), "Test too intensive for CircleCI.")
class TestMatPipe(unittest.TestCase):
    def setUp(self):
        df = load_dataset("elastic_tensor_2015").rename(
            columns={"formula": "composition"})
        self.df = df[["composition", "K_VRH"]]
        self.extra_features = df["G_VRH"]
        self.target = "K_VRH"
        self.config = get_preset_config("debug")
        self.pipe = MatPipe(**self.config)

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

    def test_benchmarking(self):
        df = self.df.iloc[500:700]
        kfold = KFold(n_splits=5)
        df_tests = self.pipe.benchmark(df, self.target, kfold)
        self.assertEqual(len(df_tests), 5)

    def test_persistence_and_digest(self):
        with self.assertRaises(NotFittedError):
            self.pipe.save()
        df = self.df[-200:]
        self.pipe.fit(df, self.target)

        filename = os.path.join(test_dir, "test_pipe.p")
        self.pipe.save(filename=filename)
        self.pipe = MatPipe.load(filename, logger=False)
        df_test = self.pipe.predict(self.df[-220:-201], self.target)
        self.assertTrue(self.target in df_test.columns)
        self.assertTrue(self.target + " predicted" in df_test.columns)

        digest_file = os.path.join(test_dir, "matdigest.txt")
        digest = self.pipe.digest(filename=digest_file)
        self.assertTrue(os.path.isfile(digest_file))
        self.assertTrue(isinstance(digest, str))



