# """
# Tests for the top level interface.
# """
#
# import unittest
# import os.path
#
# from matminer.datasets.dataset_retrieval import load_dataset
# from sklearn.metrics import r2_score
# from sklearn.exceptions import NotFittedError
#
# from mslearn.pipeline import MatPipe, debug_config
#
# test_dir = os.path.dirname(__file__)

#
# class TestMatPipe(unittest.TestCase):
#     def setUp(self):
#         df = load_dataset("elastic_tensor_2015").rename(
#             columns={"formula": "composition"})
#         self.df = df[["composition", "K_VRH"]]
#         self.extra_features = df["G_VRH"]
#         self.target = "K_VRH"
#
#     def test_transferability(self):
#         df_train = self.df.iloc[:200]
#         df_test = self.df.iloc[201:250]
#         pipe = MatPipe(**debug_config)
#         pipe.fit(df_train, self.target)
#         df_test = pipe.predict(df_test, self.target)
#         true = df_test[self.target]
#         test = df_test[self.target + " predicted"]
#         self.assertTrue("composition" not in df_test.columns)
#         self.assertTrue(r2_score(true, test) > 0.5)
#
#         # Use the same pipe object by refitting and reusing
#         df_train2 = self.df.iloc[250:450]
#         df_test2 = self.df.iloc[451:500]
#         pipe.fit(df_train2, self.target)
#         df_test2 = pipe.predict(df_test2, self.target)
#         true2 = df_test2[self.target]
#         test2 = df_test2[self.target + " predicted"]
#         self.assertTrue("composition" not in df_test2.columns)
#         self.assertTrue(r2_score(true2, test2) > 0.5)
#
#     def test_user_features(self):
#         pipe = MatPipe(**debug_config)
#         df = self.df
#         df["G_VRH"] = self.extra_features
#         self.assertTrue("G_VRH" in df.columns)
#         self.assertTrue("K_VRH" in df.columns)
#         df_train = df.iloc[:200]
#         df_test = df.iloc[201:250]
#         pipe.fit(df_train, self.target)
#
#         # If shear modulus is included as a feature it should probably show up
#         # in the final pipeline
#         self.assertTrue("G_VRH" in pipe.learner.features)
#         df_test = pipe.predict(df_test, self.target)
#         true = df_test[self.target]
#         test = df_test[self.target + " predicted"]
#         self.assertTrue(r2_score(true, test) > 0.75)
#
#     def test_benchmarking(self):
#         pipe = MatPipe(**debug_config)
#         df = self.df[500:700]
#         df_test = pipe.benchmark(df, self.target, test_spec=0.2)
#         self.assertTrue(df_test.shape[0] > 35)
#         self.assertTrue(df_test.shape[0] < 45)
#         true = df_test[self.target]
#         test = df_test[self.target + " predicted"]
#         self.assertTrue(r2_score(true, test) > 0.5)
#
#     def test_persistence_and_digest(self):
#         pipe = MatPipe(**debug_config)
#         with self.assertRaises(NotFittedError):
#             pipe.save()
#         df = self.df[-200:]
#         pipe.fit(df, self.target)
#
#         filename = os.path.join(test_dir, "test_pipe.p")
#         pipe.save(filename=filename)
#         pipe = MatPipe.load(filename, logger=False)
#         df_test = pipe.predict(self.df[-220:-201], self.target)
#         self.assertTrue(self.target in df_test.columns)
#         self.assertTrue(self.target + " predicted" in df_test.columns)
#
#         digest_file = os.path.join(test_dir, "matdigest.txt")
#         digest = pipe.digest(filename=digest_file)
#         self.assertTrue(os.path.isfile(digest_file))
#         self.assertTrue(isinstance(digest, str))



