import unittest
import os.path
import warnings

from automatminer.analytics.core import Analytics
from automatminer.presets import get_preset_config
from automatminer.pipeline import MatPipe
from matminer.datasets import load_dataset
try:
    #from skater.core.explanations import Interpretation
    #from skater.model import InMemoryModel
    warnings.warn("")
except ImportError:
    warnings.warn("skater package not found. Please install skater to use the "
                   "Analytics modeule/")
    Interpretation = None
    InMemoryModel = None


@unittest.skipIf("CI" in os.environ.keys(), "Test too intensive for CircleCI.")
class TestAnalytics(unittest.TestCase):

    def setUp(self):
        df = load_dataset('elastic_tensor_2015')
        df = df[["formula", "K_VRH"]]
        df = df.rename({"formula": "composition"}, axis=1)
        self.config = get_preset_config("debug")
        fitted_pipeline = MatPipe(**self.config).fit(df, "K_VRH")

        self.analytics = Analytics(predictive_model=fitted_pipeline)


    def test_get_feature_importance(self):
        feature_importance = self.analytics.get_feature_importance()
        print(feature_importance)
        self.assertTrue(not feature_importance.empty)


    def test_plot_partial_dependence(self):
        feature_importance = self.analytics.get_feature_importance()
        for feature in feature_importance.index[len(feature_importance.index)-15:]:
            self.analytics.plot_partial_dependence(feature, save_plot=True, show_plot=False)
        self.assertTrue(os.path.isfile("t_pdp.png"))
        self.assertTrue(os.path.isfile("m_pdp.png"))
