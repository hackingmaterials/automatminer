import unittest

from automatminer.pipeline import MatPipe
from automatminer.presets import get_preset_config
from matminer.datasets.dataset_retrieval import load_dataset
from sklearn.metrics.regression import mean_squared_error

class MSE_Example(unittest.TestCase):

    """
    The following example uses the elastic_tensor_2015 dataset and a
    default config to create a MatPipe. This MatPipe is used to benchmark
    the target property K_VRH.

    The unit tests confirm that the output of the benchmark is not empty.
    They also ensure that, based on this specific example, the mean
    squared error is between 0 and 500.

    For debugging purposes, you can use the debug config instead. In
    addition, make the range of the mean squared error be 0 - 1000 rather
    than 0 - 500.

    """

    def test_mse_example(self):
        df = load_dataset("elastic_tensor_2015")
        default_config = get_preset_config("default")
        pipe = MatPipe(**default_config)
        df = df.rename(columns={"formula": "composition"})[["composition", "structure", "K_VRH"]]
        predicted = pipe.benchmark(df, "K_VRH", test_spec=0.2)
        self.assertTrue(not predicted.empty)

        y_true = predicted["K_VRH"]
        y_test = predicted["K_VRH predicted"]
        mse = mean_squared_error(y_true, y_test)
        print("MSE: " + str(mse))
        self.assertTrue(mse < 500)
        self.assertTrue(mse > 0)


if __name__ == '__main__':
    unittest.main()