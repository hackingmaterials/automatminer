from skater.core.explanations import Interpretation
from skater.model import InMemoryModel
from matminer.datasets import load_dataset

from mslearn.pipeline import MatPipe


class Analytics:
    def __init__(self, predictive_model, feature_labels=None, dataset=None):
        if isinstance(predictive_model, MatPipe):
            self.predictive_model = predictive_model
            # self.feature_labels = predictive_model.feature_labels
            # self.dataset = predictive_model.dataset
        if feature_labels is not None:
            self.feature_labels = feature_labels
        if dataset is not None:
            self.dataset = dataset

        self.interpreter = Interpretation()
        self.interpreter.load_data(self.dataset,
                                   feature_names=self.feature_labels)
        self.model = InMemoryModel(self.predictive_model.predict,
                                   examples=self.dataset)

    def get_feature_importance(self):
        return self.interpreter.feature_importance.feature_importance(self.model)


if __name__ == '__main__':
    df = load_dataset('elastic_tensor_2015')
    df = df[["formula", "K_VRH"]]
    df = df.rename({"formula": "composition"}, axis=1)

    fitted_pipeline = MatPipe(time_limit_mins=5).fit(df, "K_VRH")
    print("Done fitting")

    analyzer = Analytics(fitted_pipeline)

    print(analyzer.get_feature_importance())
