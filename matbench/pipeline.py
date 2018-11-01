"""
The highest level classes for pipelines.
"""

from matbench.base import LoggableMixin, DataframeTransformer

#todo: needs tests - alex

class MatPipe(DataframeTransformer, LoggableMixin):
    """
    Establish an ML pipeline for transforming compositions, structures,
    bandstructures, and DOS objects into machine-learned properties.

    The pipeline includes:
        - featurization
        - ml-preprocessing
        - automl model fitting and creation

    Use the pipeline by fitting it on a training dataframe using the fit method.
    Then predict the properties of other materials by passing a dataframe to the
    transform method.

    The pipeline is transferrable. So it can be fit on one dataset and used
    to predict the properties of another. In a rigorous validation experiment,
    this is how validation should be conducted, to avoid overfitting by, for
    example, running feature reduction on a mixture of validation and training
    data.

    Args:


    Attributes:
        These attributes are set during fitting. Each has their own set of
        attributes which defines more specifically how the pipeline works.

        auto_featurizer (AutoFeaturizer): The autofeaturizer object used to
            automatically decorate the dataframe with descriptors.
        data_cleaner (DataCleaner): The data cleaner object used to get a
            featurized dataframe in ml-ready form.
        feature_reducer (FeatureReducer): The feature reducer object used to
            select the best features from a "clean" dataframe.
        automl_adaptor (AutoMLAdaptor): The auto ml adaptor object used to
            actually run a auto-ml pipeline on the clean, reduced, featurized
            dataframe.
    """
    def __init__(self, logger=False):
        self._logger = self.get_logger(logger)
        self.auto_featurizer = None
        self.data_cleaner = None
        self.feature_reducer = None
        self.automl_adaptor = None

    def fit(self, df, target):
        pass

    def predict(self, df, target):
        pass
