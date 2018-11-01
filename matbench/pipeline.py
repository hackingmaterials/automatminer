"""
The highest level classes for pipelines.
"""

from matbench.base import LoggableMixin, DataframeTransformer

#todo: needs tests - alex
#todo: tests should include using custom (user speficied) features as well

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

    Examples:
        pipe = MatPipe()                                          # make a pipe
        pipe.fit(training_df, "target_property")                  # fit it (can be used for benchmarking)
        predictions = pipe.predict(other_df, "target_property")   # use it to predict properties
        pipe.to("json")                                           # save how the pipe was constructed

    Args:
        persistence_level (int): Persistence level of 0 saves nothing. 1 saves
            intermediate dataframes and final dataframes. 2 saves all dataframes
            and all objects used to create the pipeline, and auto-saves a digest.
        time_limit_mins (int): The approximate time limit, in minutes.


    Attributes:

        The following attributes are set during fitting. Each has their own set
        of attributes which defines more specifically how the pipeline works.

        auto_featurizer (AutoFeaturizer): The autofeaturizer object used to
            automatically decorate the dataframe with descriptors.
        data_cleaner (DataCleaner): The data cleaner object used to get a
            featurized dataframe in ml-ready form.
        feature_reducer (FeatureReducer): The feature reducer object used to
            select the best features from a "clean" dataframe.
        automl_adaptor (AutoMLAdaptor): The auto ml adaptor object used to
            actually run a auto-ml pipeline on the clean, reduced, featurized
            dataframe.
        is_fit (bool): If True, the matpipe is fit. The matpipe should be
            fit before being used to predict data.
    """
    def __init__(self, persistence_lvl=2, logger=True, time_limit_mins=600):
        self._logger = self.get_logger(logger)
        self.time_limit = time_limit_mins
        self.persistence_level = persistence_lvl
        self.auto_featurizer = None
        self.data_cleaner = None
        self.feature_reducer = None
        self.automl_adaptor = None
        self.is_fit = False

    def fit(self, df, target):
        """
        Fit a matpipe to a dataframe. Once fit, can be used to predict out of
        sample data.

        The dataframe should contain columns having some materials data:
            - compositions
            - structures
            - bandstructures
            - density of states
            - user-defined features

        Any combination of these data is ok.

        Args:
            df (pandas.DataFrame): Pipe will be fit to this dataframe.
            target (str): The column in the dataframe containing the target
                property of interest

        Returns:
            MatPipe (self)

        """
        self.is_fit = False
        self.is_fit = True
        return self

    def predict(self, df, target):
        """
        Predict a target property of a set of materials.

        The dataframe should have the same target property as the dataframe
        used for fitting. The dataframe should also have the same materials
        property types at the dataframe used for fitting.

        Args:
            df (pandas.DataFrame): Pipe will be fit to this dataframe.
            target (str): The column in the dataframe containing the target property of interest

        Returns:
            (pandas.DataFrame): The dataframe with target property predictions.
        """
        pass

    def digest(self, filename, fmt="json"):
        """
        Save a serializd digest (summary) of the fitted pipeline.

        Args:
            filename (str): The filename.
            fmt (str): The format to save the pipeline in. Valid choices are
                "json", "txt".

        Returns:
            None
        """
        pass
