"""
The highest level classes for pipelines.
"""
import numpy as np

from matbench.base import LoggableMixin, DataframeTransformer
from matbench.featurization import AutoFeaturizer
from matbench.preprocessing import DataCleaner, FeatureReducer
from matbench.automl.adaptors import TPOTAdaptor
from matbench.utils.utils import regression_or_classification, check_fitted, \
    set_fitted

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
    Then predict the properties of new materials by passing a dataframe to the
    transform method.

    The pipeline is transferrable. So it can be fit on one dataset and used
    to predict the properties of another. In a rigorous validation experiment,
    this is how validation can be conducted, to avoid overfitting by, for
    example, running feature reduction on a mixture of validation and training
    data.

    Examples:
        pipe = MatPipe()                                          # make a pipe
        pipe.fit(training_df, "target_property")                  # fit it (can be used for benchmarking)
        predictions = pipe.predict(other_df, "target_property")   # use it to predict properties
        pipe.to("json")                                           # save how the pipe was constructed

    Args:
        persistence_lvl (int): Persistence level of 0 saves nothing. 1 saves
            intermediate dataframes and final dataframes. 2 saves all dataframes
            and all objects used to create the pipeline, and auto-saves a digest
        time_limit_mins (int): The approximate time limit, in minutes.
        # todo: add the new attrs


    Attributes:

        The following attributes are set during fitting. Each has their own set
        of attributes which defines more specifically how the pipeline works.

        autofeater (AutoFeaturizer): The autofeaturizer object used to
            automatically decorate the dataframe with descriptors.
        cleaner (DataCleaner): The data cleaner object used to get a
            featurized dataframe in ml-ready form.
        reducer (FeatureReducer): The feature reducer object used to
            select the best features from a "clean" dataframe.
        learner (AutoMLAdaptor): The auto ml adaptor object used to
            actually run a auto-ml pipeline on the clean, reduced, featurized
            dataframe.
        is_fit (bool): If True, the matpipe is fit. The matpipe should be
            fit before being used to predict data.
    """
    def __init__(self, persistence_lvl=2, logger=True, time_limit_mins=5):
        self._logger = self.get_logger(logger)
        self.time_limit_mins = time_limit_mins
        self.persistence_lvl = persistence_lvl
        self.autofeater = None
        self.cleaner = None
        self.reducer = None
        self.learner = None
        self.pre_fit_df = None
        self.post_fit_df = None
        self.is_fit = False
        self.ml_type = None
        self.common_kwargs = {"logger": self.logger}

        #todo: implement persistence level
        #todo: implement scoring selection

    @set_fitted
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
        self.pre_fit_df = df
        self.ml_type = regression_or_classification(df[target])

        # Get top-lvel transformers
        self.autofeater = AutoFeaturizer(**self.common_kwargs)
        self.cleaner = DataCleaner(**self.common_kwargs)
        self.reducer = FeatureReducer(**self.common_kwargs)
        self.learner = TPOTAdaptor(self.ml_type,
                                   **self.common_kwargs,
                                   max_time_mins=self.time_limit_mins)

        # Fit transformers on training data
        self.logger.info("Fitting MatPipe pipeline to data.")
        df = self.autofeater.fit_transform(df, target)
        df = self.cleaner.fit_transform(df, target)
        df = self.reducer.fit_transform(df, target)
        self.learner.fit(df, target)
        self.logger.info("MatPipe successfully fit.")
        self.post_fit_df = df
        return self

    @check_fitted
    def predict(self, df, target):
        """
        Predict a target property of a set of materials.

        The dataframe should have the same target property as the dataframe
        used for fitting. The dataframe should also have the same materials
        property types at the dataframe used for fitting (e.g., if you fit a
        matpipe to a df containing composition, your prediction df should have
        a column for composition).

        Args:
            df (pandas.DataFrame): Pipe will be fit to this dataframe.
            target (str): The column in the dataframe containing the target
                property of interest

        Returns:
            (pandas.DataFrame): The dataframe with target property predictions.
        """
        self.logger.info("Beginning MatPipe prediction using fitted pipeline.")
        df = self.autofeater.transform(df, target)
        df = self.cleaner.transform(df, target)
        df = self.reducer.transform(df, target)
        predictions = self.learner.predict(df, target)
        self.logger.info("MatPipe prediction completed.")
        return predictions

    def benchmark(self, df, target, validation_fraction=0.2):
        """

        If the target property is known for all data, perform an ML benchmark
        using MatPipe. Used for getting an idea of how well AutoML can predict
        a certain target property.

        This method featurizes and cleans the entire dataframe, then splits
        the data for training and testing. FeatureReducer and TPOT models are
        fit on the training data. Finally, these fitted models are used to
        predict the properties of the test df. This scheme allows for rigorous
        ML model evaluation, as the feature selection and model fitting are
        determined without any knowledge of the validation/test set.

        To use a random validation set for model validation, pass in a nonzero
        validation fraction. The returned df will have the validation
        predictions.

        To use a CV-only validation, use a validation frac. of 0. The original
        df will be returned having predictions made on all training data. This
        should ONLY be used to evaluate the training error!

        Whether using CV-only or validation, both will create CV information
        in the MatPipe.learner.best_models variable.

        Args:
            df (pandas.DataFrame): The dataframe for benchmarking. Must contain
            target (str): The column name to use as the ml target property.
            validation_fraction (float): Value between 0 and 1 indicating
                the validation fraction desired. If set to 0, a cross-validation
                only benchmarking is performed.

        Returns:
            testdf (pandas.DataFrame): A dataframe containing validation data
                and predicted data. If validation_fraction is set to 0, test df
                will contain PREDICTIONS MADE ON TRAINING DATA. This should be
                used to evaluate the training error only.

        """
        self.ml_type = regression_or_classification(df[target])

        # Get top-lvel transformers
        self.autofeater = AutoFeaturizer(**self.common_kwargs)
        self.cleaner = DataCleaner(**self.common_kwargs)
        self.reducer = FeatureReducer(**self.common_kwargs)
        self.learner = TPOTAdaptor(self.ml_type,
                                   max_time_mins=self.time_limit_mins,
                                   **self.common_kwargs)

        # Fit transformers on all data
        self.logger.info("Featurizing and cleaning {} samples from the entire"
                         " dataframe.".format(df.shape[0]))
        df = self.autofeater.fit_transform(df, target)
        df = self.cleaner.fit_transform(df, target)

        # Split data for steps where combined transform could otherwise over-fit
        # or leak data from validation set into training set.
        msk = np.random.rand(len(df)) < validation_fraction
        traindf = df[~msk]
        testdf = df[msk]
        self.logger.info("Dataframe split into training and testing fractions"
                         " having {} and {} samples.".format(traindf.shape[0],
                                                             testdf.shape[1]))

        # Use transformers on separate training and testing dfs
        self.logger.info("Perforing feature reduction and model selection on "
                         "the {}-sample training set.".format(traindf.shape[0]))
        traindf = self.reducer.fit_transform(traindf, target)
        self.learner.fit(traindf, target)


        if validation_fraction != 0:
            self.logger.info(
                "Using pipe fitted on training data to predict target {} on "
                "{}-sample validation dataset".format(target, testdf.shape[0]))
            testdf = self.reducer.transform(testdf, target)
            testdf = self.learner.predict(testdf, target)
            return testdf
        else:
            self.logger.warning("Validation fraction set to zero. Using "
                                "cross-validation-only benchmarking...")

    @check_fitted
    def digest(self, filename, fmt="json"):
        """
        Save a text digest (summary) of the fitted pipeline. Similar to the log
        but contains more detail in a structured format.

        Args:
            filename (str): The filename.
            fmt (str): The format to save the pipeline in. Valid choices are
                "json", "txt".

        Returns:
            None
        """
        pass


if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    from matminer.datasets.dataset_retrieval import load_dataset
    hugedf = load_dataset("elastic_tensor_2015").rename(columns={"formula": "composition"})[["composition",  "K_VRH"]]
    df = hugedf.iloc[:100]
    df2 = hugedf.iloc[101:150]
    target = "K_VRH"

    # mp = MatPipe()
    # mp.fit(df, target)
    # print(mp.predict(df2, target))

    mp = MatPipe()
    df = mp.benchmark(df, target, validation_fraction=0.2)
    print(df)
    print("Validation error is {}".format(mean_squared_error(df[target], df[target + " predicted"])))
    #
    # mp = MatPipe()
    # df = mp.benchmark(df, target, validation_fraction=0)
    # print(df)
    # print("CV scores: {}".format(mp.learner.best_scores))
