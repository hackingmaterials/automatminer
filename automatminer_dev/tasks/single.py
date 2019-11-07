import os
import warnings

from fireworks import FireTaskBase, explicit_serialize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from matminer.utils.io import load_dataframe_from_json

from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import DataCleaner, FeatureReducer
from automatminer.automl.adaptors import TPOTAdaptor, SinglePipelineAdaptor
from automatminer.pipeline import MatPipe


@explicit_serialize
class RunSingleFit(FireTaskBase):
    _fw_name = "RunSinglePipe"

    def run_task(self, fw_spec):
        # Read data from fw_spec
        pipe_config_dict = fw_spec["pipe_config"]
        target = fw_spec["target"]
        data_file = fw_spec["data_file"]
        learner_name = pipe_config_dict["learner_name"]
        learner_kwargs = pipe_config_dict["learner_kwargs"]
        reducer_kwargs = pipe_config_dict["reducer_kwargs"]
        cleaner_kwargs = pipe_config_dict["cleaner_kwargs"]
        autofeaturizer_kwargs = pipe_config_dict["autofeaturizer_kwargs"]

        # Modify data_file based on computing resource
        data_dir = os.environ["AMM_DATASET_DIR"]
        data_file = os.path.join(data_dir, data_file)

        # Modify save_dir based on computing resource
        bench_dir = os.environ["AMM_SINGLE_FIT_DIR"]
        base_save_dir = fw_spec["base_save_dir"]
        base_save_dir = os.path.join(bench_dir, base_save_dir)

        if not os.path.exists(base_save_dir):
            os.makedirs(base_save_dir)

        # Set up pipeline config
        if learner_name == "TPOTAdaptor":
            learner = TPOTAdaptor(**learner_kwargs)
        elif learner_name == "rf":
            warnings.warn(
                "Learner kwargs passed into RF regressor/classifiers bc. rf being used."
            )
            learner = SinglePipelineAdaptor(
                regressor=RandomForestRegressor(**learner_kwargs),
                classifier=RandomForestClassifier(**learner_kwargs),
            )
        else:
            raise ValueError("{} not supported yet!" "".format(learner_name))
        pipe_config = {
            "learner": learner,
            "reducer": FeatureReducer(**reducer_kwargs),
            "cleaner": DataCleaner(**cleaner_kwargs),
            "autofeaturizer": AutoFeaturizer(**autofeaturizer_kwargs),
        }
        pipe = MatPipe(**pipe_config)

        # Set up dataset
        # Dataset should already be set up correctly as json beforehand.
        # this includes targets being converted to classification, removing
        # extra columns, having the names of featurization cols set to the
        # same as the matpipe config, etc.
        df = load_dataframe_from_json(data_file)

        pipe.fit(df, target)
        pipe.save(os.path.join(base_save_dir, "pipe.p"))
