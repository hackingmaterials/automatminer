"""
Various configurations for MatPipe.

Use them like so:

    config = get_preset_config()
    pipe = MatPipe(**config)
"""

__author__ = ["Alex Dunn <ardunn@lbl.gov>"]

from sklearn.ensemble import RandomForestRegressor

from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import FeatureReducer, DataCleaner
from automatminer.automl import TPOTAdaptor, SinglePipelineAdaptor


def get_preset_config(preset='default'):
    """
    Returns preset configs for MatPipe.

    "default" - Good for typical usage
    "production" - When high accuracy is required, and you have access to
        powerful computing resources
    "fast" - Good for quick benchmarks
    "debug" - Debugging with automl enabled
    "debug_single" - Debugging with a single model in an adaptor.

    Args:
        preset (str): The name of the preset config you'd like to use.

    Returns:
        (dict) The desired preset config.

    """
    production_config = {
        "learner": TPOTAdaptor(population_size=250,
                               max_time_mins=720,
                               max_eval_time_mins=30),
        "reducer": FeatureReducer(),
        "autofeaturizer": AutoFeaturizer(preset="best"),
        "cleaner": DataCleaner()
    }

    default_config = {
        "learner": TPOTAdaptor(max_time_mins=120),
        "reducer": FeatureReducer(),
        "autofeaturizer": AutoFeaturizer(preset="best"),
        "cleaner": DataCleaner()
    }

    fast_config = {
        "learner": TPOTAdaptor(max_time_mins=30, population_size=50),
        "reducer": FeatureReducer(reducers=('corr', 'tree')),
        "autofeaturizer": AutoFeaturizer(preset="fast"),
        "cleaner": DataCleaner()
    }

    debug_config = {
        "learner": TPOTAdaptor(max_time_mins=2,
                               max_eval_time_mins=1,
                               population_size=10),
        "reducer": FeatureReducer(reducers=('corr',)),
        "autofeaturizer": AutoFeaturizer(preset="fast"),
        "cleaner": DataCleaner()
    }

    debug_single_config = {
        "learner": SinglePipelineAdaptor(
            model=RandomForestRegressor(n_estimators=10)),
        "reducer": FeatureReducer(reducers=('corr',)),
        "autofeaturizer": AutoFeaturizer(preset="fast"),
        "cleaner": DataCleaner()
    }

    if preset == "default":
        return default_config
    elif preset == "fast":
        return fast_config
    elif preset == "debug":
        return debug_config
    elif preset == "debug_single":
        return debug_single_config
    elif preset == "production":
        return production_config
    else:
        raise ValueError("{} unknown preset.".format(preset))