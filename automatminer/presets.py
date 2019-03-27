"""
Various configurations for MatPipe.

Use them like so:

    config = get_preset_config()
    pipe = MatPipe(**config)
"""

__author__ = ["Alex Dunn <ardunn@lbl.gov>"]

from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import FeatureReducer, DataCleaner
from automatminer.automl import TPOTAdaptor, SinglePipelineAdaptor


def get_preset_config(preset: str = 'express', **powerups) -> dict:
    """
    Preset configs for MatPipe.

    USER:
    "production": Used for making production predictions and benchmarks.
        Balances accuracy and timeliness.
    "heavy" - When high accuracy is required, and you have access to
        (very) powerful computing resources. May be buggier and more difficult
        to run than production.
    "express" - Good for quick benchmarks with moderate accuracy.
    "express_single" - Same as express but uses XGB trees as single models
        instead of automl TPOT. Good for even more express results.

    DEBUG:
    "debug" - Debugging with automl enabled.
    "debug_single" - Debugging with a single model.

    Args:
        preset (str): The name of the preset config you'd like to use.
        **powerups: Various modifications as kwargs.
            cache_src (str): A file path. If specified, Autofeaturizer will use
                feature caching with a file stored at this location. See
                Autofeaturizer's cache_src argument for more information.
    Returns:
        (dict) The desired preset config.

    """
    caching_kwargs = {"cache_src": powerups.get("cache_src", None)}

    if preset == "production":
        production_config = {
            "learner": TPOTAdaptor(max_time_mins=720,
                                   max_eval_time_mins=20),
            "reducer": FeatureReducer(reducers=('pca',)),
            "autofeaturizer": AutoFeaturizer(preset="best", **caching_kwargs),
            "cleaner": DataCleaner()
        }
        return production_config
    elif preset == "heavy":
        heavy_config = {
            "learner": TPOTAdaptor(max_time_mins=1440),
            "reducer": FeatureReducer(reducers=("corr", "rebate")),
            "autofeaturizer": AutoFeaturizer(preset="all", **caching_kwargs),
            "cleaner": DataCleaner()
        }
        return heavy_config
    elif preset == "express":
        express_config = {
            "learner": TPOTAdaptor(max_time_mins=60, population_size=20),
            "reducer": FeatureReducer(reducers=('corr',)),
            "autofeaturizer": AutoFeaturizer(preset="fast", **caching_kwargs),
            "cleaner": DataCleaner()
        }
        return express_config
    elif preset == "express_single":
        xgb_kwargs = {"n_estimators": 300, "max_depth": 3, "n_jobs": -1}
        express_config = {
            "learner": SinglePipelineAdaptor(
                regressor=XGBRegressor(**xgb_kwargs),
                classifier=XGBClassifier(**xgb_kwargs)),
            "reducer": FeatureReducer(reducers=('corr',)),
            "autofeaturizer": AutoFeaturizer(preset="fast", **caching_kwargs),
            "cleaner": DataCleaner()
        }
        return express_config
    elif preset == "debug":
        debug_config = {
            "learner": TPOTAdaptor(max_time_mins=2,
                                   max_eval_time_mins=1,
                                   population_size=10),
            "reducer": FeatureReducer(reducers=('corr', 'tree')),
            "autofeaturizer": AutoFeaturizer(preset="fast", **caching_kwargs),
            "cleaner": DataCleaner()
        }
        return debug_config
    elif preset == "debug_single":
        rf_kwargs = {"n_estimators": 10, "n_jobs": -1}
        debug_single_config = {
            "learner": SinglePipelineAdaptor(
                classifier=RandomForestClassifier(**rf_kwargs),
                regressor=RandomForestRegressor(**rf_kwargs)),
            "reducer": FeatureReducer(reducers=('corr',)),
            "autofeaturizer": AutoFeaturizer(preset="fast", **caching_kwargs),
            "cleaner": DataCleaner()
        }
        return debug_single_config
    else:
        raise ValueError("{} unknown preset.".format(preset))
