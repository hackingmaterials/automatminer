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
from automatminer.utils.log import AMM_DEFAULT_LOGLVL, AMM_DEFAULT_LOGGER


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

    if preset not in get_available_presets():
        raise ValueError("{} unknown preset.".format(preset))

    elif preset == "production":
        config = {
            "learner": TPOTAdaptor(max_time_mins=1440,
                                   max_eval_time_mins=20),
            "reducer": FeatureReducer(reducers=('corr', 'tree'),
                                      tree_importance_percentile=0.99),
            "autofeaturizer": AutoFeaturizer(preset="express",
                                             **caching_kwargs),
            "cleaner": DataCleaner()
        }
    elif preset == "heavy":
        config = {
            "learner": TPOTAdaptor(max_time_mins=2880),
            "reducer": FeatureReducer(reducers=("corr", "rebate")),
            "autofeaturizer": AutoFeaturizer(preset="heavy", **caching_kwargs),
            "cleaner": DataCleaner()
        }
    elif preset == "express":
        config = {
            "learner": TPOTAdaptor(max_time_mins=60, population_size=20),
            "reducer": FeatureReducer(reducers=('corr', 'tree'),
                                      tree_importance_percentile=0.99),
            "autofeaturizer": AutoFeaturizer(preset="express",
                                             **caching_kwargs),
            "cleaner": DataCleaner()
        }
    elif preset == "express_single":
        xgb_kwargs = {"n_estimators": 300, "max_depth": 3, "n_jobs": -1}
        config = {
            "learner": SinglePipelineAdaptor(
                regressor=XGBRegressor(**xgb_kwargs),
                classifier=XGBClassifier(**xgb_kwargs)),
            "reducer": FeatureReducer(reducers=('corr',)),
            "autofeaturizer": AutoFeaturizer(preset="express",
                                             **caching_kwargs),
            "cleaner": DataCleaner()
        }
    elif preset == "debug":
        config = {
            "learner": TPOTAdaptor(max_time_mins=2,
                                   max_eval_time_mins=1,
                                   population_size=10),
            "reducer": FeatureReducer(reducers=('corr', 'tree')),
            "autofeaturizer": AutoFeaturizer(preset="debug", **caching_kwargs),
            "cleaner": DataCleaner()
        }
    elif preset == "debug_single":
        rf_kwargs = {"n_estimators": 10, "n_jobs": -1}
        config = {
            "learner": SinglePipelineAdaptor(
                classifier=RandomForestClassifier(**rf_kwargs),
                regressor=RandomForestRegressor(**rf_kwargs)),
            "reducer": FeatureReducer(reducers=('corr',)),
            "autofeaturizer": AutoFeaturizer(preset="debug", **caching_kwargs),
            "cleaner": DataCleaner()
        }

    config["logger"] = powerups.get("logger", AMM_DEFAULT_LOGGER)
    config["log_level"] = powerups.get("log_lvl", AMM_DEFAULT_LOGLVL)
    return config


def get_available_presets():
    """
    Return all available presets for MatPipes.

    Returns:
        ([str]): A list of preset names.
    """
    return [
        "production",
        "heavy",
        "express",
        "express_single",
        "debug",
        "debug_single"
    ]