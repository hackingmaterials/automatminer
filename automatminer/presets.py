"""
Various configurations for MatPipe.

Use them like so:

    config = get_preset_config()
    pipe = MatPipe(**config)
"""

__author__ = ["Alex Dunn <ardunn@lbl.gov>",
              "Abhinav Ashar <AbhinavAshar@lbl.gov"]

from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import FeatureReducer, DataCleaner
from automatminer.automl import TPOTAdaptor


def get_preset_config(preset='default'):
    production_config = {"learner": TPOTAdaptor(generations=500,
                                                population_size=500,
                                                max_time_mins=720,
                                                max_eval_time_mins=60),
                         "reducer": FeatureReducer(),
                         "autofeaturizer": AutoFeaturizer(preset="best"),
                         "cleaner": DataCleaner()
                         }
    default_config = {"learner": TPOTAdaptor(max_time_mins=120),
                      "reducer": FeatureReducer(),
                      "autofeaturizer": AutoFeaturizer(preset="best"),
                      "cleaner": DataCleaner()}

    fast_config = {"learner": TPOTAdaptor(max_time_mins=30, population_size=50),
                   "reducer": FeatureReducer(reducers=('corr', 'tree')),
                   "autofeaturizer": AutoFeaturizer(preset="fast"),
                   "cleaner": DataCleaner()}

    debug_config = {"learner": TPOTAdaptor(max_time_mins=2,
                                           max_eval_time_mins=1,
                                           population_size=10),
                    "reducer": FeatureReducer(reducers=('corr',)),
                    "autofeaturizer": AutoFeaturizer(preset="fast"),
                    "cleaner": DataCleaner()}
    if preset == "default":
        return default_config
    elif preset == "fast":
        return fast_config
    elif preset == "debug":
        return debug_config
    elif preset == "production":
        return production_config
