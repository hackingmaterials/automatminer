"""
Various configurations for MatPipe.

Config dictionaries are placed in getter functions because import
statements in Python specifically import a file AND execute it. When
these variables were imported, they produced unnecessary logs. By
placing the variables in functions, the empty log issue was resolved.
Make sure that the function call does not occur unconditionally in
the global frame because that will reproduce the empty log issue.
Further information: https://www.benkuhn.net/importtime

Use them like so:

from automatminer.configs import get_default_config

...

default_config = get_default_config()

pipe = MatPipe(**default_config)

"""

from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import FeatureReducer, DataCleaner
from automatminer.automl import TPOTAdaptor


def get_production_config():
    production_config = {"learner": TPOTAdaptor(generations=500,
                                            population_size=500,
                                            max_time_mins=720,
                                            max_eval_time=60),
                     "reducer": FeatureReducer(),
                     "autofeaturizer": AutoFeaturizer(preset="best"),
                     "cleaner": DataCleaner()
                      }
    return production_config


def get_default_config():
    default_config = {"learner": TPOTAdaptor(max_time_mins=120),
                  "reducer": FeatureReducer(),
                  "autofeaturizer": AutoFeaturizer(preset="best"),
                  "cleaner": DataCleaner()}
    return default_config


def get_fast_config():
    fast_config = {"learner": TPOTAdaptor(max_time_mins=30, population_size=50),
               "reducer": FeatureReducer(reducers=('corr', 'tree')),
               "autofeaturizer": AutoFeaturizer(preset="fast"),
               "cleaner": DataCleaner()}
    return fast_config


def get_debug_config():
    debug_config = {"learner": TPOTAdaptor(max_time_mins=2,
                                       max_eval_time=1,
                                       population_size=10),
                "reducer": FeatureReducer(reducers=('corr',)),
                "autofeaturizer": AutoFeaturizer(preset="fast"),
                "cleaner": DataCleaner()}
    return debug_config