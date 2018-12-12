"""
Various configurations for MatPipe.

Use them like so:

pipe = MatPipe(**default_config)
"""

from automatminer.featurization import AutoFeaturizer
from automatminer.preprocessing import FeatureReducer, DataCleaner
from automatminer.automl import TPOTAdaptor

production_config = {"learner": TPOTAdaptor(max_time_mins=480),
                     "reducer": FeatureReducer(),
                     "autofeaturizer": AutoFeaturizer(preset="best"),
                     "cleaner": DataCleaner
                      }
default_config = {"learner": TPOTAdaptor(max_time_mins=120),
                  "reducer": FeatureReducer(),
                  "autofeaturizer": AutoFeaturizer(preset="best"),
                  "cleaner": DataCleaner()}

fast_config = {"learner": TPOTAdaptor(max_time_mins=30, population_size=50),
               "reducer": FeatureReducer(reducers=('corr', 'tree')),
               "autofeaturizer": AutoFeaturizer(preset="fast"),
               "cleaner": DataCleaner()}

debug_config = {"learner": TPOTAdaptor(max_time_mins=1, population_size=10),
                "reducer": FeatureReducer(reducers=('corr',)),
                "autofeaturizer": AutoFeaturizer(preset="fast"),
                "cleaner": DataCleaner()}