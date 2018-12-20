from automatminer.preprocessing import DataCleaner, FeatureReducer
from automatminer.automl import TPOTAdaptor
from automatminer.pipeline import MatPipe
from automatminer.configs import debug_config, default_config, \
    production_config, fast_config

__author__ = 'Alex Dunn, Qi Wang, Alex Ganose, Daniel Dopp, Anubhav Jain'
__author_email__ = 'ardunn@lbl.gov'
__license__ = 'Modified BSD'
__version__ = '2018.12.11_beta'