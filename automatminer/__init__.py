from automatminer.preprocessing import DataCleaner, FeatureReducer
from automatminer.automl import TPOTAdaptor
from automatminer.pipeline import MatPipe
from automatminer.configs import get_debug_config, get_default_config, \
    get_production_config, get_fast_config

__author__ = 'Alex Dunn, Qi Wang, Alex Ganose, Daniel Dopp, Anubhav Jain'
__author_email__ = 'ardunn@lbl.gov'
__license__ = 'Modified BSD'
__version__ = '2018.12.11_beta'