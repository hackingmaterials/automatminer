"""
The environment variables you need for this all to work are:

- AMM_BENCH_DIR: where to store benchmarks
- AMM_MODEL_DIR: where to store models (from production fits on full datasets)
- AMM_DATASET_DIR: where to store datasets
- AMM_CODE_DIR: where to run tests
"""
from fireworks import LaunchPad
from automatminer.utils.ml import AMM_CLF_NAME, AMM_REG_NAME

from hmte.db import get_connection

# Private production
LP = get_connection("hackingmaterials", write=True, connection_type="launchpad")

# Debugging locally
# LP = LaunchPad(name="automatminer")

# Constants for running benchmarks and builds
KFOLD_DEFAULT = {"shuffle": True, "random_state": 18012019, "n_splits": 5}
RUN_TESTS_CMD = "cd $AMM_CODE_DIR && coverage run setup.py test"
EXPORT_COV_CMD = "coverage xml && python-codacy-coverage -r coverage.xml"

# Local testing configuration...
LOCAL_DEBUG_REG = {
    "name": "debug_local_reg",
    "data_pickle": "jdft2d_smalldf.pickle.gz",
    "target": "exfoliation_en",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

LOCAL_DEBUG_CLF = {
    "name": "debug_local_clf",
    "data_pickle": "expt_gaps_smalldf.pickle.gz",
    "target": "is_metal",
    "problem_type": AMM_CLF_NAME,
    "clf_pos_label": True
}

LOCAL_DEBUG_SET = [LOCAL_DEBUG_CLF, LOCAL_DEBUG_REG]

# Real benchmark sets
BULK = {
    "name": "mp_bulk",
    "data_pickle": "elasticity_K_VRH.pickle.gz",
    "target": "K_VRH",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

SHEAR = {
    "name": "mp_shear",
    "data_pickle": "elasticity_G_VRH.pickle.gz",
    "target": "G_VRH",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

LOG_BULK = {
    "name": "mp_log_bulk",
    "data_pickle": "elasticity_log10(K_VRH).pickle.gz",
    "target": "log10(K_VRH)",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

LOG_SHEAR = {
    "name": "mp_log_shear",
    "data_pickle": "elasticity_log10(G_VRH).pickle.gz",
    "target": "log10(G_VRH)",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

REFRACTIVE = {
    "name": "refractive_index",
    "data_pickle": "dielectric.pickle.gz",
    "target": "n",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

JDFT2D = {
    "name": "jdft2d",
    "data_pickle": "jdft2d.pickle.gz",
    "target": "exfoliation_en",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

MP_GAP = {
    "name": "mp_gap",
    "data_pickle": "mp_gap.pickle.gz",
    "target": "gap pbe",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

MP_IS_METAL = {
    "name": "mp_is_metal",
    "data_pickle": "mp_is_metal.pickle.gz",
    "target": "is_metal",
    "problem_type": AMM_CLF_NAME,
    "clf_pos_label": True
}

MP_E_FORM = {
    "name": "mp_e_form",
    "data_pickle": "mp_e_form.pickle.gz",
    "target": "e_form",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

CASTELLI_E_FORM = {
    "name": "castelli",
    "data_pickle": "castelli.pickle.gz",
    "target": "e_form",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

GFA = {
    "name": "glass_formation",
    "data_pickle": "glass.pickle.gz",
    "target": "gfa",
    "problem_type": AMM_CLF_NAME,
    "clf_pos_label": True
}

EXPT_IS_METAL = {
    "name": "expt_is_metal",
    "data_pickle": "expt_is_metal.pickle.gz",
    "target": "is_metal",
    "problem_type": AMM_CLF_NAME,
    "clf_pos_label": True
}

EXPT_GAP = {
    "name": "expt_gap",
    "data_pickle": "expt_gap.pickle.gz",
    "target": "gap expt",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

PHONONS = {
    "name": "phonons",
    "data_pickle": "phonons.pickle.gz",
    "target": "last phdos peak",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

STEELS_YIELD = {
    "name": "steels_yield",
    "data_pickle": "steels_yield.pickle.gz",
    "target": "yield strength",
    "problem_type": AMM_REG_NAME,
    "clf_pos_label": None
}

BENCHMARK_DEBUG_SET = [JDFT2D, PHONONS, EXPT_IS_METAL, STEELS_YIELD]
BENCHMARK_FULL_SET = [BULK, SHEAR, LOG_BULK, LOG_SHEAR, REFRACTIVE, JDFT2D,
                      MP_GAP, MP_IS_METAL, MP_E_FORM, CASTELLI_E_FORM, GFA,
                      EXPT_IS_METAL, EXPT_GAP, STEELS_YIELD, PHONONS]