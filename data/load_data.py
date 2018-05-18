import ast
import pandas as pd
from pymatgen import Structure
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

"""
All load* methods return the data in pandas.DataFrame
"""

def load_double_perovskites_gap(return_lumo=False):
    """
    Electronic band gaps of double perovskites calculated using ﻿Gritsenko,
    van Leeuwen, van Lenthe and Baerends potential in GPAW.

    Args:
        return_lumo (bool): whether to return the lowest unoccupied molecular
            orbital (LUMO) energy levels (in eV).

    References:
        1) https://www.nature.com/articles/srep19375
        2) CMR database: https://cmr.fysik.dtu.dk/
    """
    df = pd.read_excel('sources/double_perovskites.xlsx', sheet_name='bandgap')
    lumo = pd.read_excel('sources/double_perovskites.xlsx', sheet_name='lumo')
    if return_lumo:
        return df, lumo
    else:
        return df

def load_mp_data(filename='sources/mp_nostruct.csv'):
    """
    Loads a pregenerated csv file containing properties of ALL materials in MP
    (approximately 70k).

    If you need a version WITH structures, use generate_data.py to create
    mp_all.csv, and use filename="sources/mp_all.csv"
    """

    df = pd.read_csv(filename)
    df = df.drop("mpid", axis=1)
    if 'structure' in df.columns.values:
        df['structure'] = df['structure'].map(ast.literal_eval).map(Structure.from_dict)
    colmap = {'material_id': 'mpid',
              'pretty_formula': 'formula',
              'band_gap': 'gap pbe',
              'e_above_hull': 'ehull',
              'elasticity.K_VRH': 'bulkmod',
              'elasticity.G_VRH': 'shearmod',
              'elasticity.elastic_anisotropy': 'elastic_anisotropy',
              'total_magnetization': 'mu_B'}
    return df.rename(columns=colmap)

def load_wolverton_oxides():
    """
    Wolverton's perovskite oxides containing composition data, lattice constants,
    and formation + vacancy formation energies. There are 5,329 compounds
    in this dataset.

    From https://www.nature.com/articles/sdata2017153#ref40
    """
    df = pd.read_csv("sources/wolverton_oxides.csv")
    colmap = {"Chemical formula": "formula",
              "A": "atom A",
              "B": "atom B",
              "Formation energy [eV/atom]": "eformation",
              "Band gap [eV]": "gap pbe",
              "Magnetic moment [mu_B]": "mu_B",
              "Vacancy energy [eV/O atom]": "eformation oxygen vacancy",
              "Stability [eV/atom]": "ehull"}
    return df.rename(columns=colmap)


def load_m2ax():
    """
    An elastic dataset of 224 stable M2AX compounds from "A comprehensive survey
    of M2AX phase elastic properties" by Cover et al. Calculations are PAW
    PW91.

    From http://iopscience.iop.org/article/10.1088/0953-8984/21/30/305403/meta
    """
    df = pd.read_csv("sources/m2ax_elastic.csv")
    colmap = {"M2AX phase": "formula",
              "B": "bulkmod",
              "G": "shearmod",
              "E": "elasticmod"}
    return df.rename(columns=colmap)

if __name__ == "__main__":
    # print(load_double_perovskites_gap().head())
    print(load_wolverton_oxides())
    # load_m2ax()
    # print(load_mp_data())

