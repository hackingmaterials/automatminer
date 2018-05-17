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
    df = pd.read_excel('data_files/double_perovskites.xlsx', sheet_name='bandgap')
    lumo = pd.read_excel('data_files/double_perovskites.xlsx', sheet_name='lumo')
    if return_lumo:
        return df, lumo
    else:
        return df

def load_mp_data(filename='data_files/mp_nostruct.csv'):
    """
    Loads a pregenerated csv file containing properties of ALL materials in MP
    (approximately 70k).

    If you need a version WITH structures, use generate_data.py to create
    mp_all.csv, and use filename="data_files/mp_all.csv"
    """

    df = pd.read_csv(filename)
    df = df.drop("mpid", axis=1)
    if 'structure' in df.columns.values:
        df['structure'] = df['structure'].map(ast.literal_eval).map(Structure.from_dict)
    df = df.rename(columns={'material_id': 'mpid', 'pretty_formula': 'formula',
                            'band_gap': 'gap pbe', 'e_above_hull': 'ehull',
                            'elasticity.K_VRH': 'bulkmod',
                            'elasticity.G_VRH': 'shearmod',
                            'elasticity.elastic_anisotropy': 'elastic_anisotropy'})
    return df

if __name__ == "__main__":
    # print(load_double_perovskites_gap().head())
    print(load_mp_data())
