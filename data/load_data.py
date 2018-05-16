import pandas as pd

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








if __name__ == "__main__":
    print(load_double_perovskites_gap().head())