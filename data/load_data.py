import pandas as pd

"""
All load* methods load the data in pandas.DataFrame
"""

def load_double_perovskites_gap(return_lumos=False):
    """
    1325??? electronic band gaps (in eV) calculated via ??? in GPAW. This data is used
    in the following references

    Args:
        return_lumos (bool): whether to return the lowest unoccupied molecular
            orbital (LUMO) energy levels (in eV).
            
    References:
        1) https://www.nature.com/articles/srep19375
        2) CMR database: https://cmr.fysik.dtu.dk/
    """
    df = pd.read_excel('data_files/double_perovskites.xlsx', sheet_name='gaps')
    lumos = pd.read_excel('data_files/double_perovskites.xlsx', sheet_name='lumos')
    if return_lumos:
        return df, lumos
    else:
        return df