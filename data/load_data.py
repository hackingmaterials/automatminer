import os
import ast
import pandas as pd
import numpy as np
from pymatgen import Structure


"""
All load* methods return the data in pandas.DataFrame. In each method a raw
data file is loaded, some preliminary transformation/renaming/cleaning done and
the results are returned. For specific columns returned refer to the documentation
of each function. If you plan to add a new dataset please follow the guidelines
and refer to documentation in load_castelli_perovskites for consistent docs.

Naming convention guidelines:
    - use small letters for column names consistently
    - return only those columns that cannot be derived from other available ones
    - use spaces between words; use _ only when it makes sense as a subscript
        e.g. "e_form" means energy of formation
    - start with property name followed by method/additional description: 
        e.g. "gap expt" means band gap measured experimentally
        e.g. "gap pbe" means band gap calculated via DFT using PBE functional
    - avoid including units in the column name, instead explain in the docs
    - roughly use a 15-character limit for column names

Possible other datasets to consider:
    matminer dielectric dataset
    matminer piezoelectric dataset
    https://www.nature.com/articles/sdata201865 (Shyam phonon) - AF
    https://www.nature.com/articles/sdata201882 (JARVIS-DFT optoelectronic)
    https://www.nature.com/articles/s41598-017-05402-0 (JARVIS-DFT-2D)
    OQMD? - AF
"""

module_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_dir, "sources")


def load_castelli_perovskites():
    """
    A dataset of 18,927 perovskites generated with ABX combinatorics, calculating
    gbllsc band gap and pbe structure, and also reporting absolute band edge
    positions and (delta)H_f

    References:
        http://pubs.rsc.org/en/content/articlehtml/2012/ee/c2ee22341d

    Returns:
        formula (input):
        fermi level (input): in eV
        fermi width (input): fermi bandwidth ??? calculated by ???
        e_form (input): heat of formation in eV???
        gap is direct (input):
        structure (input): crystal structure as pymatgen Structure object
        mu_b (input): magnetic moment in units of ????

        gap gllbsc (output): electronic band gap in eV calculated via gllbsc functional???
        vbm (output): absolute value of valence band edge calculated via gllbsc
            absolute value means that the number output from ??? is directly used
        cbm (output): similar tp vbm but for conduction band

    """
    df = pd.read_csv(os.path.join(data_dir, "castelli_perovskites.csv"))
    df["formula"] = df["A"] + df["B"] + df["anion"]
    df['vbm'] = np.where(df['is_direct'], df['VB_dir'], df['VB_ind'])
    df['cbm'] = np.where(df['is_direct'], df['CB_dir'], df['CB_ind'])
    df['gap gllbsc'] = np.where(df['is_direct'], df['gllbsc_dir-gap'], df['gllbsc_ind-gap'])
    df['structure'] = df['structure'].map(ast.literal_eval).map(Structure.from_dict)
    dropcols = ["filename", "XCFunctional", "anion_idx", "Unnamed: 0", "A", "B",
                "anion", "gllbsc_ind-gap", "gllbsc_dir-gap", "CB_dir", "CB_ind",
                "VB_dir", "VB_ind"]
    df = df.drop(dropcols, axis=1)
    colmap = {"sum_magnetic_moments": "mu_B",
              "is_direct": "gap is direct",
              "heat_of_formation_all": "heat of formation"}
    df = df.rename(columns=colmap)
    df.reindex(sorted(df.columns), axis=1)
    return df


def load_double_perovskites_gap(return_lumo=False):
    """
    Electronic band gaps of double perovskites calculated using ﻿Gritsenko,
    van Leeuwen, van Lenthe and Baerends potential (gllbsc) in GPAW.

    References:
        1) https://www.nature.com/articles/srep19375
        2) CMR database: https://cmr.fysik.dtu.dk/

    Args:
        return_lumo (bool): whether to return the lowest unoccupied molecular
            orbital (LUMO) energy levels (in eV).

    Returns:
        formula (input): chemical formula w/ sites in the order A1+B1+A2+B2+O6
            for example in KTaGaTaO6, A1=="K", B1=="Ta", A2=="Ga" and B2=="Ta"
        gap gllbsc (output): electronic band gap (in eV) calculated via gllbsc
        A1/B1/A2/B2 (input): species occupying the corresponding sites.
    """
    df = pd.read_excel(os.path.join(data_dir, 'double_perovskites.xlsx'),
                       sheet_name='bandgap')
    df = df.rename(columns={'A1_atom': 'A1', 'B1_atom': 'B1',
                            'A2_atom': 'A2', 'B2_atom': 'B2'})
    lumo = pd.read_excel(os.path.join(data_dir, 'double_perovskites.xlsx'),
                         sheet_name='lumo')
    if return_lumo:
        return df, lumo
    else:
        return df


def load_mp(filename='mp_nostruct.csv'):
    """
    Loads a pregenerated csv file containing properties of ALL materials in MP
    (approximately 70k).

    References:
        If you need a version WITH structures, use generate_data.py to create
        mp_all.csv, and use filename="sources/mp_all.csv"
    """

    df = pd.read_csv(os.path.join(data_dir, filename))
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

    References:
        https://www.nature.com/articles/sdata2017153#ref40
    """
    df = pd.read_csv(os.path.join(data_dir, "wolverton_oxides.csv"))
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

    References:
        http://iopscience.iop.org/article/10.1088/0953-8984/21/30/305403/meta
    """
    df = pd.read_csv(os.path.join(data_dir, "m2ax_elastic.csv"))
    colmap = {"M2AX phase": "formula",
              "B": "bulkmod",
              "G": "shearmod",
              "E": "elasticmod"}
    return df.rename(columns=colmap)


def load_glass_formation(phase="ternary"):
    """
    Metallic glass formation data, including ternary and binary alloys,
    collected from various experimental techniques such as melt-spinning or
    mechanical alloying.
    There are 7742 alloys in the "ternary" dataset and 5959 alloys in the
    "binary" dataset.

    Args:
        phase (str): "ternary" (as default) or "binary".

    References:
        "ternary": https://materials.springer.com/bp/docs/978-3-540-47679-5
                   https://www.nature.com/articles/npjcompumats201628
        "binary": https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01046
    """
    if phase == "ternary":
        df = pd.read_csv(os.path.join(data_dir, 'glasses_ternary.csv'),
                         index_col=0)
    elif phase == "binary":
        df = pd.read_csv(os.path.join(data_dir, 'glasses_binary.csv'),
                         index_col=0)
    else:
        raise ValueError("Unknown phase designation for glass formation "
                         "dataset: {}".format(phase))
    return df


def load_expt_formation_enthalpy():
    """
    Experimental formation enthalpies for inorganic compounds, collected from
    years of calorimetric experiments.
    There are 1,276 entries in this dataset, mostly binary compounds. Matching
    mp-ids or oqmd-ids as well as the DFT-computed formation energies are also
    added (if any).

    References:
        https://www.nature.com/articles/sdata2017162
    """
    df = pd.read_csv(os.path.join(data_dir, 'formation_enthalpy_expt.csv'))
    return df


def load_expt_gap():
    """
    Experimental band gap of inorganic semiconductors.

    References:
        https://pubs.acs.org/doi/suppl/10.1021/acs.jpclett.8b00124

    Returns:
        formula (input):
        gap expt (output): band gap (in eV) measured experimentally.
    """
    df = pd.read_csv(os.path.join(data_dir, 'zhuo_gap_expt.csv'))
    df = df.rename(columns={'composition': 'formula', 'Eg (eV)': 'gap expt'})
    return df


if __name__ == "__main__":
    # df = load_castelli_perovskites()
    # df = load_double_perovskites_gap()
    df = load_expt_gap()

    print(df.head())