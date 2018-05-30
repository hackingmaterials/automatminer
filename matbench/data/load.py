import os
import ast
import json
import pandas as pd
import numpy as np
from pymatgen import Structure
from matminer.datasets.dataframe_loader import load_piezoelectric_tensor, \
    load_dielectric_constant, load_elastic_tensor

"""
All load* methods return the data in pandas.DataFrame. In each method a raw
data file is loaded, some preliminary transformation/renaming/cleaning done and
the result df is returned. For specific columns returned refer to the 
documentation of each function. All columns come with a ML input/output
suggestion, although some columns may be used as either input or output.

If you plan to add a new dataset please follow the guidelines and refer to 
documentation in load_castelli_perovskites for consistent docs. Generally, using
column names and unit conventions already used in other load methods is 
preferred (e.g., always using e_form for heat of formation in eV).

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
    https://www.nature.com/articles/sdata201865 (Shyam phonon) - AF
    OQMD? - AF
"""

module_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(module_dir, "sources")


def load_castelli_perovskites():
    """
    18,927 perovskites generated with ABX combinatorics, calculating gbllsc band
    gap and pbe structure, and also reporting absolute band edge positions and
    heat of formation.

    References:
        http://pubs.rsc.org/en/content/articlehtml/2012/ee/c2ee22341d

    Returns:
        formula (input):
        fermi level (input): in eV
        fermi width (input): fermi bandwidth
        e_form (input): heat of formation (eV)
        gap is direct (input):
        structure (input): crystal structure as pymatgen Structure object
        mu_b (input): magnetic moment in terms of Bohr magneton

        gap gllbsc (output): electronic band gap in eV calculated via gllbsc
            functional
        vbm (output): absolute value of valence band edge calculated via gllbsc
        cbm (output): similar to vbm but for conduction band
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
    colmap = {"sum_magnetic_moments": "mu_b",
              "is_direct": "gap is direct",
              "heat_of_formation_all": "e_form",
              "FermiLevel": "fermi level",
              "FermiWidth": "fermi width"}
    df = df.rename(columns=colmap)
    df.reindex(sorted(df.columns), axis=1)
    return df


def load_double_perovskites_gap(return_lumo=False):
    """
    Band gap of 1306 double perovskites (a_1b_1a_2b_2O6) calculated using ﻿
    Gritsenko, van Leeuwen, van Lenthe and Baerends potential (gllbsc) in GPAW.
    References:
        1) https://www.nature.com/articles/srep19375
        2) CMR database: https://cmr.fysik.dtu.dk/
    Args:
        return_lumo (bool): whether to return the lowest unoccupied molecular
            orbital (LUMO) energy levels (in eV).
    Returns:
        formula (input): chemical formula w/ sites in the a_1+b_1+a_2+b_2+O6
            order; e.g. in KTaGaTaO6, a_1=="K", b_1=="Ta", a_2=="Ga", b_2=="Ta"
        a1/b1/a2/b2 (input): species occupying the corresponding sites.
        gap gllbsc (output): electronic band gap (in eV) calculated via gllbsc
    """
    df = pd.read_excel(os.path.join(data_dir, 'double_perovskites.xlsx'),
                       sheet_name='bandgap')
    df = df.rename(columns={'A1_atom': 'a_1', 'B1_atom': 'b_1',
                            'A2_atom': 'a_2', 'B2_atom': 'b_2'})
    lumo = pd.read_excel(os.path.join(data_dir, 'double_perovskites.xlsx'),
                         sheet_name='lumo')
    if return_lumo:
        return df, lumo
    else:
        return df

def load_mp(filename='mp_nostruct.csv'):
    """
    Loads a pregenerated csv file containing properties of ALL materials in MP
    (approximately 70k). To regenerate the file, use generate.py. To use a
    version with structures, run generate_mp in generate.py and use the option
    filename='mp_all.csv'.

    References:
        https://materialsproject.org/citing

    Args:
        filename (str): The generated file to be loaded. By default, loads the
            generated MP file not containing structures.

    Returns:
        mpid (input): The Materials Project mpid, as a string.
        formula (input):
        structure (input): The Pymatgen structure object. Only present if the
            csv file containing structure is generated and loaded.

        e_hull (output): The calculated energy above the convex hull, in eV.
        gap pbe (output): The band gap in eV calculated with PBE-DFT functional
        mu_b (output): The total magnetization of the unit cell.
        bulk modulus (output): in GPa, average of Voight, Reuss, and Hill
        shear modulus (output): in GPa, average of Voight, Reuss, and Hill
        elastic anisotropy (output): The ratio of elastic anisotropy.

    Notes:
        If loading the csv with structures, loading will typically take ~10 min
        if using initial structures and about ~3-4 min if only using final
        structures.
    """

    df = pd.read_csv(os.path.join(data_dir, filename))
    df = df.drop("mpid", axis=1)
    for alias in ['structure', 'initial_structure']:
        if alias in df.columns.values:
            df[alias] = df[alias].map(ast.literal_eval).map(Structure.from_dict)
    colmap = {'material_id': 'mpid',
              'pretty_formula': 'formula',
              'band_gap': 'gap pbe',
              'e_above_hull': 'e_hull',
              'elasticity.K_VRH': 'bulk modulus',
              'elasticity.G_VRH': 'shear modulus',
              'elasticity.elastic_anisotropy': 'elastic anisotropy',
              'total_magnetization': 'mu_b'}
    return df.rename(columns=colmap)


def load_wolverton_oxides():
    """
    5,329 perovskite oxides containing composition data, lattice constants,
    and formation + vacancy formation energies. All perovskites are of the form
    ABO3.

    References:
        https://www.nature.com/articles/sdata2017153#ref40

    Returns:
        formula (input):
        atom a (input): The atom in the 'A' site of the pervoskite.
        atom b (input): The atom in the 'B' site of the perovskite.
        a (input): Lattice parameter a, in A (angstrom)
        b (input): Lattice parameter b, in A
        c (input): Lattice parameter c, in A
        alpha (input): Lattice angle alpha, in degrees
        beta (input): Lattice angle beta, in degrees
        gamma (input): Lattice angle gamma, in degrees

        lowest distortion (output): Local distortion crystal structure with
            lowest energy among all considered distortions.
        e_form (output): Formation energy in eV
        gap pbe (output): Bandgap in eV from PBE calculations
        mu_b (output): Magnetic moment
        e_form oxygen vacancy (output): Formation energy of oxygen vacancy (eV)
        e_hull (output): Energy above convex hull, wrt. OQMD db (eV)
        vpa (output): Volume per atom (A^3/atom)
    """
    df = pd.read_csv(os.path.join(data_dir, "wolverton_oxides.csv"))
    dropcols = ['In literature', 'Valence A', 'Valence B', 'Radius A [ang]',
                'Radius B [ang]']
    df = df.drop(dropcols, axis=1)
    colmap = {"Chemical formula": "formula",
              "A": "atom a",
              "B": "atom b",
              "Formation energy [eV/atom]": "e_form",
              "Band gap [eV]": "gap pbe",
              "Magnetic moment [mu_B]": "mu_b",
              "Vacancy energy [eV/O atom]": "e_form oxygen vacancy",
              "Stability [eV/atom]": "e_hull",
              "Volume per atom [A^3/atom]": 'vpa',
              "a [ang]": "a",
              "b [ang]": "b",
              "c [ang]": "c",
              "alpha [deg]": "alpha",
              "beta [deg]": "beta",
              "gamma [deg]": "gamma",
              "Lowest distortion": "lowest distortion"}
    df = df.rename(columns=colmap)
    df['e_form'] = pd.to_numeric(df['e_form'], errors='coerce')
    return df.dropna()


def load_m2ax():
    """
    Elastic properties of 224 stable M2AX compounds from "A comprehensive survey
    of M2AX phase elastic properties" by Cover et al. Calculations are PAW
    PW91.

    References:
        http://iopscience.iop.org/article/10.1088/0953-8984/21/30/305403/meta

    Returns:
        formula (input):
        a (input): Lattice parameter a, in A (angstrom)
        c (input): Lattice parameter c, in A
        d_mx (input): Distance from the M atom to the X atom
        d_ma (input): Distance from the M atom to the A atom

        c11/c12/c13/c33/c44 (output): Elastic constants of the M2AX material.
            These are specific to hexagonal materials.
        bulk modulus (output): in GPa
        shear modulus (output): in GPa
        elastic modulus (output): in GPa
    """
    df = pd.read_csv(os.path.join(data_dir, "m2ax_elastic.csv"))
    colmap = {"M2AXphase": "formula",
              "B": "bulk modulus",
              "G": "shear modulus",
              "E": "elastic modulus",
              "C11": "c11",
              "C12": "c12",
              "C13": "c13",
              "C33": "c33",
              "C44": "c44",
              "dMX": "d_mx",
              "dMA": "d_ma"}
    return df.rename(columns=colmap)


def load_glass_formation(phase="ternary"):
    """
    Metallic glass formation data, including ternary and binary alloys,
    collected from various experimental techniques such as melt-spinning or
    mechanical alloying.
    There are 7742 alloys in the "ternary" dataset and 5959 alloys in the
    "binary" dataset.

    References:
        "ternary": https://materials.springer.com/bp/docs/978-3-540-47679-5
                   https://www.nature.com/articles/npjcompumats201628
        "binary": https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01046

    Args:
        phase (str): "ternary" (as default) or "binary".

    Returns:
        formula (input): chemical formula
        phase (output): only in the "ternary" dataset, designating the phase
                        obtained in glass producing experiments,
                        "AM": amorphous phase
                        "CR": crystalline phase
                        "AC": amorphous-crystalline composite phase
                        "QC": quasi-crystalline phase
        gfa (output): glass forming ability, i.e. whether the composition can
                      form monolithic glass or not,
                      1: glass forming
                      0: non-glass forming

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
    mpids or oqmdids as well as the DFT-computed formation energies are also
    added (if any).

    References:
        https://www.nature.com/articles/sdata2017162

    Returns:
        formula (input): chemical formula
        pearson symbol (input): Pearson symbol of the structure
        space group (input): space group of the structure
        mpid (input): Materials project id (if any)
        oqmdid (input): OQMD id (if any)
        e_form expt (output): experimental formation enthaply (in eV/atom)
        e_form mp (output): formation enthalpy from Materials Project
                            (in eV/atom)
        e_form oqmd (output): formation enthalpy from OQMD (in eV/atom)
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

def load_jdft2d():
    """
    Properties of 493 2D materials, most of which (in their bulk forms) are in
    Materials Project. All energy calculations in the refined columns and
    structural relaxations were performed with the optB88-vdw functional.
    Magnetic properties were computed without +U correction.

    References:
        https://www.nature.com/articles/s41598-017-05402-0

    Returns:
        formula (input):
        mpid (input): Corresponding mpid string referring to MP bulk material
        structure (input): Pymatgen structure object
        stucture initial (input): Pymatgen structure before relaxation
        mu_b (input): Magnetic moment, in terms of bohr magneton

        e_form (output): Formation energy in eV
        gap optb88 (output): Band gap in eV using functional optB88-VDW
        e_exfol (output): Exfoliation energy (monolayer formation E) in eV
    """
    with open(os.path.join(data_dir, "jdft_2d.json")) as f:
        x = json.load(f)
    df = pd.DataFrame(x)
    colmap={'exfoliation_en': 'e_exfol',
            'final_str': 'structure',
            'initial_str': 'structure initial',
            'form_enp': 'e_form',
            'magmom': 'mu_b',
            'op_gap': 'gap optb88',
            }
    dropcols = ['epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz', 'kv', 'gv',
                'jid', 'kpoints', 'incar', 'icsd', 'mbj_gap', 'fin_en']
    df = df.drop(dropcols, axis=1)
    df = df.rename(columns=colmap)
    df['structure'] = df['structure'].map(Structure.from_dict)
    df['structure initial'] = df['structure initial'].map(Structure.from_dict)
    df['formula'] = [s.composition.reduced_formula for s in df['structure']]
    return df

def load_matminer_dielectric():
    """
    1,056 structures with dielectric properties calculated with DFPT-PBE.

    References:
        1) https://www.nature.com/articles/sdata2016134
        2) https://www.sciencedirect.com/science/article/pii/S0927025618303252

    Returns:
        mpid (input): material id via MP
        formula (input):
        structure (input):
        nsites (input): The number of sites in the structure

        gap pbe (output): Band gap in eV
        refractive index (output): Estimated refractive index
        ep_e poly (output): Polycrystalline electronic contribution to
            dielectric constant (estimate/avg)
        ep poly (output): Polycrystalline dielectric constant (estimate/avg)
        pot. ferroelectic (output): If imaginary optical phonon modes present at
            the Gamma point, the material is potentially ferroelectric
    """
    df = load_dielectric_constant()
    dropcols = ['volume', 'space_group', 'e_electronic', 'e_total']
    df = df.drop(dropcols, axis=1)
    colmap={'material_id': 'mpid',
            'band_gap': 'gap pbe',
            'n': 'refractive index',
            'poly_electronic': 'ep_e poly',
            'poly_total': 'ep poly',
            'pot_ferroelectric': 'pot. ferroelectric'
            }
    df = df.rename(columns=colmap)
    return df

def load_matminer_elastic():
    """
    1,180 structures with elastic properties calculated with DFT-PBE.

    References:
        1) https://www.nature.com/articles/sdata20159
        2) https://www.sciencedirect.com/science/article/pii/S0927025618303252

    Returns:
        mpid (input): material id via MP
        formula (input):
        structure (input):
        nsites (input): The number of sites in the structure

        elastic anisotropy (output): ratio of anisotropy of elastic properties
        shear modulus (output): in GPa
        bulk modulus (output): in GPa
        poisson ratio (output):

    Notes:
        This function may return a subset of information which is present in
        load_mp. However, this dataframe is 'clean' with regard to elastic
        properties.
    """
    df = load_elastic_tensor()
    dropcols = ['volume', 'space_group', 'G_Reuss', 'G_Voigt', 'K_Reuss',
                'K_Voigt', 'compliance_tensor', 'elastic_tensor',
                'elastic_tensor_original']
    df = df.drop(dropcols, axis=1)
    colmap = {'material_id': 'mpid',
              'elastic_anisotropy': 'elastic anisotropy',
              'G_VRH': 'shear modulus',
              'K_VRH': 'bulk modulus',
              'poisson_ratio': 'poisson ratio',
              }
    df = df.rename(columns=colmap)
    return df

def load_matminer_piezoelectric():
    """
    941 structures with piezoelectric properties calculated with DFT-PBE.

    References:
        1) https://www.nature.com/articles/sdata201553
        2) https://www.sciencedirect.com/science/article/pii/S0927025618303252

    Returns:
        mpid (input):
        formula (input):
        structure (input):

        eij_max (output): Maximum attainable absolute value of the longitudinal
            piezoelectric modulus
        vmax_x/y/z (output): vmax = [vmax_x, vmax_y, vmax_z]. vmax is the
            direction of eij_max (or family of directions, e.g., <111>)
    """
    df = load_piezoelectric_tensor()
    df['v_max'] = [np.fromstring(str(x)[1:-1], sep=',') for x in df['v_max']]
    df['vmax_x'] = [v[0] for v in df['v_max']]
    df['vmax_y'] = [v[1] for v in df['v_max']]
    df['vmax_z'] = [v[2] for v in df['v_max']]

    dropcols = ['point_group', 'piezoelectric_tensor', 'volume', 'space_group',
                'v_max']
    df = df.drop(columns=dropcols, axis=1)
    colmap = {'material_id': 'mpid'}
    df = df.rename(columns=colmap)
    return df

if __name__ == "__main__":
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    # print(load_mp('mp_all.csv'))
    print(load_matminer_piezoelectric())