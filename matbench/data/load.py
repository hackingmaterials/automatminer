import os
import ast
import json
import warnings

import pandas as pd
import numpy as np
from pymatgen import Structure
from matminer.datasets.dataframe_loader import load_piezoelectric_tensor, \
    load_dielectric_constant, load_elastic_tensor, load_flla
from matminer.utils.io import load_dataframe_from_json
from matminer.featurizers.conversions import StructureToComposition

"""
All load* methods return the data in pandas.DataFrame. In each method a raw
data file is loaded, some preliminary transformation/renaming/cleaning done and
the result df is returned. For specific columns returned refer to the 
documentation of each function. All columns come with a ML input/target
suggestion, although some columns may be used as either input or target.

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
    
Data convention guidelines
    - If structures are present, the dataframe should have them contained in a
        column where each entry is a dictionary
    - The structures should NOT be strings (MP queries can return strings via 
        REST, so be cautious)
    - To convert strings to dictionary, use ast.literal_eval
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
        structure (input): crystal structure as dict representing pymatgen
            Structure
        mu_b (input): magnetic moment in terms of Bohr magneton

        gap gllbsc (target): electronic band gap in eV calculated via gllbsc
            functional
        vbm (target): absolute value of valence band edge calculated via gllbsc
        cbm (target): similar to vbm but for conduction band
    """
    df = pd.read_csv(os.path.join(data_dir, "castelli_perovskites.csv"))
    df["formula"] = df["A"] + df["B"] + df["anion"]
    df['vbm'] = np.where(df['is_direct'], df['VB_dir'], df['VB_ind'])
    df['cbm'] = np.where(df['is_direct'], df['CB_dir'], df['CB_ind'])
    df['gap gllbsc'] = np.where(df['is_direct'], df['gllbsc_dir-gap'],
                                df['gllbsc_ind-gap'])
    df['structure'] = df['structure'].map(ast.literal_eval)
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
        gap gllbsc (target): electronic band gap (in eV) calculated via gllbsc
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
        structure (input): The dict of Pymatgen structure object. Only present
            if the csv file containing structure is generated and loaded.
        initial structure (input): The dict of Pymatgen structure object before
            relaxation. Only present if the csv file containing initial
            structure is generated and loaded.

        e_hull (target): The calculated energy above the convex hull, in eV per
            atom
        gap pbe (target): The band gap in eV calculated with PBE-DFT functional
        e_form (target); Formation energy per atom (eV)
        mu_b (target): The total magnetization of the unit cell.
        bulk modulus (target): in GPa, average of Voight, Reuss, and Hill
        shear modulus (target): in GPa, average of Voight, Reuss, and Hill
        elastic anisotropy (target): The ratio of elastic anisotropy.

    Notes:
        If loading the csv with structures, loading will typically take ~10 min
        if using initial structures and about ~3-4 min if only using final
        structures.
    """

    df = pd.read_csv(os.path.join(data_dir, filename))
    dropcols = ['energy', 'energy_per_atom']
    df = df.drop(dropcols, axis=1)
    for alias in ['structure', 'initial_structure']:
        if alias in df.columns.values:
            df[alias] = df[alias].map(ast.literal_eval)
    colmap = {'material_id': 'mpid',
              'pretty_formula': 'formula',
              'band_gap': 'gap pbe',
              'e_above_hull': 'e_hull',
              'elasticity.K_VRH': 'bulk modulus',
              'elasticity.G_VRH': 'shear modulus',
              'elasticity.elastic_anisotropy': 'elastic anisotropy',
              'total_magnetization': 'mu_b',
              'initial_structure': 'initial structure',
              'formation_energy_per_atom': 'e_form'}
    return df.rename(columns=colmap)


def load_boltztrap_mp():
    """
    Effective mass and thermoelectric properties of 9036 compounds in The
    Materials Project database that are calculated by the BoltzTraP software
    package run on the GGA-PBE or GGA+U density functional theory calculation
    results.

    References:
        https://www.nature.com/articles/sdata201785

    Returns:
        mpid (input): The Materials Project mpid, as a string.
        formula (input):
        structure (input):

        m_n (target): n-type/conduction band effective mass. Units: m_e where
            m_e is the mass of an electron; i.e. m_n is a unitless ratio
        m_p (target): p-type/valence band effective mass.
        s_n (target): n-type Seebeck coefficient in micro Volts per Kelvin
        s_p (target): p-type Seebeck coefficient in micro Volts per Kelvin
        pf_n (target): n-type thermoelectric power factor in uW/cm2.K where
            uW is microwatts and a constant relaxation time of 1e-14 assumed.
        pf_p (target): p-type power factor in uW/cm2.K

    Note:
        * To avoid data leakage, one may only set the target to one of the target
        columns listed. For example, S_n is strongly correlated with PF_n
        and usually when one is available the other one is available too.
        * It is recommended that dos and bandstructure objects are retrieved
        from Materials Porject and then dos, bandstructure and composition
        featurizers are used to generate input features.
    """
    df = pd.read_csv(os.path.join(data_dir, 'boltztrap_mp.csv'),
                     index_col=False)
    df = df.rename(columns={'S_n': 's_n', 'S_p': 's_p',
                            'PF_n': 'pf_n', 'PF_p': 'pf_p'})
    df = df.dropna()
    df['structure'] = df['structure'].map(ast.literal_eval)
    warnings.warn('When training a model on the load_boltztrap_mp data, to'
                  ' avoid data leakage, one may only set the target to one of the target'
                  ' columns listed. For example, s_n is strongly correlated with pf_n'
                  ' and usually when one is available the other one is available too.')
    return df


def load_phonon_dielectric_mp():
    """
    Phonon (lattice/atoms vibrations) and dielectric properties of 1439
    compounds computed via ABINIT software package in the harmonic
    approximation based on density functional perturbation theory.

    References:
        https://www.nature.com/articles/sdata201865

    Returns:
        mpid (input): The Materials Project mpid, as a string.
        formula (input):
        structure (input):

        eps_total (target): total calculated dielectric constant. Unitless:
            it is a ratio over the dielectric constant at vacuum.
        eps_electronic (target): electronic contribution to the calculated
            dielectric constant; unitless.
        last phdos peak (target): the frequency of the last calculated phonon
            density of states in 1/cm; may be used as an estimation of dominant
            longitudinal optical phonon frequency, a descriptor.

    Notes:
        * Only one of these three targets must be used in a training to prevent
        data leakage.
        * For training, retrieval of formulas and structures via mpids hence
            the usage of composition and structure featurizers is recommended.
    """
    df = pd.read_csv(os.path.join(data_dir, 'phonon_dielectric_mp.csv'))
    df = df[df['asr_breaking'] < 30].drop('asr_breaking', axis=1)
    # remove entries not having structure, formula, or a target
    df = df.dropna()
    df['structure'] = df['structure'].map(ast.literal_eval)
    return df.reset_index(drop=True)


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

        lowest distortion (target): Local distortion crystal structure with
            lowest energy among all considered distortions.
        e_form (target): Formation energy in eV
        gap pbe (target): Bandgap in eV from PBE calculations
        mu_b (target): Magnetic moment
        e_form oxygen (target): Formation energy of oxygen vacancy (eV)
        e_hull (target): Energy above convex hull, wrt. OQMD db (eV)
        vpa (target): Volume per atom (A^3/atom)
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
              "Vacancy energy [eV/O atom]": "e_form oxygen",
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
    for k in ['e_form', 'gap pbe', 'e_hull', 'vpa', 'e_form oxygen']:
        df[k] = pd.to_numeric(df[k], errors='coerce')
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

        c11/c12/c13/c33/c44 (target): Elastic constants of the M2AX material.
            These are specific to hexagonal materials.
        bulk modulus (target): in GPa
        shear modulus (target): in GPa
        elastic modulus (target): in GPa
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


def load_glass_binary():
    """
    Metallic glass formation data for binary alloys, collected from various
    experimental techniques such as melt-spinning or mechanical alloying.
    This dataset covers all compositions with an interval of 5 at.% in 59
    binary systems, containing a total of 5959 alloys in the dataset.
    The target property of this dataset is the glass forming ability (GFA),
    i.e. whether the composition can form monolithic glass or not, which is
    either 1 for glass forming or 0 for non-full glass forming.

    References:
        https://pubs.acs.org/doi/10.1021/acs.jpclett.7b01046

    Returns:
        formula (input): chemical formula
        phase (target): only in the "ternary" dataset, designating the phase
                        obtained in glass producing experiments,
                        "AM": amorphous phase
                        "CR": crystalline phase
        gfa (target): glass forming ability, correlated with the phase column,
                      designating whether the composition can form monolithic
                      glass or not,
                      1: glass forming ("AM")
                      0: non-full-glass forming ("CR")

    """
    df = pd.read_csv(os.path.join(data_dir, 'glass_binary.csv'))
    return df


def load_glass_ternary_landolt(processing="meltspin", unique_composition=True):
    """
    Metallic glass formation dataset for ternary alloys, collected from the
    "Nonequilibrium Phase Diagrams of Ternary Amorphous Alloys,’ a volume of
    the Landolt– Börnstein collection.
    This dataset contains experimental measurements of whether it is
    possible to form a glass using a variety of processing techniques at
    thousands of compositions from hundreds of ternary systems.
    The processing techniques are designated in the "processing" column.

    There are originally 7191 experiments in this dataset, will be reduced to
    6203 after deduplicated, and will be further reduced to 6118 if combining
    multiple data for one composition.
    There are originally 6780 melt-spinning experiments in this dataset,
    will be reduced to 5800 if deduplicated, and will be further reduced to
    5736 if combining multiple experimental data for one composition.

    References:
        https://materials.springer.com/bp/docs/978-3-540-47679-5
        https://www.nature.com/articles/npjcompumats201628

    Args:
        processing (str): "meltspin" or "sputtering" or "all"
        unique_composition (bool): whether combine data from difference sources
                                   but for the same composition

    Returns:
        formula (input): chemical formula
        phase (target): only in the "ternary" dataset, designating the phase
                        obtained in glass producing experiments,
                        "AM": amorphous phase
                        "CR": crystalline phase
                        "AC": amorphous-crystalline composite phase
                        "QC": quasi-crystalline phase
        processing (condition): "meltspin" or "sputtering"
        gfa (target): glass forming ability, correlated with the phase column,
                      designating whether the composition can form monolithic
                      glass or not,
                      1: glass forming ("AM")
                      0: non-full-glass forming ("CR" or "AC" or "QC")

    """
    df = pd.read_csv(os.path.join(data_dir, 'glass_ternary_landolt.csv'))
    df.drop_duplicates()
    if processing in ["meltspin", "sputtering"]:
        df = df[df["processing"] == processing]
    df["gfa"] = df["phase"].apply(lambda x: 1 if x == "AM" else 0)
    if unique_composition:
        df = df.groupby("formula").max().reset_index()
    return df


def load_glass_ternary_hipt(system="all"):
    """
    Metallic glass formation dataset for ternary alloys, collected from the
    high-throughput sputtering experiments measuring whether it is possible
     to form a glass using sputtering.

    The hipt experimental data are of the Co-Fe-Zr, Co-Ti-Zr, Co-V-Zr and
    Fe-Ti-Nb ternary systems.

    References:
        http://advances.sciencemag.org/content/4/4/eaaq1566

    Args:
        system (str): "CoFeZr", "CoTiZr", "CoVZr","FeTiNb" or a list of these
                      systems e.g. ["CoFeZr", "CoVZr"] or "all"

    Returns:
        formula (input): chemical formula
        system (condition): selected system(s)
        processing (condition): "sputtering"
        phase (target): only in the "ternary" dataset, designating the phase
                obtained in glass producing experiments,
                "AM": amorphous phase
                "CR": crystalline phase
        gfa (target): glass forming ability, correlated with the phase column,
                      designating whether the composition can form monolithic
                      glass or not,
                      1: glass forming ("AM")
                      0: non-full-glass forming ("CR")

    """
    df = pd.read_csv(os.path.join(data_dir, 'glass_ternary_hipt.csv'))
    if isinstance(system, str):
        if system == "all":
            return df
        else:
            try:
                return df[df["system"] == system]
            except:
                raise AttributeError("this system {} is not in this dataset".
                                     format(system))
    else:
        try:
            return df[df["system"].isin(system)]
        except:
            raise AttributeError("some of the system list {} are not in this "
                                 "dataset". format(system))


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
        e_form expt (target): experimental formation enthaply (in eV/atom)
        e_form mp (target): formation enthalpy from Materials Project
                            (in eV/atom)
        e_form oqmd (target): formation enthalpy from OQMD (in eV/atom)
    """
    df = pd.read_csv(os.path.join(data_dir, 'formation_enthalpy_expt.csv'))
    return df


def load_expt_gap():
    """
    Experimental band gap of 6354 inorganic semiconductors.

    References:
        https://pubs.acs.org/doi/suppl/10.1021/acs.jpclett.8b00124

    Returns:
        formula (input):
        gap expt (target): band gap (in eV) measured experimentally.
    """
    df = pd.read_csv(os.path.join(data_dir, 'zhuo_gap_expt.csv'))
    df = df.rename(columns={'composition': 'formula', 'Eg (eV)': 'gap expt'})
    # The numbers in 323 formulas such as 'AgCNO,65' or 'Sr2MgReO6,225' are
    # space group numbers confirmed by Jakoah Brgoch the corresponding author
    df['formula'] = df['formula'].apply(lambda x: x.split(',')[0])
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
        structure (input): Dict representation of pymatgen structure object
        stucture initial (input): Pymatgen structure before relaxation (as dict)
        mu_b (input): Magnetic moment, in terms of bohr magneton

        e_form (target): Formation energy in eV
        gap optb88 (target): Band gap in eV using functional optB88-VDW
        e_exfol (target): Exfoliation energy (monolayer formation E) in eV
    """
    with open(os.path.join(data_dir, "jdft_2d.json")) as f:
        x = json.load(f)
    df = pd.DataFrame(x)
    colmap = {'exfoliation_en': 'e_exfol',
              'final_str': 'structure',
              'initial_str': 'structure initial',
              'form_enp': 'e_form',
              'magmom': 'mu_b',
              'op_gap': 'gap optb88',
              }
    dropcols = ['epsx', 'epsy', 'epsz', 'mepsx', 'mepsy', 'mepsz', 'kv', 'gv',
                'jid', 'kpoints', 'incar', 'icsd', 'mbj_gap', 'fin_en']
    df = df.drop(dropcols, axis=1)
    df = df.replace('na', np.nan)
    df = df.rename(columns=colmap)
    df['structure'] = df['structure']
    df['structure initial'] = df['structure initial']
    df['formula'] = [Structure.from_dict(s).composition.reduced_formula for s
                     in df['structure']]
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
        structure (input): dict form of Pymatgen structure
        nsites (input): The number of sites in the structure

        gap pbe (target): Band gap in eV
        refractive index (target): Estimated refractive index
        ep_e poly (target): Polycrystalline electronic contribution to
            dielectric constant (estimate/avg)
        ep poly (target): Polycrystalline dielectric constant (estimate/avg)
        pot. ferroelectric (target): If imaginary optical phonon modes present at
            the Gamma point, the material is potentially ferroelectric
    """
    df = load_dielectric_constant()
    dropcols = ['volume', 'space_group', 'e_electronic', 'e_total']
    df = df.drop(dropcols, axis=1)
    df['structure'] = [s.as_dict() for s in df['structure']]
    colmap = {'material_id': 'mpid',
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
        structure (input): dict form of Pymatgen structure
        nsites (input): The number of sites in the structure

        elastic anisotropy (target): ratio of anisotropy of elastic properties
        shear modulus (target): in GPa
        bulk modulus (target): in GPa
        poisson ratio (target):

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
    df['structure'] = [s.as_dict() for s in df['structure']]
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
        mpid (input): material id via MP
        formula (input): string formula
        structure (input): dict form of Pymatgen structure
        nsites (input): The number of sites in the structure

        eij_max (target): Maximum attainable absolute value of the longitudinal
            piezoelectric modulus
        vmax_x/y/z (target): vmax = [vmax_x, vmax_y, vmax_z]. vmax is the
            direction of eij_max (or family of directions, e.g., <111>)
    """
    df = load_piezoelectric_tensor()
    df['v_max'] = [np.fromstring(str(x)[1:-1], sep=',') for x in df['v_max']]
    df['vmax_x'] = [v[0] for v in df['v_max']]
    df['vmax_y'] = [v[1] for v in df['v_max']]
    df['vmax_z'] = [v[2] for v in df['v_max']]

    dropcols = ['point_group', 'piezoelectric_tensor', 'volume', 'space_group',
                'v_max']
    df['structure'] = [s.as_dict() for s in df['structure']]
    df = df.drop(columns=dropcols, axis=1)
    colmap = {'material_id': 'mpid'}
    df = df.rename(columns=colmap)
    return df


def load_matminer_flla():
    """
    3938 structures and formation energies from "Crystal Structure
    Representations for Machine Learning Models of Formation Energies."

    References:
        1) https://arxiv.org/abs/1503.07406
        2) https://aip.scitation.org/doi/full/10.1063/1.4812323

    Returns:
        mpid (input): material id via MP
        formula (input): string formula
        structure (input): dict form of Pymatgen structure

        e_form (target): Formation energy in eV/atom
        e_hull (target): Energy above hull, in form
    """
    df = load_flla()
    df = df.drop(["formula", "formation_energy", "nsites"], axis=1)
    df["formula"] = [s.composition.reduced_formula for s in df['structure']]
    df["structure"] = [s.as_dict() for s in df['structure']]
    df = df.rename(
        {"formation_energy_per_atom": "e_form", "e_above_hull": "e_hull",
         "material_id": "mpid"}, axis=1)
    return df


def load_heusler_magnetic():
    """
    1153 Heusler alloys with DFT-calculated magnetic and electronic properties.
    The 1153 alloys include 576 full, 449 half and 128 inverse Heusler alloys.
    The data are extracted and cleaned (including de-duplicating) from Citrine.

    References:
        https://citrination.com/datasets/150561/

    Returns:
        formula (input): chemical formula
        heusler type (input): Full, Half or Inverse Heusler
        num_electron: No. of electrons per formula unit
        struct type (input): Structure type
        latt const (input): Lattice constant
        tetragonality (input): Tetragonality, i.e. c/a

        e_form (target): Formation energy in eV/atom
        pol fermi (target?): Polarization at Fermi level in %
        mu_b (target): Magnetic moment
        mu_b saturation (target?) Saturation magnetization in emu/cc

        other columns dropped for now:
        gap width: No gap or the gap width value
        stability: True or False, can be inferred from e_form:
                   True if e_form<0, False if e_form>0

    """
    df = pd.read_csv(os.path.join(data_dir, 'heusler_magnetic.csv'))
    dropcols = ['gap width', 'stability']
    df = df.drop(dropcols, axis=1)
    return df


def load_steel_strength():
    """
    312 steels with experimental yield strength and ultimate tensile strength,
    extracted and cleaned (including de-duplicating) from Citrine.

    References:
        https://citrination.com/datasets/153092/

    Returns:
        formula (input): chemical formula
        c (input): weight percent of C
        mn (input): weight percent of Mn
        si (input): weight percent of Si
        cr (input): weight percent of Cr
        ni (input): weight percent of Ni
        mo (input): weight percent of Mo
        v (input): weight percent of V
        n (input): weight percent of N
        nb (input): weight percent of Nb
        co (input): weight percent of Co
        w (input): weight percent of W
        al (input): weight percent of Al
        ti (input): weight percent of Ti
        -These weight percent values of alloying elements are suggested as
         features by a related paper.

        yield strength (target): yield strength in GPa
        tensile strength (target): ultimate tensile strength in GPa
        elongation (target): elongation in %

    """
    df = pd.read_csv(os.path.join(data_dir, 'steel_strength.csv'))
    return df


def load_citrine_thermal_conductivity(room_temperature=True):
    """
    Thermal conductivity of 872 compounds measured experimentally and retrieved
    from Citrine database from various references. The reported values are
    measured at various temperatures of which 295 are at room temperature. The
    latter subset is return by default.

    References:
        https://citrineinformatics.github.io/python-citrination-client/

    Returns:
        formula (input): chemical formula of compounds
        k_expt (target): the experimentally measured thermal conductivity in SI
            units of W/m.K
    """
    df = pd.read_csv(os.path.join(data_dir, 'citrine_thermal_conductivity.csv'))

    df = df[df['k-units'].isin(
        ['W/m.K', 'W/m$\\cdot$K', 'W/mK', 'W\\m K', 'Wm$^{-1}$K$^{-1}$'])]
    if room_temperature:
        df = df[df['k_condition'].isin(['room temperature', 'Room temperature',
                                        'Standard', '298', '300'])]
    df = df.drop(['k-units', 'k_condition', 'k_condition_units'], axis=1)
    return df

# def load_jarvis_dft():
#     """
#     Various properties of 24,759 bulk and 2D materials computed with the
#     OptB88vdW and TBmBJ functionals taken from the JARVIS DFT database.
#
#     References:
#         https://arxiv.org/pdf/1805.07325.pdf
#         https://www.nature.com/articles/sdata201882
#         https://doi.org/10.1103/PhysRevB.98.014107
#
#     Returns:
#         formula (input): chemical formula of compounds
#         mpid (input): Materials Project id
#         jid (input): JARVIs id
#         composition (input):
#         structure (input):
#         e_exfol (target): exfoliation energy per atom in eV/atom
#         e_form (target): formation energy per atom, in eV/atom
#         gap opt (target): Band gap calculated with OptB88vDW functional, in eV
#         gap tbmbj (target): Band gap calculated with TBMBJ functional, in eV
#         mu_b (target): Magnetic moment, in Bohr Magneton
#         bulk modulus (target): VRH average calculation of bulk modulus
#         shear modulus (target): VRH average calculation of shear modulus
#         e mass_x (target): Effective electron mass in x direction (BoltzTraP)
#         e mass_y (target): Effective electron mass in y direction (BoltzTraP)
#         e mass_z (target): Effective electron mass in z direction (BoltzTraP)
#         hole mass_x (target): Effective hole mass in x direction (BoltzTraP)
#         hole mass_y (target): Effective hole mass in y direction (BoltzTraP)
#         hole mass_z (target): Effective hole mass in z direction (BoltzTraP)
#         epsilon_x opt (target): Static dielectric function in x direction
#             calculated with OptB88vDW functional.
#         epsilon_y opt (target): Static dielectric function in y direction
#             calculated with OptB88vDW functional.
#         epsilon_z opt (target): Static dielectric function in z direction
#             calculated with OptB88vDW functional.
#         epsilon_x tbmbj (target): Static dielectric function in x direction
#             calculated with TBMBJ functional.
#         epsilon_y tbmbj (target): Static dielectric function in y direction
#             calculated with TBMBJ functional.
#         epsilon_z tbmbj (target): Static dielectric function in z direction
#             calculated with TBMBJ functional.
#     """
#
#     df = load_dataframe_from_json(os.path.join(data_dir, 'jdft_3d.json'))
#
#     colmap = {"el_mass_x": "e mass_x",
#             "el_mass_y": "e mass_y",
#             "el_mass_z": "e mass_z",
#             "epsx": "epsilon_x opt",
#             "epsy": "epsilon_y opt",
#             "epsz": "epsilon_z opt",
#             "exfoliation_en": "e_exfol",
#             "form_enp": "e_form",
#             "gv": "shear modulus",
#             "hl_mass_x": "hole mass_x",
#             "hl_mass_y": "hole mass_y",
#             "hl_mass_z": "hole mass_z",
#             "kv": "bulk modulus",
#             "magmom": "mu_b",
#             "mbj_gap": "gap tbmbj",
#             "mepsx": "epsilon_x tbmbj",
#             "mepsy": "epsilon_y tbmbj",
#             "mepsz": "epsilon_z tbmbj",
#             "op_gap": "gap tbmbj",
#             "strt": "structure",
#             }
#
#     df = df.rename(columns=colmap)
#     df = df.drop(["multi_elastic", "fin_enp"], axis=1)
#     s = StructureToComposition()
#     df = s.featurize_dataframe(df, "structure")
#     return df


if __name__ == "__main__":
    # pd.set_option('display.height', 1000)
    # pd.set_option('display.max_rows', 50)
    # pd.set_option('display.max_columns', 500)
    # pd.set_option('display.width', 1000)
    print(load_heusler_magnetic())
