import six
from collections import OrderedDict

from pymatgen.core.composition import Composition, Element
from pymatgen.core.structure import Structure, IStructure
from pymatgen.core.periodic_table import DummySpecie

__author__ = ["Qi Wang <wqthu11@gmail.com>"]

def composition_statistics(compositions):
    """
    Get statistics of compositions. This is a helper function to the design
    of composition-related metafeatures.
    Args:
        compositions: a composition (str) or an iterable of compositions (list,
                      tuple, numpy.array or pandas.Series).
    """

    if isinstance(compositions, six.string_types):
        compositions = [compositions]
    stats = OrderedDict()
    for idx, composition in enumerate(compositions):
        stats[idx] = _composition_summary(composition)
    return stats


def _composition_summary(composition):
    """
    Extract some categorical messages from the composition.
    Args:
        composition (str): a given composition.
    Returns:
        dict of the following messages:
        major_category (int):
            all_metal: 1
            metal_nonmetal: 2
            all_nonmetal: 3, equal to organic? No
            unknown: 0
            ...to be continued

        minor_category (int):
            all_transitional_metal(except for rare_earth_metal/actinoid): 1
            all_rare_earth_metal: 2
            all_actinoid: 3
            all_alkali: 4
            all_alkaline: 5
            all_groupIIIA: 6
            unknown: 0
            ...to be continued

        prototype (int):
            double_perovskites: 1
            unknown: 0
            ...to be continued

        el_types_reduced ([int])：
            list of the irreducible categories of elements present in
            the composition, based on the predefined element_category function,
            sorted alphabetically.
            e.g. (1, 9) means there are both transitional metal and nonmetal
            in the composition.

        n_types (int):
            number of irreducible categories of elements in the composition.
            equal to the len(el_types_reduced).

        el_types ([int]):
            list of the unreduced categories of elements present in
            the composition, based on the predefined element_category function,
            sorted alphabetically.
            e.g. (1, 1, 9) means there are two types of transitional metal
            and one nonmetal in the composition.

        n_elements (int):
            number of elements in the composition.

        elements([str]):
            list of the symbols of elements in the composition.

    """
    try:
        c = Composition(composition)
    except BaseException:
        return {"major_composition_category": np.nan,
            "minor_composition_category": np.nan,
            "prototype": np.nan,
            "el_types_reduced": np.nan,
            "n_types": np.nan,
            "el_types": np.nan,
            "n_elements": np.nan,
            "elements": np.nan}
    elements = [x.symbol for x in c.elements]
    n_elements = len(c.elements)
    el_types = sorted([_element_category(x) for x in c.elements])
    n_types = len(el_types)
    el_types_reduced = list(set(el_types))

    major_category, minor_category = 0, 0
    # if there are only elements of one type, can be 1-11
    if len(el_types_reduced) == 1:
        if el_types_reduced[0] < 7:
            major_category = 1
        else:
            major_category = 3
        minor_category = el_types_reduced
    # if there are many types of metallic elements
    elif all([el_type < 7 for el_type in el_types_reduced]):
        major_category = 1
        minor_category = el_types_reduced  # just return the list for now
    elif any([el_type < 7 for el_type in el_types_reduced]):
        major_category = 2
        minor_category = el_types_reduced  # just return the list for now
    elif all([7 <= el_type < 11 for el_type in el_types_reduced]):
        major_category = 3

    prototype = _composition_prototype(composition)

    return {"major_composition_category": major_category,
            "minor_composition_category": minor_category,
            "prototype": prototype,
            "el_types_reduced": el_types_reduced,
            "n_types": n_types,
            "el_types": el_types,
            "n_elements": n_elements,
            "elements": elements}


def _composition_prototype(composition):
    """
    Guess the phase prototype from the integer anonymized_composition.
    Args:
        composition (str): a given composition.
    Returns:
        prototype:
            double_perovskites: 1
            unknown: 0
            ...to be continued

    """
    c = Composition(composition)
    c_int = Composition(c.get_integer_formula_and_factor()[0])
    f_int_anynomous = c_int.anonymized_formula
    prototype = 0
    if f_int_anynomous is "ABCDE6" and Element("O") in Composition(
            composition).elements:
        prototype = 1
    # to be continued
    return prototype


def _element_category(element):
    """
    Define the category of a given element.
    Args:
        element: an element symbol or a Pymatgen Element object
    Returns:
        metallic:
            is_transitional_metal(except for rare_earth_metal/actinoid): 1
            is_rare_earth_metal: 2
            is_actinoid: 3
            is_alkali: 4
            is_alkaline: 5
            is_groupIIIA_VIIA: 6 ("Al", "Ga", "In", "Tl", "Sn", "Pb",
                                  "Bi", "Po")
        non-metallic:
            is_metalloid: 7 ("B", "Si", "Ge", "As", "Sb", "Te", "Po")
            is_halogen: 8
            is_nonmetal: 9 ("C", "H", "N", "P", "O", "S", "Se")
            is_noble_gas: 10

        other-radiactive-etc:
            other: 11 (only a few elements are not covered by the
                       above categories)
    """
    if not isinstance(element, Element) and isinstance(element,
                                                       six.string_types):
        element = Element(element)
    if isinstance(element, DummySpecie):
        return 11
    elif element.is_transition_metal:
        if element.is_lanthanoid or element.symbol in {"Y", "Sc"}:
            return 2
        elif element.is_actinoid:
            return 3
        else:
            return 1
    elif element.is_alkali:
        return 4
    elif element.is_alkaline:
        return 5
    elif element.symbol in {"Al", "Ga", "In", "Tl", "Sn", "Pb", "Bi", "Po"}:
        return 6
    elif element.is_metalloid:
        return 7
    # elif element.is_chalcogen:
    #     return 8
    elif element.is_halogen:
        return 8
    elif element.symbol in {"C", "H", "N", "P", "O", "S", "Se"}:
        return 9
    elif element.is_noble_gas:
        return 10
    else:
        return 11


def structure_statistics(structures):
    """
    Get statistics of structures. This is a helper function to the design
    of strcture-related metafeatures.
    Args:
        structures: a Pymatgen Structure object or an iterable of Pymatgen
                    Structure objects (list, tuple, numpy.array or
                    pandas.Series).
    """
    if isinstance(structures, (Structure, IStructure)):
        structures = [structures]
    stats = OrderedDict()
    for idx, structure in enumerate(structures):
        stats[idx] = _structure_summary(structure)
    return stats


def _structure_summary(structure):
    """
    Extract messages from the structure.
    Args:
        structure: a Pymatgen Structure object
    Returns:
        dict of the following messages:
        nsites (int): number of sites in the structure.
        is_ordered (bool): whether the structure is ordered or not.
        ...to be continued

    """
    return {"n_sites": len(structure.sites),
            "is_ordered": structure.is_ordered}
