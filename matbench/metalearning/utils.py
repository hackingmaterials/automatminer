import six
from collections import OrderedDict
from pymatgen.core import Composition, Element, Structure


class FormulaStatistics:
    """
    Get statistics of formulas. This is a helper function to the design
    of composition-related metafeatures.
    Args:
        formulas: a formula (str) or an iterable of formulas (list,
                  tuple, numpy.array or pandas.Series).
    """
    def __init__(self, formulas):
        self.formulas = formulas

    def calc(self):
        """
        Get categorical statistics of formulas.
        """
        if isinstance(self.formulas, six.string_types):
            self.formulas = [self.formulas]
        stats = OrderedDict()
        for formula in self.formulas:
            stats[formula] = FormulaStatistics.formula_category(formula)
        return stats

    @staticmethod
    def formula_category(formula):
        """
        Extract some categorical messages from the formula.
        Args:
            formula (str): a given formula.
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
                the formula, based on the predefined element_category function,
                sorted alphabetically.
                e.g. (1, 9) means there are both transitional metal and nonmetal
                in the formula.

            n_types (int):
                number of irreducible categories of elements in the formula.
                equal to the len(el_types_reduced).

            el_types ([int]):
                list of the unreduced categories of elements present in
                the formula, based on the predefined element_category function,
                sorted alphabetically.
                e.g. (1, 1, 9) means there are two types of transitional metal
                and one nonmetal in the formula.

            n_elements (int):
                number of elements in the formula.

            elements([str]):
                list of the symbols of elements in the formula.

        """
        c = Composition(formula)
        elements = [x.symbol for x in c.elements]
        n_elements = len(c.elements)
        el_types = sorted([FormulaStatistics.element_category(x)
                           for x in c.elements])
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

        prototype = FormulaStatistics.formula_prototype(formula)

        return {"major_formula_category": major_category,
                "minor_formula_category": minor_category,
                "prototype": prototype,
                "el_types_reduced": el_types_reduced,
                "n_types": n_types,
                "el_types": el_types,
                "n_elements": n_elements,
                "elements": elements}

    @staticmethod
    def formula_prototype(formula):
        """
        Guess the phase prototype from the integer anonymized_formula.
        Args:
            formula (str): a given formula.
        Returns:
            prototype:
                double_perovskites: 1
                unknown: 0
                ...to be continued

        """
        c = Composition(formula)
        c_int = Composition(c.get_integer_formula_and_factor()[0])
        f_int_anynomous = c_int.anonymized_formula
        prototype = 0
        if f_int_anynomous is "ABCDE6" and Element("O") in Composition(
                formula).elements:
            prototype = 1
        # to be continued
        return prototype

    @staticmethod
    def element_category(element):
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
        if element.is_transition_metal:
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


class StructureStatistics:
    """
    Get statistics of structures. This is a helper function to the design
    of strcture-related metafeatures.
    Args:
        structures: a Pymatgen Structure object or an iterable of Pymatgen
                    Structure objects (list, tuple, numpy.array or
                    pandas.Series).
    """
    def __init__(self, structures):
        self.structures = structures

    def calc(self):
        if isinstance(self.structures, Structure):
            self.structures = [self.structures]
        stats = OrderedDict()
        for i, structure in enumerate(self.structures):
            stats[i] = StructureStatistics.structure_category(structure)
        return stats

    @staticmethod
    def structure_category(structure):
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
        return {"n_sites": len(structure),
                "is_ordered": structure.is_ordered}
