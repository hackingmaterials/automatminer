"""
Base classes for sets of featurizers.
"""

import abc
from typing import List

__authors__ = ["Alex Dunn <ardunn@lbl.gov>", "Alex Ganose <aganose@lbl.gov>"]


class FeaturizerSet(abc.ABC):
    """Abstract class for defining sets of featurizers.

    All FeaturizerSets should implement at least fours sets of featurizers:

        - express - The "go-to" set of featurizers
        - heavy - A more expensive and complete (though not necessarily
            better) version of express.
        - all - All featurizers available for the intended featurization type(s)
        - debug - An ultra-minimal set of featurizers for debugging purposes.

    Each set returned is a list of matminer featurizer objects. The choice of
    featurizers for a given set is at the discrtetion of the implementor.

    Args:
        exclude (list of str, optional): A list of featurizer class names that
            will be excluded from the set of featurizers returned.
    """

    def __init__(self, exclude=None):
        self.exclude = exclude if exclude else []

    def __call__(self, *args, **kwargs):
        return self.all

    @property
    @abc.abstractmethod
    def express(self) -> List:
        """A focused set of featurizers which should:

        * be reasonably fast to featurize
        * be not prone to errors/nans
        * provide informative learning features
        * do not include many irrelevant features making ML expensive
        * have each featurizer return a vector
        * allow the recognized type (structure, composition, etc.) as input.
        """
        pass

    @property
    @abc.abstractmethod
    def heavy(self) -> List:
        """A more expensive and complete (though not necessarily better)
        version of express.

        Similar to express, all featurizers selected should return useful
        learning features. However the selected featurizers may now:

        * generate many (thousands+) features
        * be expensive to featurize (1s+ per item)
        * be prone to NaNs on certain datasets
        """
        pass

    @property
    @abc.abstractmethod
    def all(self) -> List:
        """All featurizers available for this featurization type. These
        featurizers are allowed to:

        * have multiple, highly similar versions of the same featurizer,
        * not work on standard versions of the input types (e.g., SiteDOS works
            on the DOS for a single site, not structure
        * return non-vectorized outputs (e.g., matrices, other data types).
        """
        pass

    @property
    @abc.abstractmethod
    def debug(self) -> List:
        """An ultra-minimal set of featurizers for debugging."""
        pass

    def _get_featurizers(self, featurizers: List) -> List:
        """Utility function for getting featurizers not in the ignore list."""
        return [f for f in featurizers if f.__class__.__name__ not in self.exclude]
