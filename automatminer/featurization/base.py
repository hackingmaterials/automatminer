"""
Base classes for sets of featurizers.
"""

import abc

__authors__ = ["Alex Dunn <ardunn@lbl.gov>", "Alex Ganose <aganose@lbl.gov>"]


class FeaturizerSet(abc.ABC):
    """Abstract class for defining sets of featurizers.

    All FeaturizerSets should implement at least fours sets of featurizers:

        - best - The set of best featurizers
        - fast - A set of generally applicable fast featurizers
        - debug - An ultra-minimal set of featurizers for debugging
        - all - The set of all featurizers

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
    def best(self):
        """List of featurizers providing useful features in a reasonable time.

        Featurizers that take a very long time to run, which crash for many
        systems, or which produce a large number of similar features will be
        excluded.
        """
        pass

    @property
    @abc.abstractmethod
    def all(self):
        """All featurizers available for this featurization type."""
        pass

    @property
    @abc.abstractmethod
    def fast(self):
        """Fast featurizers available for this featurization type."""
        pass

    @property
    @abc.abstractmethod
    def debug(self):
        """An ultra-minimal set of featurizers for debugging."""
        pass

    def _get_featurizers(self, featurizers):
        """Utility function for getting featurizers not in the ignore list."""
        return [f for f in featurizers
                if f.__class__.__name__ not in self.exclude]
