from abc import ABCMeta, abstractmethod
import time
from collections import OrderedDict

import scipy.sparse

from autosklearn.util.logging_ import get_logger

"""
Thanks to autosklearn for the design of these classes.
"""


class MetafeatureFunctions(object):
    def __init__(self):
        self.functions = OrderedDict()
        self.dependencies = OrderedDict()
        self.values = OrderedDict()

    def clear(self):
        self.values = OrderedDict()

    def __iter__(self):
        return self.functions.__iter__()

    def __getitem__(self, item):
        return self.functions.__getitem__(item)

    def __setitem__(self, key, value):
        return self.functions.__setitem__(key, value)

    def __delitem__(self, key):
        return self.functions.__delitem__(key)

    def __contains__(self, item):
        return self.functions.__contains__(item)

    def get_value(self, key):
        return self.values[key].value

    def set_value(self, key, item):
        self.values[key] = item

    def is_calculated(self, key):
        """Return if a helper function has already been executed.

        Necessary as get_value() can return None if the helper function hasn't
        been executed or if it returned None."""
        return key in self.values

    def get_dependency(self, name):
        """Return the dependency of metafeature "name".
        """
        return self.dependencies.get(name)

    def define(self, name, dependency=None):
        """Decorator for adding metafeature functions to a "dictionary" of
        metafeatures. This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(metafeature_class):
            instance = metafeature_class()
            self.__setitem__(name, instance)
            self.dependencies[name] = dependency
            return instance
        return wrapper


class HelperFunctions(object):
    def __init__(self):
        self.functions = OrderedDict()
        self.values = OrderedDict()

    def clear(self):
        self.values = OrderedDict()
        self.computation_time = OrderedDict()

    def __iter__(self):
        return self.functions.__iter__()

    def __getitem__(self, item):
        return self.functions.__getitem__(item)

    def __setitem__(self, key, value):
        return self.functions.__setitem__(key, value)

    def __delitem__(self, key):
        return self.functions.__delitem__(key)

    def __contains__(self, item):
        return self.functions.__contains__(item)

    def is_calculated(self, key):
        """Return if a helper function has already been executed.

        Necessary as get_value() can return None if the helper function hasn't
        been executed or if it returned None."""
        return key in self.values

    def get_value(self, key):
        return self.values.get(key).value

    def set_value(self, key, item):
        self.values[key] = item

    def define(self, name):
        """Decorator for adding helper functions to a "dictionary".
        This behaves like a function decorating a function,
        not a class decorating a function"""
        def wrapper(metafeature_class):
            instance = metafeature_class()
            self.__setitem__(name, instance)
            return instance
        return wrapper


class AbstractMetaFeature(object):
    """
    Thanks to auto-sklearn for the design of AbstractMetaFeature
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        self.logger = get_logger(__name__)

    @abstractmethod
    def _calculate(cls, X, y, categorical):
        pass

    def __call__(self, X, y, categorical=None):
        if categorical is None:
            categorical = [False for i in range(X.shape[1])]
        starttime = time.time()

        try:
            if scipy.sparse.issparse(X) and hasattr(self, "_calculate_sparse"):
                value = self._calculate_sparse(X, y, categorical)
            else:
                value = self._calculate(X, y, categorical)
            comment = ""
        except MemoryError as e:
            value = None
            comment = "Memory Error"

        endtime = time.time()
        return MetaFeatureValue(self.__class__.__name__, self.type_,
                                0, 0, value, endtime-starttime, comment=comment)


class MetaFeature(AbstractMetaFeature):
    def __init__(self):
        super(MetaFeature, self).__init__()
        self.type_ = "METAFEATURE"


class HelperFunction(AbstractMetaFeature):
    def __init__(self):
        super(HelperFunction, self).__init__()
        self.type_ = "HELPERFUNCTION"

class MetaFeatureValue(object):
    def __init__(self, name, type_, fold, repeat, value, time, comment=""):
        self.name = name
        self.type_ = type_
        self.fold = fold
        self.repeat = repeat
        self.value = value
        self.time = time
        self.comment = comment

    def to_arff_row(self):
        if self.type_ == "METAFEATURE":
            value = self.value
        else:
            value = "?"

        return [self.name, self.type_, self.fold,
                self.repeat, value, self.time, self.comment]

    def __repr__(self):
        repr = "%s (type: %s, fold: %d, repeat: %d, value: %s, time: %3.3f, " \
               "comment: %s)"
        repr = repr % tuple(self.to_arff_row()[:4] +
                            [str(self.to_arff_row()[4])] +
                            self.to_arff_row()[5:])
        return repr
