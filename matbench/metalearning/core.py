from abc import ABCMeta, abstractmethod

"""

"""


class AbstractMetaFeature(object):
    """
    
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def calc(cls, X, y):
        pass


class MetaFeature(AbstractMetaFeature):
    def __init__(self, dependence=None):
        self.dependence = dependence
        super(MetaFeature, self).__init__()


# class MetaFeature(AbstractMetaFeature):
#     def __init__(self):
#         self.dependences = dict()
#         super(MetaFeature, self).__init__()
#
#     def set_dependence(self, name, dependence):
#         self.dependences[name] = dependence


class Helper(AbstractMetaFeature):
    def __init__(self):
        super(Helper, self).__init__()
