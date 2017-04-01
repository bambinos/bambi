from abc import ABCMeta, abstractmethod


class BackEnd(object):

    '''
    Base class for BackEnd hierarchy.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def _convert_to_results(self):
        pass
