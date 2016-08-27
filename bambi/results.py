import pandas as pd
import pymc3 as pm
from abc import abstractmethod, ABCMeta


class ModelResults(object):
    '''
    Base class for ModelResults hierarchy.
    Args:
        model (Model): a bambi Model instance specifying the model.
    '''

    __metaclass__ = ABCMeta

    def __init__(self, model):

        self.model = model
        self.terms = list(model.terms.values())
        self.diagnostics = model._diagnostics
        self.n_terms = len(model.terms)

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def summary(self):
        pass


class PyMC3ModelResults(ModelResults):
    '''
    Holds PyMC3 sampler results and provides plotting and summarization tools.
    Args:
        model (Model): a bambi Model instance specifying the model.
        trace (MultiTrace): a PyMC3 MultiTrace object returned by the sampler. 
    '''
    def __init__(self, model, trace):

        self.trace = trace
        self.n_samples = len(trace)
        super(PyMC3ModelResults, self).__init__(model)

    def plot(self, burn_in=0, names=None, **kwargs):
        '''
        Plots posterior distributions and sample traces. Currently just a
        wrapper for pm.traceplot().
        '''
        return pm.traceplot(trace[burn_in:], varnames=names, **kwargs)

    def summary(self, burn_in=0, fixed=True, random=True, names=None,
                **kwargs):
        '''
        Summarizes all parameter estimates. Currently just a wrapper for
        pm.summary().
        '''
        return pm.summary(trace[burn_in:], varnames=names, **kwargs)
