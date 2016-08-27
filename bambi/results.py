import pandas as pd
import pymc3 as pm
from abc import abstractmethod, ABCMeta


class ModelResults(object):

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

    def __init__(self, model, trace):

        self.trace = trace
        self.n_samples = len(trace)

    def plot(self, burn_in=0, names=None, **kwargs):
        return pm.traceplot(trace[burn_in:], varnames=names, **kwargs)

    def summary(self, burn_in=0, fixed=True, random=True, names=None,
                **kwargs):
        return pm.summary(trace[burn_in:], varnames=names, **kwargs)
