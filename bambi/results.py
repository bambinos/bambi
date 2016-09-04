import pandas as pd
from abc import abstractmethod, ABCMeta
import re


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
        self.diagnostics = model._diagnostics if hasattr(model, '_diagnostics') else None
        self.n_terms = len(model.terms)

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def summary(self):
        pass


class PyMC3Results(ModelResults):

    '''
    Holds PyMC3 sampler results and provides plotting and summarization tools.
    Args:
        model (Model): a bambi Model instance specifying the model.
        trace (MultiTrace): a PyMC3 MultiTrace object returned by the sampler. 
    '''

    def __init__(self, model, trace):

        self.trace = trace
        self.n_samples = len(trace)
        super(PyMC3Results, self).__init__(model)

    def plot(self, burn_in=0, names=None, annotate=True, **kwargs):
        '''
        Plots posterior distributions and sample traces. Code slightly modified from:
        https://pymc-devs.github.io/pymc3/notebooks/GLM-model-selection.html
        '''
        import pymc3 as pm
        if names is None: names = self.trace.varnames

        # make the basic traceplot
        if annotate:
            kwargs['lines'] = {re.sub('\__0$', '', k): v['mean']
                for k, v in pm.df_summary(self.trace[burn_in:]).iterrows()}
        ax = pm.traceplot(self.trace[burn_in:], varnames=names,
            figsize=(12,len(names)*1.5), **kwargs)

        # add lines and annotation for the means of all parameters
        if annotate:
            means = [self.trace[param][burn_in:].mean() for param in names]
            for i, mn in enumerate(means):
                ax[i,0].annotate('{:.2f}'.format(mn), xy=(mn,0),
                    xycoords='data', xytext=(5,10), textcoords='offset points',
                    rotation=90, va='bottom', fontsize='large', color='#AA0022')

        return ax

    def summary(self, burn_in=0, fixed=True, random=True, names=None,
                **kwargs):
        '''
        Summarizes all parameter estimates. Currently just a wrapper for
        pm.summary().
        '''
        import pymc3 as pm
        return pm.summary(self.trace[burn_in:], varnames=names, **kwargs)


class PyMC3ADVIResults(ModelResults):
    '''
    Holds PyMC3 ADVI results and provides plotting and summarization tools.
    Args:
        model (Model): a bambi Model instance specifying the model.
        params (MultiTrace): ADVI parameters returned by PyMC3.
    '''
    def __init__(self, model, params):

        self.means = params['means']
        self.sds = params['stds']
        self.elbo_vals = params['elbo_vals']
        super(PyMC3ADVIResults, self).__init__(model)
