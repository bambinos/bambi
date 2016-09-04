import pandas as pd
import pymc3 as pm
from pymc3.model import TransformedRV
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

        # here we determine which variables have been internally transformed 
        # (e.g., sd_log). transformed variables are actually 'untransformed' from 
        # the PyMC3 model's perspective, and it's the backtransformed (e.g., sd)
        # variable that is 'transformed'. So the logic of this is to find and
        # remove 'untranformed' variables that have a 'transformed' counterpart,
        # in that 'untransformed' varname is the 'transfornmed' varname plus
        # some suffix (such as '_log' or '_interval')
        rvs = model.backend.model.unobserved_RVs
        trans = set(var.name for var in rvs if isinstance(var, TransformedRV))
        untrans = set(var.name for var in rvs) - trans
        untrans = set(x for x in untrans if not any([t in x for t in trans]))
        self.untransformed_vars = [x for x in trace.varnames if x in trans | untrans]

        super(PyMC3Results, self).__init__(model)

    def plot(self, burn_in=0, names=None, annotate=True, hide_transformed=True, **kwargs):
        '''
        Plots posterior distributions and sample traces. Code slightly modified from:
        https://pymc-devs.github.io/pymc3/notebooks/GLM-model-selection.html
        '''
        if names is None:
            names = self.untransformed_vars if hide_transformed else self.trace.varnames

        # compute means for all variables and factors
        if annotate:
            kwargs['lines'] = {param: self.trace[param][burn_in:].mean() for param in names}
            # factors (i.e., fixed terms with shape > 1) must be handled separately
            factors = {}
            for fix in self.model.fixed_terms.values():
                if 'b_'+fix.name in names and len(fix.levels)>1:
                    # remove factor from dictionary of lines to plot
                    kwargs['lines'].pop('b_'+fix.name)
                    # add factor and its column means to dictionary of factors
                    factors.update({'b_'+fix.name:
                        {':'.join(re.findall('\[([^]]+)\]', x)):
                         self.trace['b_'+fix.name][burn_in:].mean(0)[i]
                         for i,x in enumerate(fix.levels)}})

        # make the traceplot
        ax = pm.traceplot(self.trace[burn_in:], varnames=names,
            figsize=(12,len(names)*1.5), **kwargs)

        if annotate:
            # add lines and annotation for the factors
            for f in factors.keys():
                for lev in factors[f]:
                    # draw line
                    ax[names.index(f),0].axvline(x=factors[f][lev], color="r", lw=1.5)
                    # write the mean
                    ax[names.index(f),0].annotate('{:.2f}'.format(factors[f][lev]),
                        xy=(factors[f][lev],0), xycoords='data', xytext=(5,10),
                        textcoords='offset points', rotation=90, va='bottom',
                        fontsize='large', color='#AA0022')
                    # write the factor level name
                    ax[names.index(f),0].annotate(lev,
                        xy=(factors[f][lev],0), xycoords='data', xytext=(-11,5),
                        textcoords='offset points', rotation=90, va='bottom',
                        fontsize='large', color='#AA0022')
            # add lines and annotation for the rest of the variables
            for v in kwargs['lines'].keys():
                ax[names.index(v),0].annotate('{:.2f}'.format(kwargs['lines'][v]),
                    xy=(kwargs['lines'][v],0), xycoords='data', xytext=(5,10),
                    textcoords='offset points', rotation=90, va='bottom',
                    fontsize='large', color='#AA0022')


        return ax

    def summary(self, burn_in=0, fixed=True, random=True, names=None,
                hide_transformed=True, **kwargs):
        '''
        Summarizes all parameter estimates. Currently just a wrapper for
        pm.summary().
        '''
        if names is None:
            names = self.untransformed_vars if hide_transformed else self.trace.varnames

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
