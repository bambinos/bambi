from abc import abstractmethod, ABCMeta
from bambi.external.six import string_types
import re
import warnings
import pandas as pd
import numpy as np
import pymc3 as pm
from pymc3 import diagnostics as pmd


__all__ = ['MCMCResults', 'PyMC3ADVIResults']


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
        self.diagnostics = model._diagnostics \
            if hasattr(model, '_diagnostics') else None

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def summary(self):
        pass


class MCMCResults(ModelResults):
    '''
    Holds sampler results; provides slicing, plotting, and summarization tools.
    Args:
        model (Model): a bambi Model instance specifying the model.
        data (numpy array): Raw storage of MCMC samples in array with
            dimensions 0, 1, 2 = samples, chains, variables
        names (list): Names of all Terms.
        dims (list): Numbers of levels for all Terms.
        levels (list): Names of all levels for all Terms.
        transformed (list): Optional list of variable names to treat as
            transformed--and hence, to exclude from the output by default.
    '''

    def __init__(self, model, data, names, dims, levels, transformed_vars=None):
        # store the arguments
        self.data = data
        self.names = names
        self.dims = dims
        self.levels = levels
        self.transformed_vars = transformed_vars

        # compute basic stuff to use later
        self.n_samples = data.shape[0]
        self.n_chains = data.shape[1]
        self.n_params = data.shape[2]
        self.n_terms = len(names)
        if transformed_vars is not None:
            utv = list(set(names) - set(transformed_vars))
        else:
            utv = names
        self.untransformed_vars = utv
        # this keeps track of which columns in 'data' go with which terms
        self.index = np.cumsum([0] + [x[0] if len(x) else 1 for x in dims][:-1])

        # build level_dict: dictionary of lists containing levels of each Term
        level_dict = {}
        for i, name, dim in zip(self.index, names, dims):
            dim = dim[0] if len(dim) else 1
            level_dict[name] = levels[i:(i+dim)]
        self.level_dict = level_dict

        super(MCMCResults, self).__init__(model)

    def __getitem__(self, idx):
        '''
        If a variable name, return MCMCResults with only that variable
            e.g., fit['subject']
        If a list of variable names, return MCMCResults with those variables
            e.g., fit[['subject','item]]
        If a slice, return MCMCResults with sliced samples
            e.g., fit[500:]
        If a tuple, return MCMCResults with those variables sliced
            e.g., fit[['subject','item'], 500:] OR fit[500:, ['subject','item']]
        '''

        if isinstance(idx, slice):
            var = self.names
            vslice = idx
        elif isinstance(idx, string_types):
            var = [idx]
            vslice = slice(0, self.n_samples)
        elif isinstance(idx, list):
            if not all([isinstance(x, string_types) for x in idx]):
                raise ValueError("If passing a list, all elements must be "
                                 "parameter names.")
            var = idx
            vslice = slice(0, self.n_samples)
        elif isinstance(idx, tuple):
            if len(idx) > 2:
                raise ValueError("Only two arguments can be passed. If you want"
                                 " to select multiple parameters and a subset "
                                 "of samples, pass a slice and a list of "
                                 "parameter names.")
            vslice = [i for i, x in enumerate(idx) if isinstance(x, slice)]
            if not len(vslice):
                raise ValueError("At least one argument must be a slice. If "
                                 "you want to select multiple parameters by "
                                 "name, pass a list (not a tuple) of names.")
            if len(vslice) > 1:
                raise ValueError("Slices can only be applied "
                                 "over the samples dimension.")
            var = idx[1 - vslice[0]]
            vslice = idx[vslice[0]]
            if not isinstance(var, (list, tuple)): var = [var] 
        else:
            raise ValueError("Unrecognized index type.")

        # do slicing/selection and return subsetted copy of MCMCResults
        levels = sum([self.level_dict[v] for v in var], [])
        level_iloc = [self.levels.index(x) for x in levels]
        var_iloc = [self.names.index(v) for v in var]
        return MCMCResults(model=self.model,
            data=self.data[vslice, :, level_iloc], names=var,
            dims=[self.dims[x] for x in var_iloc], levels=levels,
            transformed_vars=self.transformed_vars)

    def get_chains(self, indices):
        # Return copy of self but only for chains with the passed indices
        if not isinstance(indices, (list, tuple)): indices = [indices]
        return MCMCResults(model=self.model, data=self.data[:, indices, :],
                names=self.names, dims=self.dims, levels=self.levels,
                transformed_vars=self.transformed_vars)

    def _filter_names(self, exclude_ranefs=True, hide_transformed=True):
        names = self.untransformed_vars if hide_transformed else self.names
        if exclude_ranefs:
            names = [x for x in names \
                if re.sub(r'\[.+\]$', '', x) not in self.model.random_terms]
        return names

    def plot(self, burn_in=0, names=None, annotate=True, exclude_ranefs=False, 
        hide_transformed=True, kind='trace', **kwargs):
        '''
        Plots posterior distributions and sample traces. Basically a wrapper
        for pm.traceplot() plus some niceties, based partly on code from:
        https://pymc-devs.github.io/pymc3/notebooks/GLM-model-selection.html
        Args:
            burn_in (int): Number of initial samples to exclude before
                summary statistics are computed.
            names (list): Optional list of variable names to summarize.
            annotate (bool): If True (default), add lines marking the
                posterior means, write the posterior means next to the
                lines, and add factor level names for fixed factors with
                more than one distribution on the traceplot.
            exclude_ranefs (bool): If True, do not show trace plots for
                individual random effects. Defaults to False.
            hide_transformed (bool): If True (default), do not show trace
                plots for internally transformed variables.
            kind (str): Either 'trace' (default) or 'priors'. If 'priors',
                this just internally calls Model.plot()
        '''
        if kind == 'priors':
            return self.model.plot()

        # if no 'names' specified, filter out unwanted variables
        if names is None:
            names = self._filter_names(names, exclude_ranefs, hide_transformed)

        # compute means for all variables and factors
        if annotate:
            kwargs['lines'] = {param: self.trace[param, burn_in:].mean() \
                for param in names}
            # factors (fixed terms with shape > 1) must be handled separately
            factors = {}
            for fix in self.model.fixed_terms.values():
                if 'b_'+fix.name in names and len(fix.levels)>1:
                    # remove factor from dictionary of lines to plot
                    kwargs['lines'].pop('b_'+fix.name)
                    # add factor and its column means to dictionary of factors
                    factors.update({'b_'+fix.name:
                        {':'.join(re.findall('\[([^]]+)\]', x)):
                         self.trace['b_'+fix.name, burn_in:].mean(0)[i]
                         for i,x in enumerate(fix.levels)}})

        # make the traceplot
        ax = pm.traceplot(self.trace[burn_in:], varnames=names,
            figsize=(12,len(names)*1.5), **kwargs)

        if annotate:
            # add lines and annotation for the factors
            for f in factors.keys():
                for lev in factors[f]:
                    # draw line
                    ax[names.index(f),0].axvline(x=factors[f][lev],
                        color="r", lw=1.5)
                    # write the mean
                    ax[names.index(f),0].annotate(
                        '{:.2f}'.format(factors[f][lev]),
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
                ax[names.index(v),0].annotate(
                    '{:.2f}'.format(kwargs['lines'][v]),
                    xy=(kwargs['lines'][v],0), xycoords='data', xytext=(5,10),
                    textcoords='offset points', rotation=90, va='bottom',
                    fontsize='large', color='#AA0022')

        # For binomial models with n_trials = 1 (most common use case),
        # tell user which event is being modeled
        if self.model.family.name=='binomial' and \
            np.max(self.model.y.data) < 1.01:
            event = next(i for i,x in enumerate(self.model.y.data.flatten()) \
                if x>.99)
            warnings.warn('Modeling the probability that {}==\'{}\''.format(
                self.model.y.name,
                str(self.model.data[self.model.y.name][event])))

        return ax

    def summary(self, burn_in=0, exclude_ranefs=True, names=None,
                hide_transformed=True, mc_error=False, **kwargs):
        '''
        Returns a DataFrame of summary/diagnostic statistics for the parameters.
        Args:
            burn_in (int): Number of initial samples to exclude before
                summary statistics are computed.
            exclude_ranefs (bool): If True (default), do not print
                summary statistics for individual random effects.
            names (list): Optional list of variable names to summarize.
            hide_transformed (bool): If True (default), do not print
                summary statistics for internally transformed variables.
            mc_error (bool): If True (defaults to False), include the monte
                carlo error for each parameter estimate.
        '''

        # if no 'names' specified, filter out unwanted variables
        if names is None:
            names = self._filter_names(names, exclude_ranefs, hide_transformed)

        # get the basic DataFrame
        df = pm.df_summary(self.trace[burn_in:], varnames=names, **kwargs)
        df.set_index([[self._prettify_name(x) for x in df.index]], inplace=True)

        # append diagnostic info if there are multiple chains.
        if self.trace.nchains > 1:
            # first remove unwanted variables so we don't waste time on those
            diag_trace = self.trace[burn_in:]
            for var in set(diag_trace.varnames) - set(names):
                diag_trace.varnames.remove(var)
            # append each diagnostic statistic
            for diag_fn,diag_name in zip([pmd.effective_n, pmd.gelman_rubin],
                                         ['effective_n',   'gelman_rubin']):
                # compute the diagnostic statistic
                stat = diag_fn(diag_trace)
                # rename stat indices to match df indices
                for k, v in list(stat.items()):
                    stat.pop(k)
                    # handle categorical predictors w/ >3 levels
                    if isinstance(v, np.ndarray) and len(v) > 1:
                        for i,x in enumerate(v):
                            ugly_name = '{}__{}'.format(k, i)
                            stat[self._prettify_name(ugly_name)] = x
                    # handle all other variables
                    else:
                        stat[self._prettify_name(k)] = v
                # append to df
                stat = pd.DataFrame(stat, index=[diag_name]).T
                df = df.merge(stat, how='left', left_index=True, right_index=True)
        else:
            warnings.warn('Multiple MCMC chains (i.e., njobs > 1) are required'
                          ' in order to compute convergence diagnostics.')

        # drop the mc_error column if requested
        if not mc_error:
            df = df.drop('mc_error', axis=1)

        # For binomial models with n_trials = 1 (most common use case),
        # tell user which event is being modeled
        if self.model.family.name=='binomial' and \
            np.max(self.model.y.data) < 1.01:
            event = next(i for i,x in enumerate(self.model.y.data.flatten()) \
                if x>.99)
            warnings.warn('Modeling the probability that {}==\'{}\''.format(
                self.model.y.name,
                str(self.model.data[self.model.y.name][event])))

        return df

    def to_df(self, exclude_ranefs=True, hide_transformed=True):
        '''
        Returns the MCMC samples in a nice, neat pandas DataFrame with all
        MCMC chains concatenated.
        Args:
            exclude_ranefs (bool): If True (default), do not return samples
                for individual random effects.
            hide_transformed (bool): If True (default), do not return
                samples for internally transformed variables.
        '''
        # filter out unwanted variables
        names = self._filter_names(exclude_ranefs, hide_transformed)

        # concatenate the (pre-sliced) chains
        data = [self.data[:, i, :] for i in range(self.n_chains)]
        data = np.concatenate(data, axis=0)

        # construct the trace DataFrame
        df = sum([self.level_dict[x] for x in names], [])
        df = pd.DataFrame({x:data[:, self.levels.index(x)] for x in df})

        return df


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

