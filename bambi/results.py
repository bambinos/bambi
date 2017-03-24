from abc import abstractmethod, ABCMeta
from bambi.external.six import string_types
from bambi.diagnostics import gelman_rubin, effective_n
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
                if re.sub(r'_offset$', '', x) not in self.model.random_terms]
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

    def _hpd_interval(self, x, width):
        """
        Code adapted from pymc3.stats.calc_min_interval:
        https://github.com/pymc-devs/pymc3/blob/master/pymc3/stats.py
        """
        x = np.sort(x)
        n = len(x)

        interval_idx_inc = int(np.floor(width * n))
        n_intervals = n - interval_idx_inc
        interval_width = x[interval_idx_inc:] - x[:n_intervals]

        if len(interval_width) == 0:
            raise ValueError('Too few elements for interval calculation')

        min_idx = np.argmin(interval_width)
        hdi_min = x[min_idx]
        hdi_max = x[min_idx + interval_idx_inc]

        index = ['hpd{}_{}'.format(width, x) for x in ['lower','upper']]
        return pd.Series([hdi_min, hdi_max], index=index)

    def summary(self, exclude_ranefs=True, hide_transformed=True, hpd=.95,
                quantiles=None, diagnostics=[gelman_rubin, effective_n]):
        '''
        Returns a DataFrame of summary/diagnostic statistics for the parameters.
        Args:
            exclude_ranefs (bool): If True (default), do not print
                summary statistics for individual random effects.
            hide_transformed (bool): If True (default), do not print
                summary statistics for internally transformed variables.
            hpd (float, between 0 and 1): Show Highest Posterior Density (HPD)
                intervals with specified width/proportion for all parameters.
                If None, HPD intervals are suppressed.
            quantiles (float [or list of floats] between 0 and 1): Show
                specified quantiles of the marginal posterior distributions for
                all parameters. If None (default), no quantiles are shown.
            diagnostics (list): List of functions to use to compute convergence
                diagnostics for all parameters. Functions must return a
                DataFrame with one labeled row per parameter. If None, no
                convergence diagnostics are computed.
        '''
        samples = self.to_df(exclude_ranefs, hide_transformed)

        # build the basic DataFrame
        df = pd.DataFrame({'mean':samples.mean(0),'sd':samples.std(0)})

        # add user-specified quantiles
        if quantiles is not None:
            if not isinstance(quantiles, (list, tuple)): quantiles = [quantiles]
            qnames = ['q' + str(q) for q in quantiles]
            df = df.merge(samples.quantile(quantiles).set_index([qnames]).T,
                left_index=True, right_index=True)

        # add HPD intervals
        if hpd is not None:
            df = df.merge(samples.apply(self._hpd_interval, axis=0, width=hpd).T,
                left_index=True, right_index=True)

        # add convergence diagnostics
        if diagnostics is not None:
            if self.n_chains > 1:
                for diag in diagnostics:
                    df = df.merge(diag(self), left_index=True, right_index=True)
            else:
                warnings.warn('Multiple MCMC chains are required in order '
                              'to compute convergence diagnostics.')

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

