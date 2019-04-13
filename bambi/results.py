from abc import abstractmethod, ABCMeta
from bambi.external.six import string_types
from bambi import diagnostics as bmd
import re
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utils import listify
try:
    import pymc3 as pm
    if hasattr(pm.plots, 'kdeplot_op'):
        pma = pm.plots
    else:
        pma = pm.plots.artists
except:
    pma = None


__all__ = ['MCMCResults', 'PyMC3ADVIResults']


class ModelResults(object):

    '''Base class for ModelResults hierarchy.

    Attributes:
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

    '''Holds sampler results; provides slicing, plotting, and summarization tools.

    Attributes:
        model (Model): a bambi Model instance specifying the model.
        data (array-like): Raw storage of MCMC samples in array with
            dimensions 0, 1, 2 = samples, chains, variables
        names (list): Names of all Terms.
        dims (list): Numbers of levels for all Terms.
        levels (list): Names of all levels for all Terms.
        transformed (list): Optional list of variable names to treat as
            transformed--and hence, to exclude from the output by default.
    '''

    def __init__(self, model, data, names, dims, levels,
                 transformed_vars=None):
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
        self.index = np.cumsum(
            [0] + [x[0] if len(x) else 1 for x in dims][:-1])

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
                raise ValueError("Only two arguments can be passed. If you "
                                 "want to select multiple parameters and a "
                                 "subset of samples, pass a slice and a list "
                                 "of parameter names.")
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
            if not isinstance(var, (list, tuple)):
                var = [var]
        else:
            raise ValueError("Unrecognized index type.")

        # do slicing/selection and return subsetted copy of MCMCResults
        levels = sum([self.level_dict[v] for v in var], [])
        level_iloc = [self.levels.index(x) for x in levels]
        var_iloc = [self.names.index(v) for v in var]
        return MCMCResults(model=self.model,
                           data=self.data[vslice, :, level_iloc], names=var,
                           dims=[self.dims[x] for x in var_iloc],
                           levels=levels,
                           transformed_vars=self.transformed_vars)

    def get_chains(self, indices):
        # Return copy of self but only for chains with the passed indices
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        return MCMCResults(model=self.model, data=self.data[:, indices, :],
                           names=self.names, dims=self.dims,
                           levels=self.levels,
                           transformed_vars=self.transformed_vars)

    def _filter_names(self, varnames=None, ranefs=False, transformed=False):
        names = self.untransformed_vars if not transformed else self.names
        if varnames is not None:
            names = [n for n in listify(varnames) if n in names]
        if not ranefs:
            names = [x for x in names if re.sub(r'_offset$', '', x)
                     not in self.model.random_terms]
        Intercept = [n for n in names if "Intercept" in n]
        std = [n for n in names if "_" in n]
        rand_eff = [n for n in names if "|" in n and n not in std]
        interac = [n for n in names if ":" in n and n not in rand_eff + std]
        main_eff = [n for n in names if n not in interac + std + rand_eff + Intercept]
        names = Intercept + sorted(main_eff, key=len) + sorted(interac, key=len) \
            + sorted(rand_eff, key=len) + sorted(std, key=len)
        return names

    def plot(self, varnames=None, ranefs=True, transformed=False,
             combined=False, hist=False, bins=20, kind='trace'):
        '''Plots posterior distributions and sample traces.

        Basically a wrapperfor pm.traceplot() plus some niceties, based partly on code
        from: https://pymc-devs.github.io/pymc3/notebooks/GLM-model-selection.html.

        Args:
            varnames (list): List of variable names to plot. If None, all
                eligible variables are plotted.
            ranefs (bool): If True (default), shows trace plots for
                individual random effects.
            transformed (bool): If False (default), excludes internally
                transformed variables from plotting.
            combined (bool): If True, concatenates all chains into one before
                plotting. If False (default), plots separately lines for
                each chain (on the same axes).
            hist (bool): If True, plots a histogram for each fixed effect,
                in addition to the kde plot. To prevent visual clutter,
                histograms are never plotted for random effects.
            bins (int): If hist is True, the number of bins in the histogram.
                Ignored if hist is False.
            kind (str): Either 'trace' (default) or 'priors'. If 'priors',
                this just internally calls Model.plot()
        '''

        def _plot_row(data, row, title, hist=True):
            # density plot
            axes[row, 0].set_title(title)
            if pma is not None:
                arr = np.atleast_2d(data.values.T).T
                pma.kdeplot_op(axes[row, 0], arr, bw = 4.5)
            else:
                data.plot(kind='kde', ax=axes[row, 0], legend=False)

            # trace plot
            axes[row, 1].set_title(title)
            data.plot(kind='line', ax=axes[row, 1], legend=False)
            # histogram
            if hist:
                if pma is not None:
                    pma.histplot_op(axes[row, 0], arr)
                else:
                    data.plot(kind='hist', ax=axes[row, 0], legend=False,
                              normed=True, bins=bins)

        if kind == 'priors':
            return self.model.plot(varnames)

        # count the total number of rows in the plot
        names = self._filter_names(varnames, ranefs, transformed)
        random = [re.sub(r'_offset$', '', x) in self.model.random_terms
                  for x in names]
        rows = sum([len(self.level_dict[p]) if not r else 1
                    for p, r in zip(names, random)])

        # make the plot!
        fig, axes = plt.subplots(rows, 2, figsize=(12, 2*rows))
        if rows == 1:
            axes = np.array([axes])  # For consistent 2D indexing

        _select_args = {'ranefs': ranefs, 'transformed': transformed}

        # Create list of chains (for combined, just one list w/ all chains)
        chains = list(range(self.n_chains))
        if combined:
            chains = [chains]

        for c in chains:

            row = 0

            for p in names:

                df = self[p].to_df(chains=c, **_select_args)
                # if p == 'floor|county_sd':

                # fixed effects
                if re.sub(r'_offset$', '', p) not in self.model.random_terms:
                    for lev in self.level_dict[p]:
                        lev_df = df[lev]
                        _plot_row(lev_df, row, lev, hist)
                        row += 1

                # random effects
                else:
                    _plot_row(df, row, p, hist=False)  # too much clutter
                    row += 1

        fig.tight_layout()

        # For bernoulli models, tell user which event is being modeled
        if self.model.family.name == 'bernoulli':
            event = next(i for i, x in enumerate(self.model.y.data.flatten())
                         if x > .99)
            warnings.warn('Modeling the probability that {}==\'{}\''.format(
                self.model.y.name,
                str(self.model.clean_data[self.model.y.name][event])))

        return axes

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

        index = ['hpd{}_{}'.format(width, x) for x in ['lower', 'upper']]
        return pd.Series([hdi_min, hdi_max], index=index)

    def summary(self, varnames=None, ranefs=False, transformed=False, hpd=.95,
                quantiles=None, diagnostics=['effective_n', 'gelman_rubin']):
        '''Returns a DataFrame of summary/diagnostic statistics for the parameters.

        Args:
            varnames (list): List of variable names to include; if None
                (default), all eligible variables are included.
            ranefs (bool): Whether or not to include random effects in the
                summary. Default is False.
            transformed (bool): Whether or not to include internally
                transformed variables in the summary. Default is False.
            hpd (float, between 0 and 1): Show Highest Posterior Density (HPD)
                intervals with specified width/proportion for all parameters.
                If None, HPD intervals are suppressed.
            quantiles (float, list): Show
                specified quantiles of the marginal posterior distributions for
                all parameters. If list, must be a list of floats between 0 and 1. If
                None (default), no quantiles are shown.
            diagnostics (list): List of functions to use to compute convergence
                diagnostics for all parameters. Each element can be either a
                callable or a string giving the name of a function in the
                diagnostics module. Valid strings are 'gelman_rubin' and
                'effective_n'. Functions must accept a MCMCResults object as
                the sole input, and return a DataFrame with one labeled row per
                parameter. If None, no convergence diagnostics are computed.
        '''

        samples = self.to_df(varnames, ranefs, transformed)

        # build the basic DataFrame
        df = pd.DataFrame({'mean': samples.mean(0), 'sd': samples.std(0)})

        # add user-specified quantiles
        if quantiles is not None:
            if not isinstance(quantiles, (list, tuple)):
                quantiles = [quantiles]
            qnames = ['q' + str(q) for q in quantiles]
            df = df.merge(samples.quantile(quantiles).set_index([qnames]).T,
                          left_index=True, right_index=True)

        # add HPD intervals
        if hpd is not None:
            df = df.merge(samples.apply(self._hpd_interval, axis=0,
                          width=hpd).T, left_index=True, right_index=True)

        # add convergence diagnostics
        if diagnostics is not None:
            _names = self._filter_names(ranefs=ranefs, transformed=transformed)
            _self = self[_names]
            if self.n_chains > 1:
                for diag in diagnostics:
                    if isinstance(diag, string_types):
                        diag = getattr(bmd, diag)
                    df = df.merge(diag(_self), left_index=True,
                                  right_index=True)
            else:
                warnings.warn('Multiple MCMC chains are required in order '
                              'to compute convergence diagnostics.')

        # For bernoulli models, tell user which event is being modeled
        if self.model.family.name == 'bernoulli':
            event = next(i for i, x in enumerate(self.model.y.data.flatten())
                         if x > .99)
            warnings.warn('Modeling the probability that {}==\'{}\''.format(
                self.model.y.name,
                str(self.model.clean_data[self.model.y.name][event])))

        return df

    def to_df(self, varnames=None, ranefs=False, transformed=False,
              chains=None):
        '''
        Returns the MCMC samples in a nice, neat pandas DataFrame with all MCMC chains
        concatenated.

        Args:
            varnames (list): List of variable names to include; if None
                (default), all eligible variables are included.
            ranefs (bool): Whether or not to include random effects in the
                returned DataFrame. Default is True.
            transformed (bool): Whether or not to include internally
                transformed variables in the result. Default is False.
            chains (int, list): Index, or list of indexes, of chains to
                concatenate. E.g., [1, 3] would concatenate the first and
                third chains, and ignore any others. If None (default),
                concatenates all available chains.
        '''

        # filter out unwanted variables
        names = self._filter_names(varnames, ranefs, transformed)

        # concatenate the (pre-sliced) chains
        if chains is None:
            chains = list(range(self.n_chains))
        chains = listify(chains)
        data = [self.data[:, i, :] for i in chains]
        data = np.concatenate(data, axis=0)

        # construct the trace DataFrame
        df = sum([self.level_dict[x] for x in names], [])
        df = pd.DataFrame({x: data[:, self.levels.index(x)] for x in df})

        return df


class PyMC3ADVIResults(ModelResults):

    '''Holds PyMC3 ADVI results and provides plotting and summarization tools.

    Attributes:
        model (Model): A bambi Model instance specifying the model.
        params (MultiTrace): ADVI parameters returned by PyMC3.
    '''

    def __init__(self, model, params):

        self.means = params['means']
        self.sds = params['stds']
        self.elbo_vals = params['elbo_vals']
        super(PyMC3ADVIResults, self).__init__(model)
