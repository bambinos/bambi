import re
from abc import ABCMeta, abstractmethod

import arviz as az
import numpy as np
import pandas as pd
from bambi.external.six import string_types

from .utils import listify

try:
    import pymc3 as pm

    if hasattr(pm.plots, "kdeplot_op"):
        pma = pm.plots  # pylint: disable=invalid-name
    else:
        pma = pm.plots.artists  # pylint: disable=invalid-name, no-member
except ImportError:
    pma = None  # pylint: disable=invalid-name


__all__ = ["MCMCResults", "PyMC3ADVIResults"]


class ModelResults(metaclass=ABCMeta):
    """Base class for ModelResults hierarchy.

    Attributes:
        model (Model): a bambi Model instance specifying the model.
    """

    def __init__(self, model):

        self.model = model
        self.terms = list(model.terms.values())
        # pylint: disable=protected-access
        self.diagnostics = model._diagnostics if hasattr(model, "_diagnostics") else None

    @abstractmethod
    def plot(self):
        pass

    @abstractmethod
    def summary(self):
        pass


class MCMCResults(ModelResults):
    """Holds sampler results; provides slicing, plotting, and summarization tools.

    Attributes:
        model (Model): a bambi Model instance specifying the model.
        data (array-like): Raw storage of MCMC samples in array with
            dimensions 0, 1, 2 = samples, chains, variables
        names (list): Names of all Terms.
        dims (list): Numbers of levels for all Terms.
        levels (list): Names of all levels for all Terms.
        transformed (list): Optional list of variable names to treat as
            transformed--and hence, to exclude from the output by default.
    """

    # pylint: disable=too-many-instance-attributes
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
            level_dict[name] = levels[i : (i + dim)]
        self.level_dict = level_dict

        super(MCMCResults, self).__init__(model)

    def __getitem__(self, idx):
        """
        If a variable name, return MCMCResults with only that variable
            e.g., fit['subject']
        If a list of variable names, return MCMCResults with those variables
            e.g., fit[['subject','item]]
        If a slice, return MCMCResults with sliced samples
            e.g., fit[500:]
        If a tuple, return MCMCResults with those variables sliced
            e.g., fit[['subject','item'], 500:] OR fit[500:, ['subject','item']]
        """

        if isinstance(idx, slice):
            var = self.names
            vslice = idx
        elif isinstance(idx, string_types):
            var = [idx]
            vslice = slice(0, self.n_samples)
        elif isinstance(idx, list):
            if not all([isinstance(x, string_types) for x in idx]):
                raise ValueError("If passing a list, all elements must be " "parameter names.")
            var = idx
            vslice = slice(0, self.n_samples)
        elif isinstance(idx, tuple):
            if len(idx) > 2:
                raise ValueError(
                    "Only two arguments can be passed. If you "
                    "want to select multiple parameters and a "
                    "subset of samples, pass a slice and a list "
                    "of parameter names."
                )
            vslice = [i for i, x in enumerate(idx) if isinstance(x, slice)]
            if not len(vslice):
                raise ValueError(
                    "At least one argument must be a slice. If "
                    "you want to select multiple parameters by "
                    "name, pass a list (not a tuple) of names."
                )
            if len(vslice) > 1:
                raise ValueError("Slices can only be applied " "over the samples dimension.")
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
        return MCMCResults(
            model=self.model,
            data=self.data[vslice, :, level_iloc],
            names=var,
            dims=[self.dims[x] for x in var_iloc],
            levels=levels,
            transformed_vars=self.transformed_vars,
        )

    def get_chains(self, indices):
        # Return copy of self but only for chains with the passed indices
        if not isinstance(indices, (list, tuple)):
            indices = [indices]
        return MCMCResults(
            model=self.model,
            data=self.data[:, indices, :],
            names=self.names,
            dims=self.dims,
            levels=self.levels,
            transformed_vars=self.transformed_vars,
        )

    def _filter_names(self, var_names=None, ranefs=False, transformed=False):
        names = self.untransformed_vars if not transformed else self.names
        if var_names is not None:
            names = [n for n in listify(var_names) if n in names]
        if not ranefs:
            names = [x for x in names if re.sub(r"_offset$", "", x) not in self.model.random_terms]
        intercept = [n for n in names if "intercept" in n]
        std = [n for n in names if "_" in n]
        rand_eff = [n for n in names if "|" in n and n not in std]
        interac = [n for n in names if ":" in n and n not in rand_eff + std]
        main_eff = [n for n in names if n not in interac + std + rand_eff + intercept]
        names = (
            intercept
            + sorted(main_eff, key=len)
            + sorted(interac, key=len)
            + sorted(rand_eff, key=len)
            + sorted(std, key=len)
        )
        return names

    def plot(  # pylint: disable=arguments-differ, inconsistent-return-statements
        self,
        var_names=None,
        coords=None,
        ranefs=False,
        transformed=False,
        chains=None,
        divergences="bottom",
        figsize=None,
        textsize=None,
        lines=None,
        compact=False,
        combined=False,
        legend=False,
        kind="trace",
        plot_kwargs=None,
        fill_kwargs=None,
        rug_kwargs=None,
        hist_kwargs=None,
        trace_kwargs=None,
        max_plots=40,
    ):
        """Plot distribution (histogram or kernel density estimates) and sampled values.

        If `divergences` data is available in `sample_stats`, will plot the location of
        divergences as dashed vertical lines.

        Parameters
        ----------
        var_names : string, or list of strings
            One or more variables to be plotted.
        coords : mapping, optional
            Coordinates of var_names to be plotted. Passed to `Dataset.sel`
        ranefs : bool
            Whether or not to include random effects in the returned DataFrame. Defaults to True.
        transformed : bool
            Whether or not to include transformed variables in the result. Defaults to False.
        chains : int, list
            Index, or list of indexes, of chains to concatenate. E.g., [1, 3] would concatenate
            the first and third chains, and ignore any others. If None (default), concatenates all
            available chains.
        divergences : {"bottom", "top", None, False}
            Plot location of divergences on the traceplots. Options are "bottom", "top", or False-y.
        figsize : figure size tuple
            If None, size is (12, variables * 2)
        textsize: float
            Text size scaling factor for labels, titles and lines. If None it will be autoscaled
            based
            on figsize.
        lines : tuple
            Tuple of (var_name, {'coord': selection}, [line, positions]) to be overplotted as
            vertical lines on the density and horizontal lines on the trace.
        compact : bool
            Plot multidimensional variables in a single plot.
        combined : bool
            Flag for combining multiple chains into a single line. If False (default), chains will
            be plotted separately.
        legend : bool
            Add a legend to the figure with the chain color code.
        kind : str
             Either 'trace' (default) or 'priors'.
        plot_kwargs : dict
            Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
        fill_kwargs : dict
            Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
        rug_kwargs : dict
            Extra keyword arguments passed to `arviz.plot_dist`. Only affects continuous variables.
        hist_kwargs : dict
            Extra keyword arguments passed to `arviz.plot_dist`. Only affects discrete variables.
        trace_kwargs : dict
            Extra keyword arguments passed to `plt.plot`
        Returns
        -------
        axes : matplotlib axes
        """
        data = self.to_dict(
            var_names=var_names,
            ranefs=ranefs,
            transformed=transformed,
            chains=chains
        )

        if kind == "trace":
            axes = az.plot_trace(
                data,
                var_names=var_names,
                coords=coords,
                divergences=divergences,
                figsize=figsize,
                textsize=textsize,
                lines=lines,
                compact=compact,
                combined=combined,
                legend=legend,
                plot_kwargs=plot_kwargs,
                fill_kwargs=fill_kwargs,
                rug_kwargs=rug_kwargs,
                hist_kwargs=hist_kwargs,
                trace_kwargs=trace_kwargs,
                max_plots=max_plots,
            )
            return axes
        elif kind == "priors":
            return self.model.plot(var_names)

    def summary(  # pylint: disable=arguments-differ
        self,
        var_names=None,
        ranefs=False,
        transformed=False,
        chains=None,
        fmt="wide",
        round_to=None,
        include_circ=None,
        stat_funcs=None,
        extend=True,
        credible_interval=0.94,
        order="C",
        index_origin=0,
    ):
        """Create a data frame with summary statistics.

        Parameters
        ----------
        var_names : list
            Names of variables to include in summary
        ranefs : bool
            Whether or not to include random effects in the returned DataFrame. Defaults to True.
        transformed : bool
            Whether or not to include transformed variables in the result. Defaults to False.
        chains : int, list
            Index, or list of indexes, of chains to concatenate. E.g., [1, 3] would concatenate
            the first and third chains, and ignore any others. If None (default), concatenates all
            available chains.
        include_circ : bool
            Whether to include circular statistics
        fmt : {'wide', 'long', 'xarray'}
            Return format is either pandas.DataFrame {'wide', 'long'} or xarray.Dataset {'xarray'}.
        round_to : int
            Number of decimals used to round results. Defaults to 2. Use "none" to return raw
            numbers.
        stat_funcs : dict
            A list of functions or a dict of functions with function names as keys used to calculate
            statistics. By default, the mean, standard deviation, simulation standard error, and
            highest posterior density intervals are included.

            The functions will be given one argument, the samples for a variable as an nD array,
            The functions should be in the style of a ufunc and return a single number. For example,
            `np.mean`, or `scipy.stats.var` would both work.
        extend : boolean
            If True, use the statistics returned by `stat_funcs` in addition to, rather than in
            place of, the default statistics. This is only meaningful when `stat_funcs` is not None.
        credible_interval : float, optional
            Credible interval to plot. Defaults to 0.94. This is only meaningful when `stat_funcs`
            is None.
        order : {"C", "F"}
            If fmt is "wide", use either C or F unpacking order. Defaults to C.
        index_origin : int
            If fmt is "wide, select n-based indexing for multivariate parameters. Defaults to 0.

        Returns
        -------
        pandas.DataFrame
            With summary statistics for each variable. Defaults statistics are: `mean`, `sd`,
            `hpd_3%`, `hpd_97%`, `mcse_mean`, `mcse_sd`, `ess_bulk`, `ess_tail` and `r_hat`.
            `r_hat` is only computed for traces with 2 or more chains.
        """
        data = self.to_dict(
            var_names=var_names,
            ranefs=ranefs,
            transformed=transformed,
            chains=chains
        )

        return az.summary(
            data,
            var_names=var_names,
            fmt=fmt,
            round_to=round_to,
            include_circ=include_circ,
            stat_funcs=stat_funcs,
            extend=extend,
            credible_interval=credible_interval,
            order=order,
            index_origin=index_origin,
        )

    def to_df(self, var_names=None, ranefs=False, transformed=False, chains=None):
        """
        Returns the MCMC samples in a nice, neat pandas DataFrame with all MCMC chains
        concatenated.

        Parameters
        ----------
        var_names: list
            List of variable names to include; if None(default), all eligible variables are
            included.
        ranefs : bool)
            Whether or not to include random effects in the returned DataFrame. Default is True.
        transformed : bool
            Whether or not to include internally transformed variables in the result. Default is
            False.
        chains: int, list
            Index, or list of indexes, of chains to concatenate. E.g., [1, 3] would concatenate
            the first and third chains, and ignore any others. If None (default), concatenates all
            available chains.
        """

        # filter out unwanted variables
        names = self._filter_names(var_names, ranefs, transformed)

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

    def to_dict(self, var_names=None, ranefs=False, transformed=False, chains=None):
        """
        Returns the MCMC samples in a dictionary

        Parameters
        ----------
        var_names: list
            List of variable names to include; if None(default), all eligible variables are
            included.
        ranefs : bool)
            Whether or not to include random effects in the returned DataFrame. Default is True.
        transformed : bool
            Whether or not to include internally transformed variables in the result. Default is
            False.
        chains: int, list
            Index, or list of indexes, of chains to concatenate. E.g., [1, 3] would concatenate
            the first and third chains, and ignore any others. If None (default), concatenates all
            available chains.
        """

        # filter out unwanted variables
        names = self._filter_names(var_names, ranefs, transformed)

        # concatenate the (pre-sliced) chains
        if chains is None:
            chains = list(range(self.n_chains))
        chains = listify(chains)
        data = self.data.T
        return {name: data[idx] for idx, name in enumerate(names)}


class PyMC3ADVIResults(ModelResults):
    """Holds PyMC3 ADVI results and provides plotting and summarization tools.

    Attributes:
        model (Model): A bambi Model instance specifying the model.
        params (MultiTrace): ADVI parameters returned by PyMC3.
    """

    def __init__(self, model, params):

        self.means = params["means"]
        self.sds = params["stds"]
        self.elbo_vals = params["elbo_vals"]
        super(PyMC3ADVIResults, self).__init__(model)

    def plot(self):
        raise NotImplementedError

    def summary(self):
        raise NotImplementedError
