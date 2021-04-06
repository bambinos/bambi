# pylint: disable=no-name-in-module
# pylint: disable=too-many-lines
import re
import logging
from copy import deepcopy

import numpy as np
import pandas as pd
import pymc3 as pm

from arviz.plots import plot_posterior
from arviz.data import from_dict
from numpy.linalg import matrix_rank
from formulae import design_matrices

from .backends import PyMC3BackEnd
from .priors import Prior, PriorFactory, PriorScaler, Family
from .terms import ResponseTerm, Term, GroupSpecificTerm
from .utils import listify
from .version import __version__

_log = logging.getLogger("bambi")


class Model:
    """Specification of model class.

    Parameters
    ----------
    data : DataFrame or str
        The dataset to use. Either a pandas ``DataFrame``, or the name of the file containing
        the data, which will be passed to ``pd.read_csv()``.
    default_priors : dict or str
        An optional specification of the default priors to use for all model terms. Either a
        dictionary containing named distributions, families, and terms (see the documentation in
        ``priors.PriorFactory`` for details), or the name of a JSON file containing the same
        information.
    auto_scale : bool
        If ``True`` (default), priors are automatically rescaled to the data
        (to be weakly informative) any time default priors are used. Note that any priors
        explicitly set by the user will always take precedence over default priors.
    dropna : bool
        When ``True``, rows with any missing values in either the predictors or outcome are
        automatically dropped from the dataset in a listwise manner.
    taylor : int
        Order of Taylor expansion to use in approximate variance when constructing the default
        priors. Should be between 1 and 13. Lower values are less accurate, tending to undershoot
        the correct prior width, but are faster to compute and more stable. Odd-numbered values
        tend to work better. Defaults to 5 for Normal models and 1 for non-Normal models. Values
        higher than the defaults are generally not recommended as they can be unstable.
    noncentered : bool
        If ``True`` (default), uses a non-centered parameterization for normal hyperpriors on
        grouped parameters. If ``False``, naive (centered) parameterization is used.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        data=None,
        default_priors=None,
        auto_scale=True,
        dropna=False,
        taylor=None,
        noncentered=True,
    ):

        if isinstance(data, str):
            data = pd.read_csv(data, sep=None, engine="python")

        self.default_priors = PriorFactory(default_priors)

        obj_cols = data.select_dtypes(["object"]).columns
        data[obj_cols] = data[obj_cols].apply(lambda x: x.astype("category"))
        self.data = data
        self.reset()

        self.auto_scale = auto_scale
        self.dropna = dropna
        self.taylor = taylor
        self.noncentered = noncentered
        self._backend_name = None

        # _build() will loop over this, calling _set_priors()
        self._added_priors = {}

        # Design object returned by formulae.design_matrices()
        self._design = None

        # attributes that are set later
        self.formula = None
        self.response = None  # _add_response()
        self.family = None  # _add_response()
        self.backend = None  # _set_backend()
        self.dm_statistics = None  # _build()
        self._diagnostics = None  # _build()
        self.built = False  # _build()
        self.terms = {}

    def __str__(self):
        if self.backend is None:
            return ""

        priors = [f"  {term.name} ~ {term.prior}" for term in self.terms.values()]
        priors_extra_params = [
            f"  {k} ~ {v}"
            for k, v in self.family.prior.args.items()
            if k not in ["observed", self.family.parent]
        ]
        priors = priors + priors_extra_params
        foot = ["* To see a plot of the priors call the .plot_priors() method."]

        if self.backend.fit:
            foot_ = "* To see a summary or plot of the posterior pass the object returned by"
            foot_ += " .fit() to az.summary() or az.plot_trace()"
            foot += [foot_]

        str_list = [
            f"Formula: {self.formula}",
            f"Family name: {self.family.name.capitalize()}",
            f"Link: {self.family.link}",
            "Priors:",
            "\n".join(priors),
            "------",
        ]

        return "\n".join(str_list + foot)

    def __repr__(self):
        return self.__str__()

    def reset(self):
        """Reset list of terms and response variable."""
        self.formula = None
        self.terms = {}
        self.response = None
        self.backend = None
        self._added_priors = {}
        self._design = None

    def _set_backend(self, backend):
        backend = backend.lower()
        if backend.startswith("pymc"):
            self.backend = PyMC3BackEnd()
        else:
            raise ValueError("At the moment, only the PyMC3 backend is supported.")

        self._backend_name = backend

    def _build(self, backend="pymc"):
        """Set up the model for sampling/fitting.

        Performs any steps that require access to all model terms (e.g., scaling priors
        on each term), then calls the backend's ``_build()`` method.

        Parameters
        ----------
        backend : str
            The name of the backend to use for model fitting.
            Currently only ``'pymc'`` is supported.
        """

        # set custom priors
        self._set_priors(**self._added_priors)

        # prepare all priors
        for name, term in self.terms.items():
            type_ = (
                "intercept"
                if name == "Intercept"
                else "group_specific"
                if self.terms[name].group_specific
                else "common"
            )
            term.prior = self._prepare_prior(term.prior, type_)

        # check for backend
        if backend is None:
            if self._backend_name is None:
                raise ValueError(
                    "No backend was passed or set in the Model; did you forget to call fit()?"
                )
            backend = self._backend_name

        # check for outcome
        if self.response is None:
            raise ValueError(
                "No outcome variable is set! Please specify "
                "an outcome variable using the formula interface "
                "before _build() or fit()."
            )

        # Only compute the mean stats if there are multiple terms in the model
        terms = [t for t in self.common_terms.values() if t.name != "Intercept"]

        if len(self.common_terms) > 1:
            x_matrix = [pd.DataFrame(x.data, columns=x.levels) for x in terms]
            x_matrix = pd.concat(x_matrix, axis=1)
            self.dm_statistics = {"mean_x": x_matrix.mean(axis=0)}

        # throw informative error message if any categorical predictors have 1 category
        num_cats = [x.data.size for x in self.common_terms.values()]
        if any(np.array(num_cats) == 0):
            raise ValueError("At least one categorical predictor contains only 1 category!")

        # only set priors if there is at least one term in the model
        if self.terms:
            # Get and scale default priors if none are defined yet
            if self.taylor is not None:
                taylor = self.taylor
            else:
                taylor = 5 if self.family.name == "gaussian" else 1
            scaler = PriorScaler(self, taylor=taylor)
            scaler.scale()

        # Tell user which event is being modeled
        if self.family.name == "bernoulli":
            _log.info(
                "Modeling the probability that %s==%s",
                self.response.name,
                str(self.response.success_event),
            )

        self._set_backend(backend)
        self.backend.build(self)
        self.built = True

    def fit(
        self,
        formula=None,
        priors=None,
        family="gaussian",
        link=None,
        run=True,
        categorical=None,
        omit_offsets=True,
        backend="pymc",
        **kwargs,
    ):
        """Fit the model using the specified backend.

        Parameters
        ----------
        formula : str
            A model description written in model formula language.
        priors : dict
            Optional specification of priors for one or more terms. A dictionary where the keys are
            the names of terms in the model, 'common' or 'group_specific' and the values are either
            instances of class ``Prior`` or ``int``, ``float``, or ``str`` that specify the
            width of the priors on a standardized scale.
        family : str or Family
            A specification of the model family (analogous to the family object in R). Either
            a string, or an instance of class ``priors.Family``. If a string is passed, a family
            with the corresponding name must be defined in the defaults loaded at ``Model``
            initialization.Valid pre-defined families are ``'gaussian'``, ``'bernoulli'``,
            ``'poisson'``, ``'gamma'``, ``'wald'``, and ``'negativebinomial'``.
            Defaults to ``'gaussian'``.
        link : str
            The model link function to use. Can be either a string (must be one of the options
            defined in the current backend; typically this will include at least ``'identity'``,
            ``'logit'``, ``'inverse'``, and ``'log'``), or a callable that takes a 1D ndarray or
            theano tensor as the sole argument and returns one with the same shape.
        run : bool
            Whether or not to immediately begin fitting the model once any set up of passed
            arguments is complete. Defaults to ``True``.
        categorical : str or list
            The names of any variables to treat as categorical. Can be either a single variable
            name, or a list of names. If categorical is ``None``, the data type of the columns in
            the ``DataFrame`` will be used to infer handling. In cases where numeric columns are
            to be treated as categoricals (e.g., group specific factors coded as numerical IDs),
            explicitly passing variable names via this argument is recommended.
        omit_offsets: bool
            Omits offset terms in the ``InferenceData`` object when the model includes group
            specific effects. Defaults to ``True``.
        backend : str
            The name of the backend to use. Currently only ``'pymc'`` backend is supported.
        """

        if priors is None:
            priors = {}
        else:
            priors = deepcopy(priors)

        data = self.data
        # alter this pandas flag to avoid false positive SettingWithCopyWarnings
        data._is_copy = False  # pylint: disable=protected-access

        # Explicitly convert columns to category if desired--though this
        # can also be done within the formula using C().
        if categorical is not None:
            data = data.copy()
            cats = listify(categorical)
            data[cats] = data[cats].apply(lambda x: x.astype("category"))

        na_action = "drop" if self.dropna else "error"
        if formula is not None:
            # Only reset self.terms and self.response (e.g., keep priors)
            self.formula = formula
            self.terms = {}
            self.response = None
            self._design = design_matrices(formula, data, na_action, eval_env=1)
        else:
            if self._design is None:
                raise ValueError("Can't fit a model without a description of the model.")

        if self._design.response is not None:
            _family = family.name if isinstance(family, Family) else family
            self._add_response(self._design.response, family=_family, link=link)

        if self._design.common:
            self._add_common(self._design.common, priors)

        if self._design.group:
            self._add_group_specific(self._design.group, priors)

        if backend is None:
            backend = "pymc" if self._backend_name is None else self._backend_name

        if run:
            if not self.built or backend != self._backend_name:
                self._build(backend)
            return self.backend.run(omit_offsets=omit_offsets, **kwargs)

        self._backend_name = backend
        return None

    def _add_response(self, response, prior=None, family="gaussian", link=None):
        """Add a response (or outcome/dependent) variable to the model.

        Parameters
        ----------
        response : formulae.ResponseVector
            An instance of ``formulae.ResponseVector`` as returned by
            ``formulae.design_matrices()``.
        prior : Prior, int, float, str
            Optional specification of prior. Can be an instance of class ``Prior``, a numeric value,
            or a string describing the width. In the numeric case, the distribution specified in
            the defaults will be used, and the passed value will be used to scale the appropriate
            variance parameter. For strings (e.g., ``'wide'``, ``'narrow'``, ``'medium'``, or
            ``'superwide'``), predefined values will be used.
        family : str or Family
            A specification of the model family (analogous to the family object in R). Either a
            string, or an instance of class ``priors.Family``. If a string is passed, a family with
            the corresponding name must be defined in the defaults loaded at Model initialization.
            Valid pre-defined families are ``'gaussian'``, ``'bernoulli'``, ``'poisson'``,
            ``'gamma'``, ``'wald'``, and ``'negativebinomial'``. Defaults to ``'gaussian'``.
        link : str
            The model link function to use. Can be either a string (must be one of the options
            defined in the current backend; typically this will include at least ``'identity'``,
            ``'logit'``, ``'inverse'``, and ``'log'``), or a callable that takes a 1D ndarray or
            theano tensor as the sole argument and returns one with the same shape.
        """
        if isinstance(family, str):
            family = self.default_priors.get(family=family)
        self.family = family

        # Override family's link if another is explicitly passed
        if link is not None:
            self.family.link = link

        if prior is None:
            prior = self.family.prior

        if self.family.name == "gaussian":
            prior.update(sigma=Prior("HalfStudentT", nu=4, sigma=np.std(response.design_vector)))

        if response.refclass is not None and self.family.name != "bernoulli":
            raise ValueError("Index notation for response only available for 'bernoulli' family")

        self.response = ResponseTerm(response, prior, self.family.name)
        self.built = False

    def _add_common(self, common, priors):
        if matrix_rank(common.design_matrix) < common.design_matrix.shape[1]:
            raise ValueError(
                "Design matrix for common effects is not full-rank. "
                "Bambi does not support sparse settings yet."
            )

        for name, term in common.terms_info.items():
            data = common[name]
            prior = priors.pop(name, priors.get("common", None))
            if isinstance(prior, Prior):
                any_hyperprior = any(isinstance(x, Prior) for x in prior.args.values())
                if any_hyperprior:
                    raise ValueError(
                        f"Trying to set hyperprior on '{name}'. "
                        "Can't set a hyperprior on common effects."
                    )
            self.terms[name] = Term(name, term, data, prior)

    def _add_group_specific(self, group, priors):
        for name, term in group.terms_info.items():
            data = group[name]
            prior = priors.pop(name, priors.get("group_specific", None))
            self.terms[name] = GroupSpecificTerm(name, term, data, prior)

    def _match_derived_terms(self, name):
        """Return all (group_specific) terms whose named are derived from the specified string.

        For example, ``'condition|subject'`` should match the terms with names ``'1|subject'``,
        ``'condition[T.1]|subject'``, and so on.
        Only works for strings with grouping operator ``('|')``.
        """
        if "|" not in name:
            return None

        patt = r"^([01]+)*[\s\+]*([^\|]+)*\|(.*)"
        intcpt, pred, grpr = re.search(patt, name).groups()
        intcpt = f"1|{grpr}"
        if not pred:
            return [self.terms[intcpt]] if intcpt in self.terms else None

        source = f"{pred}|{grpr}"
        found = [
            t
            for (n, t) in self.terms.items()
            if n == intcpt or re.sub(r"(\[.*?\])", "", n) == source
        ]
        # If only the intercept matches, return None, because we want to err
        # on the side of caution and not consider '1|subject' to be a match for
        # 'condition|subject' if no slopes are found (e.g., the intercept could
        # have been set by some other specification like 'gender|subject').
        return found if found and (len(found) > 1 or found[0].name != intcpt) else None

    def set_priors(self, priors=None, common=None, group_specific=None, match_derived_names=True):
        """Set priors for one or more existing terms.

        Parameters
        ----------
        priors : dict
            Dictionary of priors to update. Keys are names of terms to update; values are the new
            priors (either a ``Prior`` instance, or an int or float that scales the default priors).
            Note that a tuple can be passed as the key, in which case the same prior will be applied
            to all terms named in the tuple.
        common : Prior, int, float or str
            A prior specification to apply to all common terms included in the model.
        group_specific : Prior, int, float or str
            A prior specification to apply to all group specific terms included in the model.
        match_derived_names : bool
            If ``True``, the specified prior(s) will be applied not only to terms that match the
            keyword exactly, but to the levels of group specific effects that were derived from the
            original specification with the passed name. For example,
            ``priors={'condition|subject':0.5}`` would apply the prior to the terms with names
            ``'1|subject'``, ``'condition[T.1]|subject'``, and so on. If ``False``, an exact
            match is required for the prior to be applied.
        """
        # save arguments to pass to _set_priors() at build time
        kwargs = dict(
            zip(
                ["priors", "common", "group_specific", "match_derived_names"],
                [priors, common, group_specific, match_derived_names],
            )
        )
        self._added_priors.update(kwargs)

        self.built = False

    def _set_priors(self, priors=None, common=None, group_specific=None, match_derived_names=True):
        """Internal version of ``set_priors()``, with same arguments.

        Runs during ``Model._build()``.
        """
        targets = {}

        if common is not None:
            targets.update({name: common for name in self.common_terms.keys()})

        if group_specific is not None:
            targets.update({name: group_specific for name in self.group_specific_terms.keys()})

        if priors is not None:
            for k, prior in priors.items():
                for name in listify(k):
                    term_names = list(self.terms.keys())
                    msg = f"No terms in model match {name}."
                    if name not in term_names:
                        terms = self._match_derived_terms(name)
                        if not match_derived_names or terms is None:
                            raise ValueError(msg)
                        for term in terms:
                            targets[term.name] = prior
                    else:
                        targets[name] = prior

        for name, prior in targets.items():
            self.terms[name].prior = prior

    def _prepare_prior(self, prior, _type):
        """Helper function to correctly set default priors, auto scaling, etc.

        Parameters
        ----------
        prior : Prior object, or float, or None.
        _type : string
            accepted values are: ``'intercept'``, ``'common'``, or ``'group_specific'``.
        """
        if prior is None and not self.auto_scale:
            prior = self.default_priors.get(term=_type + "_flat")

        if isinstance(prior, Prior):
            prior._auto_scale = False  # pylint: disable=protected-access
        else:
            _scale = prior
            prior = self.default_priors.get(term=_type)
            prior.scale = _scale
            if prior.scale is not None:
                prior._auto_scale = False  # pylint: disable=protected-access
        return prior

    def plot_priors(
        self,
        draws=5000,
        var_names=None,
        random_seed=None,
        figsize=None,
        textsize=None,
        hdi_prob=None,
        round_to=2,
        point_estimate="mean",
        kind="kde",
        bins=None,
        omit_offsets=True,
        omit_group_specific=True,
        ax=None,
    ):
        """
        Samples from the prior distribution and plots its marginals.

        Parameters
        ----------
        draws : int
            Number of draws to sample from the prior predictive distribution. Defaults to 5000.
        var_names : str or list
            A list of names of variables for which to compute the posterior predictive
            distribution. Defaults to both observed and unobserved RVs.
        random_seed : int
            Seed for the random number generator.
        figsize: tuple
            Figure size. If ``None`` it will be defined automatically.
        textsize: float
            Text size scaling factor for labels, titles and lines. If ``None`` it will be
            autoscaled based on ``figsize``.
        hdi_prob: float
            Plots highest density interval for chosen percentage of density.
            Use ``'hide'`` to hide the highest density interval. Defaults to 0.94.
        round_to: int
            Controls formatting of floats. Defaults to 2 or the integer part, whichever is bigger.
        point_estimate: str
            Plot point estimate per variable. Values should be ``'mean'``, ``'median'``, ``'mode'``
             or ``None``. Defaults to ``'auto'`` i.e. it falls back to default set in
             ArviZ's rcParams.
        kind: str
            Type of plot to display (``'kde'`` or ``'hist'``) For discrete variables this argument
            is ignored and a histogram is always used.
        bins: integer or sequence or 'auto'
            Controls the number of bins, accepts the same keywords ``matplotlib.hist()`` does.
            Only works if ``kind == hist``. If ``None`` (default) it will use ``auto`` for
            continuous variables and ``range(xmin, xmax + 1)`` for discrete variables.
        omit_offsets: bool
            Whether to omit offset terms in the plot. Defaults to ``True``.
        omit_group_specific: bool
            Whether to omit group specific effects in the plot. Defaults to ``True``.
        ax: numpy array-like of matplotlib axes or bokeh figures
            A 2D array of locations into which to plot the densities. If not supplied, ArviZ will
            create its own array of plot areas (and return it).
        **kwargs
            Passed as-is to ``plt.hist()`` or ``plt.plot()`` function depending on the value of
            ``kind``.

        Returns
        -------
        axes: matplotlib axes or bokeh figures
        """
        if not self.built:
            raise ValueError("Cannot plot priors until model is built!")

        unobserved_rvs_names = []
        flat_rvs = []
        for unobserved in self.backend.model.unobserved_RVs:
            if "Flat" in unobserved.__str__():
                flat_rvs.append(unobserved.name)
            else:
                unobserved_rvs_names.append(unobserved.name)
        if var_names is None:
            var_names = pm.util.get_default_varnames(
                unobserved_rvs_names, include_transformed=False
            )
        else:
            flat_rvs = [fv for fv in flat_rvs if fv in var_names]
            var_names = [vn for vn in var_names if vn not in flat_rvs]

        if flat_rvs:
            _log.info(
                "Variables %s have flat priors, and hence they are not plotted", ", ".join(flat_rvs)
            )

        if omit_offsets:
            omitted = [f"{rt}_offset" for rt in self.group_specific_terms]
            var_names = [vn for vn in var_names if vn not in omitted]

        if omit_group_specific:
            omitted = list(self.group_specific_terms)
            var_names = [vn for vn in var_names if vn not in omitted]

        axes = None
        if var_names:
            pps = self.prior_predictive(draws=draws, var_names=var_names, random_seed=random_seed)

            axes = plot_posterior(
                pps,
                group="prior",
                figsize=figsize,
                textsize=textsize,
                hdi_prob=hdi_prob,
                round_to=round_to,
                point_estimate=point_estimate,
                kind=kind,
                bins=bins,
                ax=ax,
            )
        return axes

    def prior_predictive(self, draws=500, var_names=None, omit_offsets=True, random_seed=None):
        """
        Generate samples from the prior predictive distribution.

        Parameters
        ----------
        draws : int
            Number of draws to sample from the prior predictive distribution. Defaults to 500.
        var_names : str or list
            A list of names of variables for which to compute the posterior predictive
            distribution. Defaults to both observed and unobserved RVs.
        random_seed : int
            Seed for the random number generator.

        Returns
        -------
        InferenceData
            ``InferenceData`` object with the groups `prior`, ``prior_predictive`` and
            ``observed_data``.
        """
        if var_names is None:
            variables = self.backend.model.unobserved_RVs + self.backend.model.observed_RVs
            variables_names = [v.name for v in variables]
            var_names = pm.util.get_default_varnames(variables_names, include_transformed=False)

        if omit_offsets:
            offset_vars = [f"{rt}_offset" for rt in self.group_specific_terms]
            var_names = [vn for vn in var_names if vn not in offset_vars]

        pps_ = pm.sample_prior_predictive(
            samples=draws, var_names=var_names, model=self.backend.model, random_seed=random_seed
        )
        # pps_ keys are not in the same order as `var_names` because `var_names` is converted
        # to set within pm.sample_prior_predictive()
        pps = {name: pps_[name] for name in var_names}

        response_name = self.response.name

        if response_name in pps:
            prior_predictive = {response_name: np.moveaxis(pps.pop(response_name), 2, 0)}
            observed_data = {response_name: self.response.data.squeeze()}
        else:
            prior_predictive = {}
            observed_data = {}

        prior = {k: v[np.newaxis] for k, v in pps.items()}

        idata = from_dict(
            prior_predictive=prior_predictive,
            prior=prior,
            observed_data=observed_data,
            coords=self.backend.model.coords,  # new line
            attrs={
                "inference_library": self.backend.name,
                "inference_library_version": self.backend.name,
                "modeling_interface": "bambi",
                "modeling_interface_version": __version__,
            },
        )

        return idata

    def posterior_predictive(
        self, idata, draws=500, var_names=None, inplace=True, random_seed=None
    ):
        """
        Generate samples from the posterior predictive distribution.

        Parameters
        ----------
        idata : InfereceData
            ``InfereceData`` with samples from the posterior distribution.
        draws : int
            Number of draws to sample from the prior predictive distribution. Defaults to 500.
        var_names : str or list
            A list of names of variables for which to compute the posterior predictive
            distribution. Defaults to both observed and unobserved RVs.
        inplace : bool
            If ``True`` it will add a ``posterior_predictive`` group to idata, otherwise it will
            return a copy of idata with the added group. If ``True`` and idata already have a
            ``posterior_predictive`` group it will be overwritten.
        random_seed : int
            Seed for the random number generator.

        Returns
        -------
        None or InferenceData
            When ``inplace=True`` add ``posterior_predictive`` group to idata and return
            ``None``. Otherwise a copy of idata with a ``posterior_predictive`` group.

        """
        if var_names is None:
            variables = self.backend.model.observed_RVs
            variables_names = [v.name for v in variables]
            var_names = pm.util.get_default_varnames(variables_names, include_transformed=False)

        pps = pm.sample_posterior_predictive(
            trace=idata,
            samples=draws,
            var_names=var_names,
            model=self.backend.model,
            random_seed=random_seed,
        )

        if not inplace:
            idata = deepcopy(idata)
        if "posterior_predictive" in idata:
            del idata.posterior_predictive

        idata.add_groups(
            {"posterior_predictive": {k: v.squeeze()[np.newaxis] for k, v in pps.items()}}
        )

        getattr(idata, "posterior_predictive").attrs["modeling_interface"] = "bambi"
        getattr(idata, "posterior_predictive").attrs["modeling_interface_version"] = __version__
        if inplace:
            return None
        else:
            return idata

    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """
        Produce a graphviz Digraph from a Bambi model.

        Requires graphviz, which may be installed most easily with
            ``conda install -c conda-forge python-graphviz``

        Alternatively, you may install the ``graphviz`` binaries yourself, and then
        ``pip install graphviz`` to get the python bindings.
        See http://graphviz.readthedocs.io/en/stable/manual.html for more information.

        Parameters
        ----------
        formatting : str
            One of ``'plain'`` or ``'plain_with_params'``. Defaults to ``'plain'``.
        name : str
            Name of the figure to save. Defaults to None, no figure is saved.
        figsize : tuple
            Maximum width and height of figure in inches. Defaults to None, the figure size is set
            automatically. If defined and the drawing is larger than the given size, the drawing is
            uniformly scaled down so that it fits within the given size.  Only works if ``name``
            is not None.
        dpi : int
            Point per inch of the figure to save.
            Defaults to 300. Only works if ``name`` is not None.
        fmt : str
            Format of the figure to save.
            Defaults to ``'png'``. Only works if ``name`` is not None.
        """
        if self.backend is None:
            raise ValueError("The model is empty, please define a Bambi model")

        graphviz = pm.model_to_graphviz(model=self.backend.model, formatting=formatting)

        width, height = (None, None) if figsize is None else figsize

        if name is not None:
            graphviz_ = graphviz.copy()
            graphviz_.graph_attr.update(size=f"{width},{height}!")
            graphviz_.graph_attr.update(dpi=str(dpi))
            graphviz_.render(filename=name, format=fmt, cleanup=True)

        return graphviz

    def _get_pymc_coords(self):
        # categorical attribute is important because of this coordinates stuff
        common_terms = {
            k + "_dim_0": v.cleaned_levels for k, v in self.common_terms.items() if v.categorical
        }
        # Include all group specific terms
        group_specific_terms = {
            k + "_dim_0": v.cleaned_levels for k, v in self.group_specific_terms.items()
        }
        return {**common_terms, **group_specific_terms}

    @property
    def term_names(self):
        """Return names of all terms in order of addition to model."""
        return list(self.terms.keys())

    @property
    def common_terms(self):
        """Return dict of all and only common effects in model."""
        return {k: v for (k, v) in self.terms.items() if not v.group_specific}

    @property
    def group_specific_terms(self):
        """Return dict of all and only group specific effects in model."""
        return {k: v for (k, v) in self.terms.items() if v.group_specific}
