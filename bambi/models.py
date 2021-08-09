# pylint: disable=no-name-in-module
# pylint: disable=too-many-lines
import logging
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
import pymc3 as pm

from arviz.plots import plot_posterior
from arviz.data import from_dict
from numpy.linalg import matrix_rank
from formulae import design_matrices

from .backends import PyMC3BackEnd
from .defaults import get_default_prior, get_builtin_family
from .priors import Family, Prior, PriorScaler, PriorScalerMLE, extract_family_prior
from .terms import ResponseTerm, Term, GroupSpecificTerm
from .utils import listify, link_match_family
from .version import __version__

_log = logging.getLogger("bambi")


class Model:
    """Specification of model class.

    Parameters
    ----------
    formula : str
        A model description written in model formula language.
    data : DataFrame or str
        The dataset to use. Either a pandas ``DataFrame``, or the name of the file containing
        the data, which will be passed to ``pd.read_csv()``.
    family : str or Family
        A specification of the model family (analogous to the family object in R). Either
        a string, or an instance of class ``priors.Family``. If a string is passed, a family
        with the corresponding name must be defined in the defaults loaded at ``Model``
        initialization. Valid pre-defined families are ``'gaussian'``, ``'bernoulli'``, ``'beta'``,
        ``'binomial'``, ``'poisson'``, ``'gamma'``, ``'wald'``, and ``'negativebinomial'``.
        Defaults to ``'gaussian'``.
    priors : dict
        Optional specification of priors for one or more terms. A dictionary where the keys are
        the names of terms in the model, 'common' or 'group_specific' and the values are either
        instances of class ``Prior`` or ``int``, ``float``, or ``str`` that specify the
        width of the priors on a standardized scale.
    link : str
        The model link function to use. Can be either a string (must be one of the options
        defined in the current backend; typically this will include at least ``'identity'``,
        ``'logit'``, ``'inverse'``, and ``'log'``), or a callable that takes a 1D ndarray or
        theano tensor as the sole argument and returns one with the same shape.
    categorical : str or list
        The names of any variables to treat as categorical. Can be either a single variable
        name, or a list of names. If categorical is ``None``, the data type of the columns in
        the ``DataFrame`` will be used to infer handling. In cases where numeric columns are
        to be treated as categoricals (e.g., group specific factors coded as numerical IDs),
        explicitly passing variable names via this argument is recommended.
    potentials : A list of 2-tuples.
        Optional specification of potentials. A potential is an arbitrary expression added to the
        likelihood, this is generally useful to add constrains to models, that are difficult to
        express otherwise. The first term of a 2-tuple is the name of a variable in the model, the
        second a lambda function expressing the desired constraint.
        If a constraint involves n variables, you can pass n 2-tuples or pass a tuple which first
        element is a n-tuple and second element is a lambda function with n arguments. The number
        and order of the lambda function has to match the number and order of the variables names.
    dropna : bool
        When ``True``, rows with any missing values in either the predictors or outcome are
        automatically dropped from the dataset in a listwise manner.
    auto_scale : bool
        If ``True`` (default), priors are automatically rescaled to the data
        (to be weakly informative) any time default priors are used. Note that any priors
        explicitly set by the user will always take precedence over default priors.
    automatic_priors: str
        An optional specification to compute/scale automatic priors. ``"default"`` means to use
        a method inspired on the R rstanarm library. ``"mle"`` means to use old default priors in
        Bambi that rely on maximum likelihood estimations obtained via the statsmodels library.
    noncentered : bool
        If ``True`` (default), uses a non-centered parameterization for normal hyperpriors on
        grouped parameters. If ``False``, naive (centered) parameterization is used.
    priors_cor = dict
        The value of eta in the prior for the correlation matrix of group-specific terms.
        Keys in the dictionary indicate the groups, and values indicate the value of eta.
    taylor : int
        Order of Taylor expansion to use in approximate variance when constructing the default
        priors. Should be between 1 and 13. Lower values are less accurate, tending to undershoot
        the correct prior width, but are faster to compute and more stable. Odd-numbered values
        tend to work better. Defaults to 5 for Normal models and 1 for non-Normal models. Values
        higher than the defaults are generally not recommended as they can be unstable.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        formula=None,
        data=None,
        family="gaussian",
        priors=None,
        link=None,
        categorical=None,
        potentials=None,
        dropna=False,
        auto_scale=True,
        automatic_priors="default",
        noncentered=True,
        priors_cor=None,
        taylor=None,
    ):
        # attributes that are set later
        self.terms = {}
        self.built = False  # build()
        self._backend_name = None

        # build() will loop over this, calling _set_priors()
        self._added_priors = {}

        self._design = None
        self.formula = None
        self.response = None  # _add_response()
        self.family = None  # _add_response()
        self.backend = None  # _set_backend()
        self.priors_cor = {}  # _add_priors_cor()

        self.auto_scale = auto_scale
        self.dropna = dropna
        self.taylor = taylor
        self.noncentered = noncentered
        self.potentials = potentials

        # Read and clean data
        if isinstance(data, str):
            data = pd.read_csv(data, sep=None, engine="python")
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a string with a path to a .csv or a pandas DataFrame.")

        # To avoid SettingWithCopyWarning when converting object columns to category
        data._is_copy = False

        # Object columns converted to category by default.
        obj_cols = data.select_dtypes(["object"]).columns
        data[obj_cols] = data[obj_cols].apply(lambda x: x.astype("category"))

        # Explicitly convert columns to category if desired--though this
        # can also be done within the formula using C().
        if categorical is not None:
            data = data.copy()
            cats = listify(categorical)
            data[cats] = data[cats].apply(lambda x: x.astype("category"))

        self.data = data

        # Handle priors
        if priors is None:
            priors = {}
        else:
            priors = deepcopy(priors)

        self.automatic_priors = automatic_priors

        # Obtain design matrices and related objects.
        na_action = "drop" if dropna else "error"
        if formula is not None:
            self.formula = formula
            self._design = design_matrices(formula, data, na_action, eval_env=1)
        else:
            raise ValueError("Can't instantiate a model without a model formula.")

        if self._design.response is not None:
            family_prior = extract_family_prior(family, priors)
            if family_prior and self._design.common:
                conflict = [name for name in family_prior if name in self._design.common.terms_info]
                if conflict:
                    raise ValueError(
                        f"The prior name for {', '.join(conflict)} conflicts with the name of a "
                        "parameter in the response distribution.\n"
                        "Please rename the term(s) to prevent an unexpected behaviour."
                    )
            self._add_response(self._design.response, family, link, family_prior)
        else:
            raise ValueError(
                "No outcome variable is set! "
                "Please specify an outcome variable using the formula interface."
            )

        if self._design.common:
            self._add_common(self._design.common, priors)

        if self._design.group:
            self._add_group_specific(self._design.group, priors)

        if priors_cor:
            self._add_priors_cor(priors_cor)

        # Build priors
        self._build_priors()

    def fit(
        self,
        omit_offsets=True,
        **kwargs,
    ):
        """Fit the model using the specified backend.

        Parameters
        ----------
        omit_offsets: bool
            Omits offset terms in the ``InferenceData`` object when the model includes group
            specific effects. Defaults to ``True``.
        """

        if not self.built:
            self.build()

        # Tell user which event is being modeled
        if self.family.name == "bernoulli":
            _log.info(
                "Modeling the probability that %s==%s",
                self.response.name,
                str(self.response.success_event),
            )

        return self.backend.run(omit_offsets=omit_offsets, **kwargs)

    def build(self):
        """Set up the model for sampling/fitting.

        Performs any steps that require access to all model terms (e.g., scaling priors
        on each term), then calls the backend's ``build()`` method.
        """
        self.backend = PyMC3BackEnd()
        self.backend.build(self)
        self.built = True

    def set_priors(self, priors=None, common=None, group_specific=None):
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
        """
        # save arguments to pass to _set_priors() at build time
        kwargs = dict(zip(["priors", "common", "group_specific"], [priors, common, group_specific]))
        self._added_priors.update(kwargs)
        # After updating, we need to rebuild priors.
        # There is redundancy here, so there's place for performance improvements.
        self._build_priors()
        self.built = False

    def _build_priors(self):
        """Carry out all operations related to the construction and/or scaling of priors."""
        # Set custom priors that have been passed via `Model.set_priors()`
        self._set_priors(**self._added_priors)

        # Prepare all priors
        for term in self.terms.values():
            if term.group_specific:
                type_ = "group_specific"
            elif term.type == "intercept":
                type_ = "intercept"
            else:
                type_ = "common"
            term.prior = self._prepare_prior(term.prior, type_)

        # Scale priors if there is at least one term in the model and auto_scale is True
        if self.terms and self.auto_scale:
            method = self.automatic_priors
            if method == "default":
                scaler = PriorScaler(self)
            elif method == "mle":
                if self.taylor is not None:
                    taylor = self.taylor
                else:
                    taylor = 5 if self.family.name == "gaussian" else 1
                scaler = PriorScalerMLE(self, taylor=taylor)
            else:
                raise ValueError(
                    f"{method} is not a valid method for default priors." "Use 'default' or 'mle'."
                )
            self.scaler = scaler
            self.scaler.scale()

    def _set_priors(self, priors=None, common=None, group_specific=None):
        """Internal version of ``set_priors()``, with same arguments.

        Runs during ``Model._build_priors()``.
        """
        # First, it constructs a `targets` dict where it store key-value (name-prior) pairs that
        # are going to be updated. Finally, the update is done in the last for loop in this method.
        targets = {}

        if common is not None:
            targets.update({name: common for name in self.common_terms.keys()})

        if group_specific is not None:
            targets.update({name: group_specific for name in self.group_specific_terms.keys()})

        if priors is not None:
            # Prepare priors for response auxiliary parameters
            family_prior = extract_family_prior(self.family, priors)
            if family_prior:
                for prior in family_prior.values():
                    prior.auto_scale = False
                self.family.likelihood.priors.update(family_prior)

            # Prepare priors for explanatory terms.
            for names, prior in priors.items():
                # In case we have tuple-keys, we loop throuh each of them.
                for name in listify(names):
                    if name not in list(self.terms.keys()):
                        raise ValueError(f"No terms in model match {name}.")
                    targets[name] = prior

        # Set priors for explanatory terms.
        for name, prior in targets.items():
            self.terms[name].prior = prior

    def _prepare_prior(self, prior, type_):
        """Helper function to correctly set default priors, auto scaling, etc.

        Parameters
        ----------
        prior : Prior, float, or None.
        type_ : string
            Accepted values are: ``'intercept'``, ``'common'``, or ``'group_specific'``.
        """

        if prior is None and not self.auto_scale:
            prior = get_default_prior(type_ + "_flat")
        if isinstance(prior, Prior):
            prior.auto_scale = False
        else:
            _scale = prior
            prior = get_default_prior(type_)
            prior.scale = _scale
        return prior

    def _add_response(self, response, family="gaussian", link=None, priors=None):
        """Add a response (or outcome/dependent) variable to the model.

        Parameters
        ----------
        response : formulae.ResponseVector
            An instance of ``formulae.ResponseVector`` as returned by
            ``formulae.design_matrices()``.
        family : str or Family
            A specification of the model family (analogous to the family object in R). Either a
            string, or an instance of class ``priors.Family``. If a string is passed, a family with
            the corresponding name must be defined in the defaults loaded at Model initialization.
            Valid pre-defined families are ``'gaussian'``, ``'bernoulli'``, ``'beta'``,
            ``'binomial'``, ``'poisson'``, ``'gamma'``, ``'wald'``, and ``'negativebinomial'``.
            Defaults to ``'gaussian'``.
        link : str
            The model link function to use. Can be either a string (must be one of the options
            defined in the current backend; typically this will include at least ``'identity'``,
            ``'logit'``, ``'inverse'``, and ``'log'``), or a callable that takes a 1D ndarray or
            theano tensor as the sole argument and returns one with the same shape.
        priors : dict
            Optional dictionary with specification of priors for the parameters in the family of
            the response. Keys are names of other parameters than the mean in the family
            (i.e. they cannot be equal to family.parent) and values can be an instance of class
            ``Prior``, a numeric value, or a string describing the width. In the numeric case,
            the distribution specified in the defaults will be used, and the passed value will be
            used to scale the appropriate variance parameter. For strings (e.g., ``'wide'``,
            ``'narrow'``, ``'medium'``, or ``'superwide'``), predefined values will be used.
        """
        if isinstance(family, str):
            family = get_builtin_family(family)
        elif not isinstance(family, Family):
            raise ValueError("family must be a string or a Family object.")

        # Override family's link if another is explicitly passed
        if link is not None:
            if link_match_family(link, family.name):
                family._set_link(link)  # pylint: disable=protected-access
            else:
                raise ValueError(f"Link '{link}'' cannot be used with family '{family.name}'")

        # Update auxiliary parameters
        if priors:
            for prior in priors.values():
                if isinstance(prior, Prior):
                    prior.auto_scale = False
            family.likelihood.priors.update(priors)

        if response.success is not None and family.name != "bernoulli":
            raise ValueError("Index notation for response is only available for 'bernoulli' family")

        self.family = family
        self.response = ResponseTerm(response, family)
        self.built = False

    def _add_common(self, common, priors):
        """Add common (or fixed) terms to the model.

        Parameters
        ----------
        common : formulae.CommonEffectsMatrix
            Representation of the design matrix for the common effects of a model. It contains all
            the necessary information to build the ``Term`` objects associated with each common
            term in the model.
        priors : dict
            Optional specification of priors for one or more terms. A dictionary where the keys are
            any of the names of the common terms in the model or 'common' and the values are either
            instances of class ``Prior`` or ``int``, ``float``, or ``str`` that specify the width
            of the priors on a standardized scale.
        """
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
        """Add group-specific (or random) terms to the model.

        Parameters
        ----------
        group : formulae.GroupEffectsMatrix
            Representation of the design matrix for the group specific effects of a model. It
            contains all the necessary information to build the ``GroupSpecificTerm`` objects
            associated with each group-specific term in the model.
        priors : dict
            Optional specification of priors for one or more terms. A dictionary where the keys are
            any of the names of the group-specific terms in the model or 'group_specific' and the
            values are either instances of class ``Prior`` or ``int``, ``float``, or ``str``
            that specify the width of the priors on a standardized scale.
        """
        for name, term in group.terms_info.items():
            data = group[name]
            prior = priors.pop(name, priors.get("group_specific", None))
            self.terms[name] = GroupSpecificTerm(name, term, data, prior)

    def _add_priors_cor(self, priors):
        # priors: dictionary. names are groups, values are the "eta" in the lkj prior
        groups = self._get_group_specific_groups()
        for group in groups:
            if group in priors:
                self.priors_cor[group] = priors[group]
            else:
                raise KeyError(f"The name {group} is not a group in any group-specific term.")

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
            raise ValueError(
                "Cannot plot priors until model is built!! "
                "Call .build() to build the model or .fit() to build and sample from the posterior."
            )

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
            var_names = [name for name in var_names if not name.endswith("_offset")]

        if omit_group_specific:
            omitted = list(self.group_specific_terms)
            var_names = [vn for vn in var_names if vn not in omitted]

        axes = None
        if var_names:
            # Sort variable names so Intercept is in the beginning
            if "Intercept" in var_names:
                var_names.insert(0, var_names.pop(var_names.index("Intercept")))
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
            var_names = [name for name in var_names if not name.endswith("_offset")]

        pps_ = pm.sample_prior_predictive(
            samples=draws, var_names=var_names, model=self.backend.model, random_seed=random_seed
        )
        # pps_ keys are not in the same order as `var_names` because `var_names` is converted
        # to set within pm.sample_prior_predictive()

        pps = {}
        for name in var_names:
            if name in self.terms and self.terms[name].categorical:
                pps[name] = pps_[name]
            else:
                pps[name] = pps_[name].squeeze()

        response_name = self.response.name
        if response_name in pps:
            prior_predictive = {response_name: pps.pop(response_name)}
            observed_data = {response_name: self.response.data.squeeze()}
        else:
            prior_predictive = {}
            observed_data = {}

        prior = {k: v[np.newaxis] for k, v in pps.items()}

        coords = {}
        dims = {}
        for name in var_names:
            if name in self.terms:
                coords.update(**self.terms[name].pymc_coords)
                dims[name] = list(self.terms[name].pymc_coords.keys())

        idata = from_dict(
            prior_predictive=prior_predictive,
            prior=prior,
            observed_data=observed_data,
            coords=coords,
            dims=dims,
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
        idata : InferenceData
            ``InferenceData`` with samples from the posterior distribution.
        draws : int
            Number of draws to sample from the posterior predictive distribution. Defaults to 500.
        var_names : str or list
            A list of names of variables for which to compute the posterior predictive
            distribution. Defaults to observed RVs.
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

        warnings.warn(
            "Model.posterior_predictive() is deprecated. "
            "Use Model.predict() with kind='pps' instead.",
            FutureWarning,
        )
        if inplace:
            return None
        else:
            return idata

    # pylint: disable=protected-access
    def predict(self, idata, kind="mean", data=None, draws=None, inplace=True):
        """Predict method for Bambi models

        Obtains in-sample and out-sample predictions from a fitted Bambi model.

        Parameters
        ----------
        idata : InferenceData
            ``InferenceData`` with samples from the posterior distribution.
        kind: str
            Indicates the type of prediction required. Can be ``"mean"`` or ``"pps"``. The
            first returns posterior distribution of the mean, while the latter returns the posterior
            predictive distribution (i.e. the posterior probability distribution for a new
            observation). Defaults to ``"mean"``.
        data: pd.DataFrame or None
            An optional data frame in which to look for variables with which to predict.
            If omitted, the fitted linear predictors are used.
        draws: None
            The number of random draws per chain. Only used if ``kind="pps"``. Not recommended
            unless more than ndraws times nchains posterior predictive samples are needed.
            Defaults to ``None`` which means ndraws times nchains.
        inplace: bool
            If ``True`` it will add a ``posterior_predictive`` group to idata, otherwise it will
            return a copy of idata with the added group. If ``True`` and idata already have a
            ``posterior_predictive`` group it will be overwritten.

        Returns
        -------
        np.ndarray
            A NumPy array with predictions.
        """

        if kind not in ["mean", "pps"]:
            raise ValueError("'kind' must be one of 'mean' or 'pps'")

        linear_predictor = 0
        X = None
        Z = None

        chain_n = len(idata.posterior["chain"])
        draw_n = len(idata.posterior["draw"])
        posterior = idata.posterior.stack(sample=["chain", "draw"])

        if draws is None:
            draws = draw_n

        if not inplace:
            idata = deepcopy(idata)

        in_sample = data is None

        # Create design matrices
        if self._design.common:
            if in_sample:
                X = self._design.common.design_matrix
            else:
                X = self._design.common._evaluate_new_data(data).design_matrix

        if self._design.group:
            if in_sample:
                Z = self._design.group.design_matrix
            else:
                Z = self._design.group._evaluate_new_data(data).design_matrix

        # Obtain posterior and compute linear predictor
        if X is not None:
            beta_x_list = [np.atleast_2d(posterior[name]) for name in self.common_terms]
            if self.intercept_term:
                beta_x_list = [np.atleast_2d(posterior["Intercept"])] + beta_x_list
            beta_x = np.vstack(beta_x_list)
            linear_predictor += np.dot(X, beta_x)
        if Z is not None:
            beta_z = np.vstack(
                [np.atleast_2d(posterior[name]) for name in self.group_specific_terms]
            )
            linear_predictor += np.dot(Z, beta_z)

        # Compute mean prediction
        # Transposed so it is (chain, draws)?
        mu = self.family.link.linkinv(linear_predictor).T

        # Reshape mu
        obs_n = mu.size // (chain_n * draw_n)
        mu = mu.reshape((chain_n, draw_n, obs_n))

        # Predictions for the mean
        if kind == "mean":
            name = self.response.name + "_mean"
            coord_name = name + "_dim_0"

            # Drop var/dim if already present
            if name in idata.posterior.data_vars:
                idata.posterior = idata.posterior.drop_vars(name).drop_dims(coord_name)

            idata.posterior[name] = (("chain", "draw", coord_name), mu)
            idata.posterior = idata.posterior.assign_coords({coord_name: list(range(obs_n))})

        # Compute posterior predictive distribution
        else:
            # Sample mu values and auxiliary params
            if not in_sample and self.family.name == "binomial":
                n = self._design.response._evaluate_new_data(data)
                pps = self.family.likelihood.pps(self, idata.posterior, mu, draws, draw_n, trials=n)
            else:
                pps = self.family.likelihood.pps(self, idata.posterior, mu, draws, draw_n)

            if "posterior_predictive" in idata:
                del idata.posterior_predictive

            idata.add_groups({"posterior_predictive": {self.response.name: pps}})
            getattr(idata, "posterior_predictive").attrs["modeling_interface"] = "bambi"
            getattr(idata, "posterior_predictive").attrs["modeling_interface_version"] = __version__

        if inplace:
            return None
        else:
            return idata

    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """
        Produce a graphviz Digraph from a built Bambi model.

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

        Example
        --------
        >>> model = Model('y ~ x + (1|z)')
        >>> model.build()
        >>> model.graph()

        >>> model = Model('y ~ x + (1|z)')
        >>> model.fit()
        >>> model.graph()

        """
        if self.backend is None:
            raise ValueError(
                "The model is empty. "
                "Are you forgetting to first call .build() or .fit() on the Bambi model?"
            )

        graphviz = pm.model_to_graphviz(model=self.backend.model, formatting=formatting)

        width, height = (None, None) if figsize is None else figsize

        if name is not None:
            graphviz_ = graphviz.copy()
            graphviz_.graph_attr.update(size=f"{width},{height}!")
            graphviz_.graph_attr.update(dpi=str(dpi))
            graphviz_.render(filename=name, format=fmt, cleanup=True)

        return graphviz

    def _get_pymc_coords(self):
        coords = {}
        for term in self.terms.values():
            coords.update(**term.pymc_coords)
        return coords

    def _get_group_specific_groups(self):
        groups = {}
        for term_name in self.group_specific_terms:
            factor = term_name.split("|")[1]
            if factor not in groups:
                groups[factor] = [term_name]
            else:
                groups[factor].append(term_name)
        return groups

    def __str__(self):
        priors = ""
        priors_common = [f"    {t.name} ~ {t.prior}" for t in self.common_terms.values()]
        priors_group = [f"    {t.name} ~ {t.prior}" for t in self.group_specific_terms.values()]

        # Prior for the correlation matrix in group-specific terms
        priors_cor = [f"    {k} ~ LKJCorr({v})" for k, v in self.priors_cor.items()]

        # Priors for auxiliary parameters, e.g., standard deviation in normal linear model
        priors_aux = [f"    {k} ~ {v}" for k, v in self.family.likelihood.priors.items()]

        if self.intercept_term:
            t = self.intercept_term
            priors_common = [f"    {t.name} ~ {t.prior}"] + priors_common
        if priors_common:
            priors += "\n".join(["  Common-level effects", *priors_common]) + "\n\n"
        if priors_group:
            priors += "\n".join(["  Group-level effects", *priors_group]) + "\n\n"
        if priors_cor:
            priors += "\n".join(["  Group-level correlation", *priors_cor]) + "\n\n"
        if priors_aux:
            priors += "\n".join(["  Auxiliary parameters", *priors_aux]) + "\n\n"

        str_list = [
            f"Formula: {self.formula}",
            f"Family name: {self.family.name.capitalize()}",
            f"Link: {self.family.link.name}",
            f"Observations: {self.response.data.shape[0]}",
            "Priors:",
            priors,
        ]
        if self.backend and self.backend.fit:
            extra_foot = "------\n"
            extra_foot += "* To see a plot of the priors call the .plot_priors() method.\n"
            extra_foot += "* To see a summary or plot of the posterior pass the object returned"
            extra_foot += " by .fit() to az.summary() or az.plot_trace()\n"
            str_list += [extra_foot]

        return "\n".join(str_list)

    def __repr__(self):
        return self.__str__()

    @property
    def term_names(self):
        """Return names of all terms in order of addition to model."""
        return list(self.terms.keys())

    @property
    def common_terms(self):
        """Return dict of all and only common effects in model."""
        return {
            k: v for (k, v) in self.terms.items() if not v.group_specific and v.type != "intercept"
        }

    @property
    def group_specific_terms(self):
        """Return dict of all and only group specific effects in model."""
        return {k: v for (k, v) in self.terms.items() if v.group_specific}

    @property
    def intercept_term(self):
        """Return the intercept term"""
        term = [v for v in self.terms.values() if not v.group_specific and v.type == "intercept"]
        if term:
            return term[0]
        else:
            return None
