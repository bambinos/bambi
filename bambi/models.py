# pylint: disable=no-name-in-module
# pylint: disable=too-many-lines
import logging
import warnings

from copy import deepcopy

import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from arviz.plots import plot_posterior

from formulae import design_matrices

from bambi.backend import PyMCModel
from bambi.defaults import get_default_prior, get_builtin_family
from bambi.families import Family, univariate, multivariate
from bambi.priors import Prior, PriorScaler
from bambi.terms import CommonTerm, GroupSpecificTerm, OffsetTerm, ResponseTerm
from bambi.utils import listify, extra_namespace
from bambi.version import __version__

_log = logging.getLogger("bambi")


class Model:
    """Specification of model class.

    Parameters
    ----------
    formula : str
        A model description written using the formula syntax from the ``formulae`` library.
    data : pandas.DataFrame
        A pandas dataframe containing the data on which the model will be fit, with column
        names matching variables defined in the formula.
    family : str or bambi.families.Family
        A specification of the model family (analogous to the family object in R). Either
        a string, or an instance of class ``bambi.families.Family``. If a string is passed, a
        family with the corresponding name must be defined in the defaults loaded at ``Model``
        initialization. Valid pre-defined families are ``"bernoulli"``, ``"beta"``,
        ``"binomial"``, ``"categorical"``, ``"gamma"``, ``"gaussian"``, ``"negativebinomial"``,
        ``"poisson"``, ``"t"``, and ``"wald"``. Defaults to ``"gaussian"``.
    priors : dict
        Optional specification of priors for one or more terms. A dictionary where the keys are
        the names of terms in the model, "common," or "group_specific" and the values are
        instances of class ``Prior``. If priors are unset, uses automatic priors inspired by
        the R rstanarm library.
    link : str
        The name of the link function to use. Valid names are ``"cloglog"``, ``"identity"``,
        ``"inverse_squared"``, ``"inverse"``, ``"log"``, ``"logit"``, ``"probit"``, and
        ``"softmax"``. Not all the link functions can be used with all the families.
    categorical : str or list
        The names of any variables to treat as categorical. Can be either a single variable
        name, or a list of names. If categorical is ``None``, the data type of the columns in
        the ``data`` will be used to infer handling. In cases where numeric columns are
        to be treated as categorical (e.g., group specific factors coded as numerical IDs),
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
    noncentered : bool
        If ``True`` (default), uses a non-centered parameterization for normal hyperpriors on
        grouped parameters. If ``False``, naive (centered) parameterization is used.
    priors_cor : dict
        An optional value for eta in the LKJ prior for the correlation matrix of group-specific
        terms. Keys in the dictionary indicate the groups, and values indicate the value of eta.
        This is a very experimental feature. Defaults to ``None``, which means priors for the
        group-specific terms are independent.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        formula,
        data,
        family="gaussian",
        priors=None,
        link=None,
        categorical=None,
        potentials=None,
        dropna=False,
        auto_scale=True,
        noncentered=True,
        priors_cor=None,
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
        self.noncentered = noncentered
        self.potentials = potentials

        # Read and clean data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame.")

        # Convert 'object' and explicitly asked columns to categorical.
        object_cols = list(data.select_dtypes("object").columns)
        cols_to_convert = list(set(object_cols + listify(categorical)))
        if cols_to_convert:
            data = data.copy()  # don't modify original data frame
            data[cols_to_convert] = data[cols_to_convert].apply(lambda x: x.astype("category"))

        self.data = data

        # Handle priors
        if priors is None:
            priors = {}
        else:
            priors = deepcopy(priors)

        # Obtain design matrices and related objects.
        na_action = "drop" if dropna else "error"
        self.formula = formula
        self._design = design_matrices(formula, data, na_action, 1, extra_namespace)

        if self._design.response is None:
            raise ValueError(
                "No outcome variable is set! "
                "Please specify an outcome variable using the formula interface."
            )

        self._set_family(family, link, priors)
        self._add_response(self._design.response)

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
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        include_mean=False,
        inference_method="mcmc",
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        **kwargs,
    ):
        """Fit the model using PyMC.

        Parameters
        ----------
        draws: int
            The number of samples to draw from the posterior distribution. Defaults to 1000.
        tune : int
            Number of iterations to tune. Defaults to 1000. Samplers adjust the step sizes,
            scalings or similar during tuning. These tuning samples are be drawn in addition to the
            number specified in the ``draws`` argument, and will be discarded unless
            ``discard_tuned_samples`` is set to ``False``.
        discard_tuned_samples : bool
            Whether to discard posterior samples of the tune interval. Defaults to ``True``.
        omit_offsets: bool
            Omits offset terms in the ``InferenceData`` object returned when the model includes
            group specific effects. Defaults to ``True``.
        include_mean: bool
            Compute the posterior of the mean response. Defaults to ``False``.
        inference_method: str
            The method to use for fitting the model. By default, ``"mcmc"``. This automatically
            assigns a MCMC method best suited for each kind of variables, like NUTS for continuous
            variables and Metropolis for non-binary discrete ones. Alternatively, ``"vi"``, in
            which case the model will be fitted using variational inference as implemented in PyMC
            using the ``fit`` function.
            Finally, ``"laplace"``, in which case a Laplace approximation is used and is not
            recommended other than for pedagogical use.
            To use the PyMC numpyro and blackjax samplers, use ``nuts_numpyro`` or ``nuts_blackjax``
            respectively. Both methods will only work if you can use NUTS sampling, so your model
            must be differentiable.
        init: str
            Initialization method. Defaults to ``"auto"``. The available methods are:
            * auto: Use ``"jitter+adapt_diag"`` and if this method fails it uses ``"adapt_diag"``.
            * adapt_diag: Start with a identity mass matrix and then adapt a diagonal based on the
            variance of the tuning samples. All chains use the test value (usually the prior mean)
            as starting point.
            * jitter+adapt_diag: Same as ``"adapt_diag"``, but use test value plus a uniform jitter
            in [-1, 1] as starting point in each chain.
            * advi+adapt_diag: Run ADVI and then adapt the resulting diagonal mass matrix based on
            the sample variance of the tuning samples.
            * advi+adapt_diag_grad: Run ADVI and then adapt the resulting diagonal mass matrix based
            on the variance of the gradients during tuning. This is **experimental** and might be
            removed in a future release.
            * advi: Run ADVI to estimate posterior mean and diagonal mass matrix.
            * advi_map: Initialize ADVI with MAP and use MAP as starting point.
            * map: Use the MAP as starting point. This is strongly discouraged.
            * adapt_full: Adapt a dense mass matrix using the sample covariances. All chains use the
            test value (usually the prior mean) as starting point.
            * jitter+adapt_full: Same as ``"adapt_full"``, but use test value plus a uniform jitter
            in [-1, 1] as starting point in each chain.
        n_init: int
            Number of initialization iterations. Only works for ``"advi"`` init methods.
        chains: int
            The number of chains to sample. Running independent chains is important for some
            convergence statistics and can also reveal multiple modes in the posterior. If ``None``,
            then set to either ``cores`` or 2, whichever is larger.
        cores : int
            The number of chains to run in parallel. If ``None``, it is equal to the number of CPUs
            in the system unless there are more than 4 CPUs, in which case it is set to 4.
        random_seed : int or list of ints
            A list is accepted if cores is greater than one.
        **kwargs:
            For other kwargs see the documentation for ``PyMC.sample()``.

        Returns
        -------
        An ArviZ ``InferenceData`` instance if inference_method is  ``"mcmc"`` (default),
        "nuts_numpyro", "nuts_blackjax" or "laplace".
        An ``Approximation`` object if  ``"vi"``.
        """
        method = kwargs.pop("method", None)
        if method is not None:
            if inference_method == "vi":
                kwargs["method"] = method
            else:
                warnings.warn(
                    "the method argument has been deprecated, please use inference_method",
                    FutureWarning,
                )
                inference_method = method

        if not self.built:
            self.build()

        # Tell user which event is being modeled
        if isinstance(self.family, univariate.Bernoulli):
            _log.info(
                "Modeling the probability that %s==%s",
                self.response.name,
                str(self.response.success),
            )

        return self.backend.run(
            draws=draws,
            tune=tune,
            discard_tuned_samples=discard_tuned_samples,
            omit_offsets=omit_offsets,
            include_mean=include_mean,
            inference_method=inference_method,
            init=init,
            n_init=n_init,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            **kwargs,
        )

    def build(self):
        """Set up the model for sampling/fitting.

        Creates an instance of the underlying PyMC model and adds all the necessary terms to it.
        """
        self.backend = PyMCModel()
        self.backend.build(self)
        self.built = True

    def set_priors(self, priors=None, common=None, group_specific=None):
        """Set priors for one or more existing terms.

        Parameters
        ----------
        priors: dict
            Dictionary of priors to update. Keys are names of terms to update; values are the new
            priors (either a ``Prior`` instance, or an int or float that scales the default priors).
            Note that a tuple can be passed as the key, in which case the same prior will be applied
            to all terms named in the tuple.
        common: Prior, int, float or str
            A prior specification to apply to all common terms included in the model.
        group_specific: Prior, int, float or str
            A prior specification to apply to all group specific terms included in the model.
        """
        kwargs = dict(zip(["priors", "common", "group_specific"], [priors, common, group_specific]))
        self._added_priors.update(kwargs)
        # After updating, we need to rebuild priors.
        self._build_priors()
        self.built = False

    def _build_priors(self):
        """Carry out all operations related to the construction and/or scaling of priors."""
        # Set custom priors that have been passed via `Model.set_priors()`
        self._set_priors(**self._added_priors)

        # Prepare all priors
        for term in self.terms.values():
            if isinstance(term, GroupSpecificTerm):
                kind = "group_specific"
            elif isinstance(term, CommonTerm) and term.kind == "intercept":
                kind = "intercept"
            elif term.kind == "offset":
                continue
            else:
                kind = "common"
            term.prior = prepare_prior(term.prior, kind, self.auto_scale)

        # Scale priors if there is at least one term in the model and auto_scale is True
        if self.terms and self.auto_scale:
            self.scaler = PriorScaler(self)
            self.scaler.scale()

    def _set_priors(self, priors=None, common=None, group_specific=None):
        """Internal version of ``set_priors()``, with same arguments.

        Runs during ``Model._build_priors()``.
        """
        # First, it constructs a `targets` dict where it store key-value (name-prior) pairs that
        # are going to be updated. Finally, the update is done in the last for loop in this method.
        targets = {}

        if common is not None:
            targets.update({name: common for name in self.common_terms})

        if group_specific is not None:
            targets.update({name: group_specific for name in self.group_specific_terms})

        if priors is not None:
            priors = deepcopy(priors)
            # Prepare priors for auxiliary parameters of the response
            family_prior = {
                name: priors.pop(name) for name in self.family.likelihood.priors if name in priors
            }
            if family_prior:
                for prior in family_prior.values():
                    prior.auto_scale = False
                self.family.likelihood.priors.update(family_prior)

            # Prepare priors for explanatory terms.
            for names, prior in priors.items():
                # In case we have tuple-keys, we loop through each of them.
                for name in listify(names):
                    if name not in self.terms:
                        raise ValueError(f"No terms in model match '{name}'.")
                    targets[name] = prior

        # Set priors for explanatory terms.
        for name, prior in targets.items():
            self.terms[name].prior = prior

    def _set_family(self, family, link, priors):
        """Set the Family of the model.

        Parameters
        ----------
        family: str or bambi.families.Family
            A specification of the model family (analogous to the family object in R). Either a
            string, or an instance of class ``families.Family``. If a string is passed, a family
            with the corresponding name must be defined in the defaults loaded at model
            initialization.
        link: str
            The name of the link function to use. Valid names are ``"cloglog"``, ``"identity"``,
            ``"inverse_squared"``, ``"inverse"``, ``"log"``, ``"logit"``, ``"probit"``, and
            ``"softmax"``. Not all the link functions can be used with all the families.
        priors: dict
            A dictionary with specification of priors for the parameters in the family of
            the response. Keys are names of other parameters than the mean in the family
            (i.e. they cannot be equal to ``family.parent``) and values can be an instance of class
            ``Prior``, a numeric value, or a string describing the width. In the numeric case,
            the distribution specified in the defaults will be used, and the passed value will be
            used to scale the appropriate variance parameter.
        """

        # If string, get builtin family
        if isinstance(family, str):
            family = get_builtin_family(family)

        # Always ensure family is indeed instance of Family
        if not isinstance(family, Family):
            raise ValueError("'family' must be a string or a Family object.")

        # Get the names of the auxiliary parameters in the family
        aux_params = list(family.likelihood.priors)

        # Check if any of the names of the auxiliary params match the names of terms in the model
        # If that happens, raise an error.
        if aux_params and self._design.common:
            conflicts = [name for name in aux_params if name in self._design.common.terms]
            if conflicts:
                raise ValueError(
                    f"The prior name for {', '.join(conflicts)} conflicts with the name of a "
                    "parameter in the response distribution.\n"
                    "Please rename the term(s) to prevent an unexpected behaviour."
                )

        # Extract priors for auxiliary params
        priors = {k: v for k, v in priors.items() if k in aux_params}

        # Update auxiliary parameters
        if priors:
            for prior in priors.values():
                if isinstance(prior, Prior):
                    prior.auto_scale = False
            family.likelihood.priors.update(priors)

        # Override family's link if another is explicitly passed
        if link is not None:
            family.link = link

        self.family = family

    def set_alias(self, aliases):
        """Set aliases for the terms and auxiliary parameters in the model

        Parameters
        ----------
        aliases: dict
            A dictionary where key represents the original term name and the value is the alias.
        """
        if not isinstance(aliases, dict):
            raise ValueError(f"'aliases' must be a dictionary, not a {type(aliases)}.")

        for name, alias in aliases.items():
            if name in self.terms:
                self.terms[name].alias = alias
            if name in self.family.likelihood.priors:
                self.family.set_alias(name, alias)
            if name == self.response.name:
                self.response.alias = alias

            # Now add aliases for hyperpriors in group specific terms
            for term in self.group_specific_terms.values():
                if name in term.prior.args:
                    term.hyperprior_alias = {name: alias}

        # Model needs to be rebuilt after modifying aliases
        self.built = False

    def _add_response(self, response):
        """Add a response (or outcome/dependent) variable to the model.

        Parameters
        ----------
        response : formulae.ResponseMatrix
            An instance of ``formulae.ResponseMatrix`` as returned by
            ``formulae.design_matrices()``.
        """

        if hasattr(response.term.term.components[0], "reference"):
            reference = response.term.term.components[0].reference
        else:
            reference = None

        if reference is not None and not isinstance(self.family, univariate.Bernoulli):
            raise ValueError("Index notation for response is only available for 'bernoulli' family")

        if isinstance(self.family, univariate.Bernoulli):
            if response.kind == "categoric" and response.levels is None and reference is None:
                raise ValueError("Categoric response must be binary for 'bernoulli' family.")
            if response.kind == "numeric" and not all(np.isin(response.design_matrix, [0, 1])):
                raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family.")

        self.response = ResponseTerm(response, self.family)
        self.built = False

    def _add_common(self, common, priors):
        """Add common (a.k.a. fixed) terms to the model.

        Parameters
        ----------
        common: formulae.CommonEffectsMatrix
            Representation of the design matrix for the common effects of a model. It contains all
            the necessary information to build the ``Term`` objects associated with each common
            term in the model.
        priors: dict
            Optional specification of priors for one or more terms. A dictionary where the keys are
            any of the names of the common terms in the model or ``"common"`` and the values are
            either instances of class ``Prior`` or ``int``, ``float``, or ``str`` that specify the
            width of the priors on a standardized scale.
        """
        for name, term in common.terms.items():
            data = common[name]
            prior = priors.pop(name, priors.get("common", None))
            if isinstance(prior, Prior):
                any_hyperprior = any(isinstance(x, Prior) for x in prior.args.values())
                if any_hyperprior:
                    raise ValueError(
                        f"Trying to set hyperprior on '{name}'. "
                        "Can't set a hyperprior on common effects."
                    )
            if term.kind == "offset":
                self.terms[name] = OffsetTerm(name, term, data)
            else:
                self.terms[name] = CommonTerm(term, prior)

    def _add_group_specific(self, group, priors):
        """Add group-specific (a.k.a. random) terms to the model.

        Parameters
        ----------
        group: formulae.GroupEffectsMatrix
            Representation of the design matrix for the group specific effects of a model. It
            contains all the necessary information to build the ``GroupSpecificTerm`` objects
            associated with each group-specific term in the model.
        priors: dict
            Optional specification of priors for one or more terms. A dictionary where the keys are
            any of the names of the group-specific terms in the model or ``"group_specific"`` and
            the values are either instances of class ``Prior`` or ``int``, ``float``, or ``str``
            that specify the width of the priors on a standardized scale.
        """
        for name, term in group.terms.items():
            prior = priors.pop(name, priors.get("group_specific", None))
            self.terms[name] = GroupSpecificTerm(term, prior)

    def _add_priors_cor(self, priors):
        # priors: dictionary. names are groups, values are the "eta" in the lkj prior
        groups = self._get_group_specific_groups()
        for group in groups:
            if group in priors:
                self.priors_cor[group] = priors[group]
            else:
                raise KeyError(f"The name {group} is not a group in any group-specific term.")

    def _check_built(self):
        # Checks if model is built, raises ValueError if not
        if not self.built:
            raise ValueError(
                "Model is not built yet! "
                "Call .build() to build the model or .fit() to build and sample from the posterior."
            )

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
        draws: int
            Number of draws to sample from the prior predictive distribution. Defaults to 5000.
        var_names: str or list
            A list of names of variables for which to compute the posterior predictive
            distribution. Defaults to ``None`` which means to include both observed and
            unobserved RVs.
        random_seed: int
            Seed for the random number generator.
        figsize: tuple
            Figure size. If ``None`` it will be defined automatically.
        textsize: float
            Text size scaling factor for labels, titles and lines. If ``None`` it will be
            autoscaled based on ``figsize``.
        hdi_prob: float or str
            Plots highest density interval for chosen percentage of density.
            Use ``"hide"`` to hide the highest density interval. Defaults to 0.94.
        round_to: int
            Controls formatting of floats. Defaults to 2 or the integer part, whichever is bigger.
        point_estimate: str
            Plot point estimate per variable. Values should be ``"mean"``, ``"median"``, ``"mode"``
            or ``None``. Defaults to ``"auto"`` i.e. it falls back to default set in
            ArviZ's rcParams.
        kind: str
            Type of plot to display (``"kde"`` or ``"hist"``) For discrete variables this argument
            is ignored and a histogram is always used.
        bins: integer or sequence or "auto"
            Controls the number of bins, accepts the same keywords ``matplotlib.pyplot.hist()``
            does. Only works if ``kind == "hist"``. If ``None`` (default) it will use ``"auto"``
            for continuous variables and ``range(xmin, xmax + 1)`` for discrete variables.
        omit_offsets: bool
            Whether to omit offset terms in the plot. Defaults to ``True``.
        omit_group_specific: bool
            Whether to omit group specific effects in the plot. Defaults to ``True``.
        ax: numpy array-like of matplotlib axes or bokeh figures
            A 2D array of locations into which to plot the densities. If not supplied, ArviZ will
            create its own array of plot areas (and return it).
        **kwargs
            Passed as-is to ``matplotlib.pyplot.hist()`` or ``matplotlib.pyplot.plot()`` function
            depending on the value of ``kind``.

        Returns
        -------
        axes: matplotlib axes
        """
        self._check_built()

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
            var_names = [vn for vn in var_names if vn not in self.group_specific_terms]

        axes = None
        if var_names:
            # Sort variable names so Intercept is in the beginning
            if "Intercept" in var_names:
                var_names.insert(0, var_names.pop(var_names.index("Intercept")))
            pps = self.prior_predictive(draws=draws, var_names=var_names, random_seed=random_seed)

            axes = plot_posterior(
                pps,
                group="prior",
                var_names=var_names,
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
        draws: int
            Number of draws to sample from the prior predictive distribution. Defaults to 500.
        var_names: str or list
            A list of names of variables for which to compute the prior predictive distribution.
            Defaults to ``None`` which means both observed and unobserved RVs.
        random_seed: int
            Seed for the random number generator.
        Returns
        -------
        InferenceData
            ``InferenceData`` object with the groups ``prior``, ``prior_predictive`` and
            ``observed_data``.
        """
        self._check_built()

        if var_names is None:
            variables = self.backend.model.unobserved_RVs + self.backend.model.observed_RVs
            variables_names = [v.name for v in variables]
            var_names = pm.util.get_default_varnames(variables_names, include_transformed=False)

        if omit_offsets:
            var_names = [name for name in var_names if not name.endswith("_offset")]

        idata = pm.sample_prior_predictive(
            samples=draws, var_names=var_names, model=self.backend.model, random_seed=random_seed
        )

        if hasattr(idata, "prior"):
            to_drop = [dim for dim in idata.prior.dims if dim.endswith("_dim_0")]
            idata.prior = idata.prior.squeeze(to_drop).reset_coords(to_drop, drop=True)

        for group in idata.groups():
            getattr(idata, group).attrs["modeling_interface"] = "bambi"
            getattr(idata, group).attrs["modeling_interface_version"] = __version__

        return idata

    def predict(self, idata, kind="mean", data=None, inplace=True, include_group_specific=True):
        """Predict method for Bambi models

        Obtains in-sample and out-of-sample predictions from a fitted Bambi model.

        Parameters
        ----------
        idata: InferenceData
            The ``InferenceData`` instance returned by ``.fit()``.
        kind: str
            Indicates the type of prediction required. Can be ``"mean"`` or ``"pps"``. The
            first returns draws from the posterior distribution of the mean, while the latter
            returns the draws from the posterior predictive distribution
            (i.e. the posterior probability distribution for a new observation).
            Defaults to ``"mean"``.
        data: pandas.DataFrame or None
            An optional data frame with values for the predictors that are used to obtain
            out-of-sample predictions. If omitted, the original dataset is used.
        include_group_specific: bool
            If ``True`` make predictions including the group specific effects. Otherwise,
            predictions are made with common effects only (i.e. group specific are set
            to zero).
        inplace: bool
            If ``True`` it will modify ``idata`` in-place. Otherwise, it will return a copy of
            ``idata`` with the predictions added. If ``kind="mean"``, a new variable ending in
            ``"_mean"`` is added to the ``posterior`` group. If ``kind="pps"``, it appends a
            ``posterior_predictive`` group to ``idata``. If any of these already exist, it will be
            overwritten.

        Returns
        -------
        InferenceData or None
        """
        if kind not in ("mean", "pps"):
            raise ValueError("'kind' must be one of 'mean' or 'pps'")

        linear_predictor = 0
        x_offsets = []
        posterior = idata.posterior
        in_sample = data is None

        if not inplace:
            idata = deepcopy(idata)

        # Prepare dims objects
        response_dim = self.response.name + "_obs"
        response_levels_dim = self.response.name + "_dim"
        linear_predictor_dims = ("chain", "draw", response_dim)
        to_stack_dims = ("chain", "draw")
        design_matrix_dims = (response_dim, "__variables__")

        if isinstance(self.family, multivariate.MultivariateFamily):
            to_stack_dims = to_stack_dims + (response_levels_dim,)
            linear_predictor_dims = linear_predictor_dims + (response_levels_dim,)

        # Create design matrices
        if self._design.common:
            if in_sample:
                X = self._design.common.design_matrix
            else:
                X = self._design.common.evaluate_new_data(data).design_matrix

            # Add offset columns to their own design matrix and remove then from common matrix
            for term in self.offset_terms:
                term_slice = self._design.common.slices[term]
                x_offsets.append(X[:, term_slice])
                X = np.delete(X, term_slice, axis=1)

            # Create DataArray
            X_terms = list(self.common_terms)
            if self.intercept_term:
                X_terms.insert(0, "Intercept")
            b = posterior[X_terms].to_stacked_array("__variables__", to_stack_dims)

            # Add contribution due to the common terms
            X = xr.DataArray(X, dims=design_matrix_dims)
            linear_predictor += xr.dot(X, b)

        if self._design.group and include_group_specific:
            if in_sample:
                Z = self._design.group.design_matrix
            else:
                Z = self._design.group.evaluate_new_data(data).design_matrix

            # Create DataArray
            Z_terms = list(self.group_specific_terms)
            u = posterior[Z_terms].to_stacked_array("__variables__", to_stack_dims)

            # Add contribution due to the group specific terms
            Z = xr.DataArray(Z, dims=design_matrix_dims)
            linear_predictor += xr.dot(Z, u)

        # If model contains offset, add directly to the linear predictor
        if x_offsets:
            linear_predictor += np.column_stack(x_offsets).sum(axis=1)[:, np.newaxis, np.newaxis]

        # Sort dimensions
        linear_predictor = linear_predictor.transpose(*linear_predictor_dims)

        # Add coordinates for the observation number
        obs_n = len(linear_predictor[response_dim])
        linear_predictor = linear_predictor.assign_coords({response_dim: list(range(obs_n))})

        if kind == "mean":
            idata.posterior = self.family.predict(self, posterior, linear_predictor)
        else:
            pps_kwargs = {
                "model": self,
                "posterior": posterior,
                "linear_predictor": linear_predictor,
            }

            if not in_sample and isinstance(self.family, univariate.Binomial):
                pps_kwargs["trials"] = self._design.response.evaluate_new_data(data)

            pps = self.family.posterior_predictive(**pps_kwargs)
            pps = pps.to_dataset(name=self.response.name)

            if "posterior_predictive" in idata:
                del idata.posterior_predictive
            idata.add_groups({"posterior_predictive": pps})
            idata.posterior_predictive = idata.posterior_predictive.assign_attrs(
                modeling_interface="bambi", modeling_interface_version=__version__
            )

        if inplace:
            return None
        else:
            return idata

    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """Produce a graphviz Digraph from a built Bambi model.

        Requires graphviz, which may be installed most easily with
            ``conda install -c conda-forge python-graphviz``

        Alternatively, you may install the ``graphviz`` binaries yourself, and then
        ``pip install graphviz`` to get the python bindings.
        See http://graphviz.readthedocs.io/en/stable/manual.html for more information.

        Parameters
        ----------
        formatting: str
            One of ``"plain"`` or ``"plain_with_params"``. Defaults to ``"plain"``.
        name: str
            Name of the figure to save. Defaults to ``None``, no figure is saved.
        figsize: tuple
            Maximum width and height of figure in inches. Defaults to ``None``, the figure size is
            set automatically. If defined and the drawing is larger than the given size, the drawing
            is uniformly scaled down so that it fits within the given size.  Only works if ``name``
            is not ``None``.
        dpi: int
            Point per inch of the figure to save.
            Defaults to 300. Only works if ``name`` is not ``None``.
        fmt: str
            Format of the figure to save.
            Defaults to ``"png"``. Only works if ``name`` is not ``None``.

        Example
        --------
        >>> model = Model("y ~ x + (1|z)")
        >>> model.build()
        >>> model.graph()

        >>> model = Model("y ~ x + (1|z)")
        >>> model.fit()
        >>> model.graph()

        """
        self._check_built()

        graphviz = pm.model_to_graphviz(model=self.backend.model, formatting=formatting)

        width, height = (None, None) if figsize is None else figsize

        if name is not None:
            graphviz_ = graphviz.copy()
            graphviz_.graph_attr.update(size=f"{width},{height}!")
            graphviz_.graph_attr.update(dpi=str(dpi))
            graphviz_.render(filename=name, format=fmt, cleanup=True)

        return graphviz

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
        priors_common = [f"    {t.name} ~ {t.prior}" for t in self.common_terms.values()]
        if self.intercept_term:
            term = self.intercept_term
            priors_common = [f"    {term.name} ~ {term.prior}"] + priors_common

        priors_group = [f"    {t.name} ~ {t.prior}" for t in self.group_specific_terms.values()]

        # Prior for the correlation matrix in group-specific terms
        priors_cor = [f"    {k} ~ LKJCorr({v})" for k, v in self.priors_cor.items()]

        # Priors for auxiliary parameters, e.g., standard deviation in normal linear model
        priors_aux = [f"    {k} ~ {v}" for k, v in self.family.likelihood.priors.items()]

        # Offsets
        offsets = [f"    {t.name} ~ 1" for t in self.offset_terms.values()]

        priors_dict = {
            "Common-level effects": priors_common,
            "Group-level effects": priors_group,
            "Group-level correlation": priors_cor,
            "Offset effects": offsets,
            "Auxiliary parameters": priors_aux,
        }

        priors_list = []
        for group, values in priors_dict.items():
            if values:
                priors_list += ["\n".join([f"  {group}"] + values)]
        priors_message = "\n\n".join(priors_list)

        str_list = [
            f"Formula: {self.formula}",
            f"Family name: {self.family.name.capitalize()}",
            f"Link: {self.family.link.name}",
            f"Observations: {self.response.data.shape[0]}",
            "Priors:",
            priors_message,
        ]
        if self.backend and self.backend.fit:
            extra_foot = (
                "------\n"
                "* To see a plot of the priors call the .plot_priors() method.\n"
                "* To see a summary or plot of the posterior pass the object returned "
                "by .fit() to az.summary() or az.plot_trace()\n"
            )
            str_list += [extra_foot]

        return "\n".join(str_list)

    def __repr__(self):
        return self.__str__()

    @property
    def term_names(self):
        """Return names of all terms in order of addition to model."""
        return list(self.terms)

    @property
    def common_terms(self):
        """Return dict of all common effects in model."""
        return {
            k: v
            for (k, v) in self.terms.items()
            if not isinstance(v, GroupSpecificTerm) and v.kind not in ["intercept", "offset"]
        }

    @property
    def group_specific_terms(self):
        """Return dict of all group specific effects in model."""
        return {k: v for (k, v) in self.terms.items() if isinstance(v, GroupSpecificTerm)}

    @property
    def intercept_term(self):
        """Return the intercept term"""
        term = [
            v
            for v in self.terms.values()
            if not isinstance(v, GroupSpecificTerm) and v.kind == "intercept"
        ]
        if term:
            return term[0]
        else:
            return None

    @property
    def offset_terms(self):
        """Return dict of all offset effects in model."""
        return {
            k: v
            for (k, v) in self.terms.items()
            if not isinstance(v, GroupSpecificTerm) and v.kind == "offset"
        }


def prepare_prior(prior, kind, auto_scale):
    """Helper function to correctly set default priors, auto scaling, etc.

    Parameters
    ----------
    prior : Prior, float, or None.
    kind : string
        Accepted values are: ``"intercept"``, ``"common"``, or ``"group_specific"``.
    """
    if prior is None and not auto_scale:
        prior = get_default_prior(kind + "_flat")
    if isinstance(prior, Prior):
        prior.auto_scale = False
    else:
        scale = prior
        prior = get_default_prior(kind)
        prior.scale = scale
    return prior
