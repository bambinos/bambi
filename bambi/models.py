# pylint: disable=no-name-in-module
# pylint: disable=too-many-lines
# pylint: disable=too-many-positional-arguments
import logging
import warnings
from copy import copy, deepcopy
from importlib.metadata import version

import formulae as fm
import pandas as pd
import pymc as pm
from arviz_plots import plot_dist
from arviz_stats import residual_r2

from bambi.backend import PyMCModel
from bambi.config import config
from bambi.defaults import get_builtin_family
from bambi.parameters import ConditionalParameter, MarginalParameter
from bambi.families import Family
from bambi.families.builtin import Bernoulli, Cumulative, StoppingRatio
from bambi.families.types import DimType
from bambi.formula import Formula, check_ordinal_formula
from bambi.priors import Prior, scale_priors
from bambi.terms import ResponseTerm
from bambi.transformations import transformations_namespace
from bambi.utils import (
    clean_formula_lhs,
    indentify,
    listify,
    remove_common_intercept,
    wrapify,
)

_log = logging.getLogger("bambi")

ORDINAL_FAMILIES = (Cumulative, StoppingRatio)

__version__ = version("bambi")


class Model:
    """Specification of model class

    Parameters
    ----------
    formula : str or Formula
        A model description written using the formula syntax from the `formulae` library.
    data : pd.DataFrame
        A pandas dataframe containing the data on which the model will be fit, with column
        names matching variables defined in the formula.
    family : str or bambi.Family, optional
        A specification of the model family (analogous to the family object in R). Either
        a string, or an instance of class [](`bambi.Family`). If a string is passed, a
        family with the corresponding name must be defined in the defaults loaded at `Model`
        initialization. Valid pre-defined families are `"bernoulli"`, `"beta"`,
        `"binomial"`, `"categorical"`, `"gamma"`, `"gaussian"`, `"negativebinomial"`,
        `"poisson"`, `"t"`, and `"wald"`. Defaults to `"gaussian"`.
    priors : dict, optional
        Optional specification of priors for one or more terms. A dictionary where the keys are
        the names of terms in the model, "common," "group_specific," or the name of a model
        component, and the values are instances of class `Prior`. A distributional component
        name (e.g. "sigma" when it is modeled with a formula) maps to a nested dictionary of
        the same form; a constant component name maps directly to a `Prior`, a number, or an
        array. If priors are unset, use automatic priors inspired by the R rstanarm library.
        Names that don't match any term or component are reported with a warning; set
        `bmb.config["UNUSED_PRIORS"]` to `"error"` or `"ignore"` to change that.
        Bare term priors can be combined with priors nested under the parent component. If both
        specify the same term, the nested parent prior takes precedence.
    link : str or dict of str to str, optional
        The name of the link function to use. Valid names are `"cloglog"`, `"identity"`,
        `"inverse_squared"`, `"inverse"`, `"log"`, `"logit"`, `"probit"`, and
        `"softmax"`. Not all the link functions can be used with all the families.
        If a dictionary, keys are the names of the target parameters and the values are the names
        of the link functions.
    categorical : str or list of str, optional
        The names of any variables to treat as categorical. Can be either a single variable
        name, or a list of names. If categorical is `None`, the data type of the columns in
        the `data` will be used to infer handling. In cases where numeric columns are
        to be treated as categorical (e.g., group specific factors coded as numerical IDs),
        explicitly passing variable names via this argument is recommended.
    potentials : A list of 2-tuples, optional
        Optional specification of potentials. A potential is an arbitrary expression added to the
        likelihood, this is generally useful to add constrains to models, that are difficult to
        express otherwise. The first term of a 2-tuple is the name of a variable in the model, the
        second a lambda function expressing the desired constraint.
        If a constraint involves n variables, you can pass n 2-tuples or pass a tuple which first
        element is an n-tuple and second element is a lambda function with n arguments. The number
        and order of the lambda function has to match the number and order of the variable names.
    dropna : bool, optional
        When `True`, rows with any missing values in either the predictors or outcome are
        automatically dropped from the dataset in a listwise manner, optional.
    auto_scale : bool
        If `True` (default), priors are automatically rescaled to the data
        (to be weakly informative) any time default priors are used. Note that any priors
        explicitly set by the user will always take precedence over default priors.
    noncentered : bool or dict[str, bool], optional
        Default parameterization for group-specific terms.
        `True` (default) uses non-centered; `False` uses centered. Can also be a `dict`
        keyed by component name (e.g. `{"mu": True, "sigma": False}`) for per-parameter
        defaults; missing keys default to `True`, unknown keys raise. Per-`Prior`
        `noncentered=` overrides this setting.
    center_predictors : bool, optional
        If `True` (default), and if there is an intercept in the common terms, the data is
        centered by subtracting the mean. The centering is undone after sampling to provide
        the actual intercept in all conditional parameters that have an intercept. Note
        that this changes the interpretation of the prior on the intercept because it refers
        to the intercept of the centered data.
    extra_namespace : dict, optional
        Additional user supplied variables with transformations or data to include in the
        environment where the formula is evaluated. Defaults to `None`.
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
        center_predictors=True,
        extra_namespace=None,
    ):
        # attributes that are set later
        self.parameters = {}
        self.built = False  # build()

        # build() will loop over this, calling _set_priors()
        self._added_priors = {}

        self.family = None  # _add_response()
        self.backend = None  # _set_backend()

        self.auto_scale = auto_scale
        self.dropna = dropna
        self.formula = formula
        self.noncentered = noncentered
        self.potentials = potentials
        self.center_predictors = center_predictors

        # Read and clean data
        if not isinstance(data, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame.")

        # Some columns are converted to categorical
        self.data = with_categorical_cols(data, categorical)

        # Handle priors
        priors = {} if priors is None else deepcopy(priors)

        # Obtain design matrices and related objects.
        na_action = "drop" if dropna else "error"

        # Handle additional namespaces
        additional_namespace = transformations_namespace.copy()
        if not isinstance(extra_namespace, (type(None), dict)):
            raise ValueError("'namespace' must be a dictionary or None")

        if isinstance(extra_namespace, dict):
            additional_namespace.update(extra_namespace)

        # Create family
        self._set_family(family, link)

        ## Main parameter
        if isinstance(self.family, ORDINAL_FAMILIES):
            self.formula = check_ordinal_formula(self.formula)
            # Notice the intercept is added so formulae constrains categorical predictors, avoiding
            # linear dependencies with the cutpoints.
            # Then the intercept is removed from the design matrix because of the cutpoints.
            design = fm.design_matrices(
                self.formula.main + " + 1",
                self.data,
                na_action,
                1,
                additional_namespace,
            )
            design = remove_common_intercept(design)
        else:
            design = fm.design_matrices(
                self.formula.main, self.data, na_action, 1, additional_namespace
            )

        if design.response is None:
            raise ValueError(
                "No outcome variable is set! "
                "Please specify an outcome variable using the formula interface."
            )

        # Merge bare term priors with nested parent priors; nested entries take precedence.
        parent_name = self.family.likelihood.parent
        parent_priors = {name: prior for name, prior in priors.items() if name != parent_name}
        if parent_name in priors:
            parent_priors.update(priors[parent_name])

        # Add response
        self.response_term = ResponseTerm(design.response)

        if self.response_term.is_cr:
            # Competing risks use a model-local family with cause-specific parameters.
            self.family = copy(self.family)
            self.family.PARAMETERS = {
                name: self.family.get_param_spec(name)._replace(ndim=1, coefs_dim=DimType.RESPONSE)
                for name in self.family.likelihood.params
            }

        # Add parent parameter
        self.parameters[self.family.likelihood.parent] = ConditionalParameter(
            self.family.likelihood.parent, design, parent_priors, self, is_parent=True
        )

        # Get auxiliary parameters, so we add either conditional or marginal parameters
        auxiliary_parameters = list(self.family.auxiliary_parameters)

        ## Other parameters
        ### Conditional
        for name, extra_formula in zip(self.formula.additionals_lhs, self.formula.additionals):
            # Check 'name' is part of parameter values
            if name not in auxiliary_parameters:
                raise ValueError(
                    f"'{name}' is not a parameter of the family."
                    f"Available parameters: {auxiliary_parameters}."
                )

            # Create design matrix, only for the response part
            design = fm.design_matrices(
                clean_formula_lhs(extra_formula),
                self.data,
                na_action,
                1,
                additional_namespace,
            )

            # If priors were not passed, pass an empty dictionary
            parameter_priors = priors.get(name, {})

            # Create conditional parameter
            self.parameters[name] = ConditionalParameter(
                name, design, parameter_priors, self, is_parent=False
            )

            # Remove parameter name from the list
            auxiliary_parameters.remove(name)

        ### Marginal
        for name in auxiliary_parameters:
            parameter_prior = priors.get(name, None)
            self.parameters[name] = MarginalParameter(name, parameter_prior, self)

        # Validate prior names, now that every component and its terms are known.
        self._check_prior_names(priors)

        # Validate per-parameter noncentered dict, now that all parameters are known.
        if isinstance(self.noncentered, dict):
            unknown = set(self.noncentered) - set(self.parameters)
            if unknown:
                raise ValueError(
                    f"Unknown parameter name(s) in `noncentered`: {sorted(unknown)}. "
                    f"Valid parameter names for this model: {sorted(self.parameters)}."
                )

        # Build priors
        self._build_priors()

    def fit(
        self,
        draws=1000,
        tune=1000,
        discard_tuned_samples=True,
        omit_offsets=True,
        include_mean=None,
        include_response_params=False,
        inference_method=None,
        init="auto",
        n_init=50000,
        chains=None,
        cores=None,
        random_seed=None,
        nuts=None,
        **kwargs,
    ):
        """Fit the model using PyMC

        Parameters
        ----------
        draws : int, optional
            The number of samples to draw from the posterior distribution. Defaults to 1000.
        tune : int, optional
            Number of iterations to tune. Defaults to 1000. Samplers adjust the step sizes,
            scalings or similar during tuning. These tuning samples are be drawn in addition to the
            number specified in the `draws` argument, and will be discarded unless
            `discard_tuned_samples` is set to `False`.
        discard_tuned_samples : bool, optional
            Whether to discard posterior samples of the tune interval. Defaults to `True`.
        omit_offsets : bool, optional
            Omits offset terms in the `DataTree` object returned when the model includes
            group specific effects. Defaults to `True`.
        include_mean : bool, optional, deprecated
            **This argument is deprecated and will be removed in future versions**.
            Use `include_response_params`.
        include_response_params : bool, optional
            Include parameters of the response distribution in the output. These usually take more
            space than other parameters as there's one of them per observation. Defaults to `False`.
        inference_method : str or None, optional
            The method to use for fitting the model. If `None` (default), Bambi lets the backend
            select the MCMC sampler: PyMC uses nutpie when it is installed and compatible with the
            model and sampling options, otherwise it uses its built-in sampler. Pass `"pymc"` to
            always use the built-in sampler. NUTS implementations include `"pymc"`, `"nutpie"`,
            `"blackjax"`, and `"numpyro"`. Alternatively, `"vi"` fits the model using variational
            inference as implemented in PyMC's `fit` function. Finally, `"laplace"` uses a Laplace
            approximation and is not recommended other than for pedagogical use.
        init : str, optional
            Initialization method. Defaults to `"auto"`. The available methods are:

            - `"auto"`: Use `"jitter+adapt_diag"` and if this method fails it uses `"adapt_diag"`.
            - `"adapt_diag"`: Start with an identity mass matrix and then adapt a diagonal based on
            the variance of the tuning samples. All chains use the test value
            (usually the prior mean) as starting point.
            - `"jitter+adapt_diag"`: Same as `"adapt_diag"`, but use test value plus a uniform
            jitter in [-1, 1] as starting point in each chain.
            - `"advi+adapt_diag"`: Run ADVI and then adapt the resulting diagonal mass matrix based
            on the sample variance of the tuning samples.
            - `"advi+adapt_diag_grad"`: Run ADVI and then adapt the resulting diagonal mass matrix
            based on the variance of the gradients during tuning. This is **experimental** and might
            be removed in a future release.
            - `"advi"`: Run ADVI to estimate posterior mean and diagonal mass matrix.
            - `"advi_map"`: Initialize ADVI with MAP and use MAP as starting point.
            - `"map"`: Use the MAP as starting point. This is strongly discouraged.
            - `"adapt_full"`: Adapt a dense mass matrix using the sample covariances.
            All chains use the test value (usually the prior mean) as starting point.
            - `"jitter+adapt_full"`: Same as `"adapt_full"`, but use test value plus a uniform
            jitter in [-1, 1] as starting point in each chain.

        n_init : int, optional
            Number of initialization iterations. Only works for `"advi"` init methods.
        chains : int, optional
            The number of chains to sample. Running independent chains is important for some
            convergence statistics and can also reveal multiple modes in the posterior. If `None`,
            then set to either `cores` or 2, whichever is larger.
        cores : int, optional
            The number of chains to run in parallel. If `None`, it is equal to the number of CPUs
            in the system unless there are more than 4 CPUs, in which case it is set to 4.
        random_seed : int or list of ints, optional
            A list is accepted if cores is greater than one.
        nuts : dict, optional
            A dictionary of NUTS sampler settings passed directly to `pm.sample(nuts=...)`, e.g.
            `model.fit(nuts={"target_accept": 0.9, "max_treedepth": 12})`.
        kwargs : dict
            For other kwargs see the documentation for ``pm.sample()``.

        Returns
        -------
        `DataTree` or `Approximation`
            It returns a `DataTree` if `inference_method` is `"pymc"`, `"nutpie"`,
            `"blackjax"`, `"numpyro"`, or `"laplace"`, and an `Approximation` object if  `"vi"`.
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

        if "nuts_sampler_kwargs" in kwargs:
            warnings.warn(
                "'nuts_sampler_kwargs' is deprecated. Pass NUTS settings via the 'nuts' parameter "
                "instead, e.g. model.fit(nuts={'target_accept': 0.9}).",
                FutureWarning,
                stacklevel=2,
            )
            legacy = kwargs.pop("nuts_sampler_kwargs")
            if nuts is None:
                nuts = legacy
            else:
                nuts = {**legacy, **nuts}

        if not self.built:
            self.build()

        # Tell user which event is being modeled
        if isinstance(self.family, Bernoulli):
            _log.info(
                "Modeling the probability that %s==%s",
                self.response_term.name,
                str(self.response_term.reference or 1),
            )

        if include_mean is not None:
            warnings.warn(
                "'include_mean' has been replaced by 'include_response_params' and "
                "is not going to work in the future",
                FutureWarning,
            )
            include_response_params = include_mean

        return self.backend.run(
            draws=draws,
            tune=tune,
            discard_tuned_samples=discard_tuned_samples,
            omit_offsets=omit_offsets,
            include_response_params=include_response_params,
            inference_method=inference_method,
            init=init,
            n_init=n_init,
            chains=chains,
            cores=cores,
            random_seed=random_seed,
            nuts=nuts,
            **kwargs,
        )

    def build(self):
        """Set up the model for sampling/fitting

        Creates an instance of the underlying PyMC model and adds all the necessary terms to it.
        """
        self.backend = PyMCModel(self)
        self.backend.build()
        self.built = True

    def set_priors(self, priors=None, common=None, group_specific=None):
        """Set priors for one or more existing terms.

        Parameters
        ----------
        priors : dict or None, optional
            Dictionary of priors to update. Accepts the same specification as the `priors`
            argument of `Model`, and entries here take precedence over the `common` and
            `group_specific` arguments. Names that don't match any term or component are
            reported with a warning; set `bmb.config["UNUSED_PRIORS"]` to `"error"` or
            `"ignore"` to change that.
        common : Prior, int, float or None, optional
            A prior specification to apply to all common terms included in the model.
        group_specific : Prior, int, float or None, optional
            A prior specification to apply to all group specific terms included in the model.
        """
        if priors is not None:
            # Validate before touching any state, so a rejected call leaves the model as it was.
            self._check_prior_names(priors)

        kwargs = dict(zip(["priors", "common", "group_specific"], [priors, common, group_specific]))
        self._added_priors.update(kwargs)
        self._build_priors()  # After updating, we need to rebuild priors.
        self.built = False

    def _build_priors(self):
        """Carry out all operations related to the construction and/or scaling of priors."""
        # Set custom priors that have been passed via `Model.set_priors()`
        self._set_priors(**self._added_priors)

        # Prepare all priors
        for parameter in self.conditional_parameters.values():
            parameter.build_priors()

        for name, parameter in self.marginal_parameters.items():
            if isinstance(parameter.prior, Prior):
                parameter.prior.auto_scale = False
            elif isinstance(parameter.prior, (int, float)):
                continue
            elif parameter.prior is not None:
                raise ValueError(f"'{parameter.prior}' is not a valid prior.")
            else:
                default_prior = self.family.default_priors.get(name, None)
                if default_prior is None:
                    raise ValueError(f"The parameter '{name}' needs a prior.")
                parameter.prior = default_prior

        # Scale priors if there is at least one term in the model and auto_scale is True
        if self.auto_scale:
            scale_priors(self)

    def _check_prior_names(self, priors):
        """Report names in `priors` that don't match any term or parameter."""
        behavior = config["UNUSED_PRIORS"]
        if behavior == "ignore":
            return

        parent_name = self.family.likelihood.parent
        valid = (
            set(self.parameters)
            | set(self.parameters[parent_name].terms)
            | {"common", "group_specific"}
        )

        unused = []
        for name, value in priors.items():
            if name not in valid:
                unused.append(name)
            elif isinstance(value, dict) and name in self.conditional_parameters:
                nested_valid = set(self.parameters[name].terms) | {"common", "group_specific"}
                unused.extend(f"{name}.{n}" for n in sorted(set(value) - nested_valid))

        if not unused:
            return

        message = f"Unused name(s) in `priors`: {sorted(unused)}."
        hint = ' Set `bmb.config["UNUSED_PRIORS"]` to change this.'
        if behavior == "warn":
            warnings.warn(message + " They will be ignored." + hint, UserWarning)
        else:
            raise ValueError(f"{message} Valid names for this model: {sorted(valid)}.{hint}")

    def _set_priors(self, priors=None, common=None, group_specific=None):
        """Internal version of `set_priors()`, with same arguments.

        Runs during `Model._build_priors()`.
        """
        # Arguments `common` and `group_specific` only affect the parent parameter.
        parent_name = self.family.likelihood.parent

        # 'common' and 'group_specific' only apply to the parent parameter
        parent_parameter = self.parameters[self.family.likelihood.parent]
        if common is not None:
            for term in parent_parameter.common_terms.values():
                term.prior = common

        if group_specific is not None:
            for term in parent_parameter.group_specific_terms.values():
                term.prior = group_specific

        if priors is not None:
            # `normalized_priors` maps component names to their prior specifications:
            #   - a term-to-prior dict for conditional parameters,
            #   - a single prior for marginal parameters.
            # Bare term priors are merged into the parent component, with explicitly nested
            # parent priors taking precedence.
            normalized_priors = {name: priors[name] for name in self.parameters if name in priors}
            parent_priors = {
                name: prior for name, prior in priors.items() if name not in self.parameters
            }
            if parent_name in normalized_priors:
                parent_priors.update(normalized_priors[parent_name])
            normalized_priors[parent_name] = parent_priors

            # Make sure mutation of Prior objects within update_priors does not have side effects.
            normalized_priors = deepcopy(normalized_priors)

            for name, component in self.parameters.items():
                prior = normalized_priors.get(name)
                if prior is not None:
                    component.update_priors(prior)

    def _set_family(self, family, link):
        """Set the Family of the model

        Parameters
        ----------
        family : str or bambi.families.Family
            A specification of the model family.
            Either a string, or an instance of class `families.Family`.
            If a string is passed, a family with the corresponding name must be defined in the
            defaults loaded at model initialization.
        link : str or dict of str to str
            The name of the link function to use. Valid names are `"cloglog"`, `"identity"`,
            `"inverse_squared"`, `"inverse"`, `"log"`, `"logit"`, `"probit"`, and
            `"softmax"`. Not all the link functions can be used with all the families.
            If a dictionary, keys are the names of the target parameters and the values are the
            names of the link functions.

        Returns
        -------
        `None`
        """

        # If string, get builtin family
        if isinstance(family, str):
            family = get_builtin_family(family)

        # Always ensure family is indeed instance of Family
        if not isinstance(family, Family):
            raise ValueError("'family' must be a string or a Family object.")

        # Override family's link if another is explicitly passed
        # If `link` is string, we assume it wants to override only the `parent` parameter
        if link is not None:
            if isinstance(link, str):
                links = family.link.copy()
                links[family.likelihood.parent] = link
            elif isinstance(link, dict):
                links = link
            else:
                raise ValueError("'link' must be of type 'str' or 'dict'.")
            family.link = links

        self.family = family

    def set_alias(self, aliases):
        """Set aliases for the terms and auxiliary parameters in the model

        Parameters
        ----------
        aliases : dict of str to str
            A dictionary where key represents the original term name and the value is the alias.

        Returns
        -------
        `None`
        """
        if not isinstance(aliases, dict):
            raise ValueError(f"'aliases' must be a dictionary, not a {type(aliases)}.")

        # Keep track of any passed aliases that are not used
        missing_names = []

        # If there is a single conditional parameter (the response)
        #   * Keys are the names of the terms and the values are their aliases.
        # If there are multiple conditional parameters
        #   * Keys are the names of the response parameters
        #     * If it's a marginal parameter, the value must be a string
        #     * If it's a conditional parameter, the value must be a dictionary
        #        * Here, names are term names, and values are their aliases
        #     * There's unavoidable redundancy in the response name
        #       "sigma": {"sigma": "alias"}}
        if len(self.conditional_parameters) == 1:  # pylint: disable=too-many-nested-blocks
            parent_parameter = self.parameters[self.family.likelihood.parent]
            for name, alias in aliases.items():
                assert isinstance(alias, str)

                # Monitor if this particular alias is used
                is_used = False

                # If it's the name of the parent parameter
                if name == self.family.likelihood.parent:
                    parent_parameter.alias = alias
                    is_used = True

                if name in self.marginal_parameters:
                    assert isinstance(alias, str)
                    self.marginal_parameters[name].alias = alias
                    is_used = True

                # If it's a term name
                if name in parent_parameter.terms:
                    parent_parameter.terms[name].alias = alias
                    is_used = True

                # Now add aliases for hyperpriors in group specific terms
                for term in parent_parameter.group_specific_terms.values():
                    if name in term.prior.args:
                        term.hyperprior_alias = {name: alias}
                        is_used = True

                # If it's the name of the response
                if name in (self.response_term.name, self.response_term.full_name):
                    self.response_term.alias = alias
                    is_used = True

                # Add any aliases not used in prior logic to unused alias list
                if is_used is False:
                    missing_names.append(name)
        else:
            for parameter_name, parameter_aliases in aliases.items():
                if parameter_name in self.marginal_parameters:
                    assert isinstance(parameter_aliases, str)
                    self.marginal_parameters[parameter_name].alias = parameter_aliases
                elif parameter_name in (self.response_term.name, self.response_term.full_name):
                    assert isinstance(parameter_aliases, str)
                    self.response_term.alias = parameter_aliases
                else:
                    assert isinstance(parameter_aliases, dict)
                    assert parameter_name in self.conditional_parameters
                    parameter = self.conditional_parameters[parameter_name]
                    for name, alias in parameter_aliases.items():
                        is_used = False

                        if name in parameter.terms:
                            parameter.terms[name].alias = alias
                            is_used = True

                        # Useful for non-response conditional parameters
                        if name == parameter.name:
                            parameter.alias = alias
                            is_used = True

                        for term in parameter.group_specific_terms.values():
                            if name in term.prior.args:
                                term.hyperprior_alias = {name: alias}
                                is_used = True

                        # Add any aliases not used in prior logic to unused alias list
                        if is_used is False:
                            missing_names.append(name)

        # Report unused aliases
        if missing_names:
            # If only a few, tell user explicitly which aren't used
            if len(missing_names) <= 5:
                warnings.warn(
                    "The following names do not match any terms, their aliases were "
                    f"not assigned: {', '.join(missing_names)}",
                    UserWarning,
                )
            # If many, throw a generic warning
            else:
                warnings.warn(
                    f"There are {len(missing_names)} names that do not match any terms, "
                    "so their aliases were not assigned.",
                    UserWarning,
                )
        # Model needs to be rebuilt after modifying aliases
        self.built = False

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
        filter_vars=None,
        kind="kde",
        ci_kind=None,
        ci_prob=None,
        point_estimate=None,
        plot_collection=None,
        backend=None,
        labeller=None,
        aes_by_visuals=None,
        visuals=None,
        stats=None,
        figsize=None,
        omit_offsets=True,
        omit_group_specific=True,
        random_seed=None,
        bins=None,
        hdi_prob=None,
        round_to=None,
        **pc_kwargs,
    ):
        """Samples from the prior distribution and plots its marginals.

        Parameters
        ----------
        draws : int, optional
            Number of draws to sample from the prior predictive distribution. Defaults to 5000.
        var_names : str or list of str, optional
            A list of names of variables for which to compute the prior predictive
            distribution. Defaults to `None` which means to include both observed and
            unobserved RVs.
        filter_vars : {"like", "regex"} or None, optional
            If `None`, interpret `var_names` as the real variable names.
            If `"like"`, interpret `var_names` as substrings of the real variable names.
            If `"regex"`, interpret `var_names` as regular expressions on the real variable names.
            Forwarded to [](`arviz_plots.plot_dist`).
        kind : str, optional
            Type of plot to display (`"kde"` or `"hist"`). For discrete variables this argument
            is ignored and a histogram is always used. Forwarded to [](`arviz_plots.plot_dist`).
        ci_kind : {"eti", "hdi"}, optional,
            Which credible interval to use. Defaults to `arviz_base.rcParams["stats.ci_kind"]`.
            Forwarded to [](`arviz_plots.plot_dist`).
        ci_prob : float, optional
            Indicates the probability that should be contained within the plotted credible interval.
            Defaults to `arviz_base.rcParams["stats.ci_prob"]`.
            Forwarded to [](`arviz_plots.plot_dist`).
        point_estimate : str, optional
            Plot point estimate per variable. Values should be `"mean"`, `"median"`, `"mode"`
            or `None`. When `None` (default) use `arviz_base.rcParams["stats.point_estimate"]`.
            Forwarded to [](`arviz_plots.plot_dist`).
        plot_collection : arviz_plots.PlotCollection, optional
            The plot collection to use. Forwarded to [](`arviz_plots.plot_dist`).
        backend : {"matplotlib", "plotly", "bokeh"}, optional
            The backend to use for plotting.
            If `None`, it inspects whether `plot_connection` is not `None`.
            If it's not, it uses `plot_collection.backend`.
            Otherweise, it uses `arviz_base.rcParams["plot.backend"]`.
            Forwarded to [](`arviz_plots.plot_dist`).
        labeller : arviz_base.labels.BaseLabeller, optional
            The labeller. If `None`, it uses [](`arviz_base.labels.BaseLabeller`).
            Forwarded to [](`arviz_plots.plot_dist`).
        aes_by_visuals : mapping of {str : sequence of str}, optional
            Forwarded to [](`arviz_plots.plot_dist`). See `aes_by_visuals` in there.
        visuals : mapping of {str : mapping or bool}, optional
            Forwarded to [](`arviz_plots.plot_dist`). See `visuals` in there.
        stats : mapping, optional
            Forwarded to [](`arviz_plots.plot_dist`). See `stats` in there.
        figsize : tuple, optional
            Figure size. If `None` it will be defined automatically.
        omit_offsets : bool
            Whether to omit offset terms in the plot. Defaults to `True`.
        omit_group_specific : bool, optional
            Whether to omit group specific effects in the plot. Defaults to `True`.
        random_seed : int or None, optional
            Seed for random number generator.
            Passed down to [Model.prior_predictive](`bambi.Model.prior_predictive`).
        bins : int, optional, deprecated
            **This argument is deprecated and will be removed in future versions**.
        hdi_prob : float or str, optional, deprecated
            Plots highest density interval for chosen percentage of density.
            Use `"hide"` to hide the highest density interval.
            **This argument is deprecated and will be removed in future versions**.
        round_to : int, optional, deprecated
            Controls formatting of floats. Defaults to 2 or the integer part, whichever is bigger.
            **This argument is deprecated and will be removed in future versions**.
        pc_kwargs : dict
            Passed to [](`arviz_plots.PlotCollection.wrap`)

        Returns
        -------
        pc : arviz_plots.PlotCollection

        """
        self._check_built()

        if stats is None:
            stats = {}
        else:
            stats = stats.copy()
            stats["dist"] = stats.get("dist", {}).copy()

        unobserved_rvs_names = []
        flat_rvs = []

        if hdi_prob is not None:
            warnings.warn(
                "'hdi_prob' has been renamed to 'ci_prob' and will be removed in future versions",
                FutureWarning,
            )
            ci_prob = hdi_prob

        if bins is not None:
            warnings.warn(
                """'bins' argument is deprecated and will be removed in future versions
                please use `stats={"dist": {"bins": bins}}`
                """,
                FutureWarning,
            )
            stats.get("dist", {}).setdefault("bins", bins)

        if round_to is not None:
            warnings.warn(
                """'round_to' argument is deprecated and will be removed in future versions
                please use `stats={"dist": {"round_to": round_to}}`""",
                FutureWarning,
            )
            stats.get("dist", {}).setdefault("round_to", round_to)

        if pc_kwargs is None:
            pc_kwargs = {}

        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        if figsize is not None:
            pc_kwargs["figure_kwargs"]["figsize"] = figsize

        for unobserved in self.backend.model.unobserved_RVs:
            if "Flat" in str(unobserved):
                flat_rvs.append(unobserved.name)
            else:
                # Don't include deterministics that go into the likelihood (e.g. 'mu' normal model)
                is_likelihood_param = unobserved.name in self.family.likelihood.params
                is_deterministic = unobserved in self.backend.model.deterministics
                if is_likelihood_param and is_deterministic:
                    continue
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
                "Variables %s have flat priors, and hence they are not plotted",
                ", ".join(flat_rvs),
            )

        if omit_offsets:
            var_names = [name for name in var_names if not name.endswith("_offset")]

        if omit_group_specific:
            group_specific_var_names = [
                name
                for parameter in self.conditional_parameters.values()
                for name in parameter.group_specific_terms
            ]
            var_names = [name for name in var_names if name not in group_specific_var_names]

        pc = None
        if var_names:
            # Sort variable names so Intercept is in the beginning
            if "Intercept" in var_names:
                var_names.insert(0, var_names.pop(var_names.index("Intercept")))
            pps = self.prior_predictive(draws=draws, var_names=var_names, random_seed=random_seed)

            pc = plot_dist(
                pps,
                group="prior",
                var_names=var_names,
                filter_vars=filter_vars,
                kind=kind,
                point_estimate=point_estimate,
                ci_kind=ci_kind,
                ci_prob=ci_prob,
                plot_collection=plot_collection,
                backend=backend,
                labeller=labeller,
                aes_by_visuals=aes_by_visuals,
                visuals=visuals,
                stats=stats,
                **pc_kwargs,
            )
        return pc

    def prior_predictive(self, draws=500, var_names=None, omit_offsets=True, random_seed=None):
        """Generate samples from the prior predictive distribution.

        Parameters
        ----------
        draws : int, optional
            Number of draws to sample from the prior predictive distribution. Defaults to 500.
        var_names : str, list of str or None, optional
            A list of names of variables for which to compute the prior predictive distribution.
            Defaults to `None` which means both observed and unobserved RVs.
        omit_offsets : bool, optional
            Whether to omit offset terms in the plot. Defaults to `True`.
        random_seed : int or None, optional
            Seed for the random number generator.

        Returns
        -------
        DataTree
            `DataTree` object with the groups `prior`, `prior_predictive` and
            `observed_data`, and, when applicable, `constant_data`.
        """
        self._check_built()

        if var_names is None:
            variables = self.backend.model.unobserved_RVs + self.backend.model.observed_RVs
            variables_names = [v.name for v in variables]
            var_names = pm.util.get_default_varnames(variables_names, include_transformed=False)

        if omit_offsets:
            var_names = [name for name in var_names if not name.endswith("_offset")]

        idata = pm.sample_prior_predictive(
            draws=draws,
            var_names=var_names,
            model=self.backend.model,
            random_seed=random_seed,
        )

        for group in idata.children:
            getattr(idata, group).attrs["modeling_interface"] = "bambi"
            getattr(idata, group).attrs["modeling_interface_version"] = __version__

        return idata

    def predict(
        self,
        idata,
        kind="response_params",
        data=None,
        inplace=True,
        include_group_specific=True,
        random_seed=None,
        progressbar=False,
    ):
        """Predict method for Bambi models

        Obtains in-sample and out-of-sample predictions from a fitted Bambi model.

        Parameters
        ----------
        idata : DataTree
            The `DataTree` instance returned by `.fit()`.
        kind : str, optional
            Indicates the type of prediction required. Can be `"response_params"`,
            `"response"`, `"response_conditional"`, or `"time_and_cause"`.

            * `"response_params"` returns draws from the posterior distribution of the
              likelihood parameters.
            * `"response"` returns draws from the posterior predictive response distribution.
              It is available for every response type and ignores censoring or truncation of the
              observed response. For competing-risks responses, it draws the first-event time.
            * `"response_conditional"` is available for censored, truncated, and competing-risks
              responses only. For censored responses, it conditions the underlying response on the
              observed censoring restriction. For truncated responses, it draws from the truncated
              distribution. For competing risks, it conditions right-censored observations on the
              first event occurring after the observed time. Exact events are drawn as `"response"`.
            * `"time_and_cause"` is available for competing-risks responses only. It returns
              first-event times and one-based cause codes in separate variables.

            Defaults to `"response_params"`.
        data : pd.DataFrame or None, optional
            An optional data frame with values for the predictors that are used to obtain
            out-of-sample predictions. If omitted, the original dataset is used.
        inplace : bool, optional
            If `True` it will modify `idata` in-place. Otherwise, it will return a copy of
            `idata` with the predictions added. If `kind="response_params"`, a new variable
            with the name of the parent parameter, e.g. `"mu"` and `"sigma"` for a Gaussian
            likelihood, or `"p"` for a Bernoulli likelihood, is added to the `posterior` group
            for in-sample predictions or to `predictions` for out-of-sample data. With
            `kind="response"`, the draws are added to `posterior_predictive` in sample or to
            `predictions` out of sample. The same applies to
            `kind="response_conditional"`. With `kind="time_and_cause"`, the
            first-event times and cause codes are added as separate variables named
            `<response>_time` and `<response>_cause`. Existing output groups are
            overwritten. For a transformed univariate response, `<response>` is the first
            argument of the transformation; for example, predictions from `censored(y, status)`
            are named `y`. A `counts(y1, y2)` response is named `y1_y2`.
        include_group_specific : bool, optional
            Determines if predictions incorporate group-specific effects. If `False`, predictions
            are made with common effects only (i.e. group specific are set to zero). Defaults to
            `True`.
        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.
        progressbar : bool, optional
            Whether to display a progress bar. Defaults to `False`.

        Returns
        -------
        DataTree or None
        """
        if kind not in (
            "mean",
            "pps",
            "response_params",
            "response",
            "response_conditional",
            "time_and_cause",
        ):
            raise ValueError(
                "'kind' must be one of 'response_params', 'response', 'response_conditional', "
                "or 'time_and_cause'"
            )

        if kind == "time_and_cause" and not self.response_term.is_cr:
            raise ValueError("'kind=time_and_cause' is only available for competing-risks models.")

        if kind == "response_conditional" and not (
            self.response_term.is_censored
            or self.response_term.is_truncated
            or self.response_term.is_cr
        ):
            raise ValueError(
                "'kind=response_conditional' is only available for censored, truncated, "
                "or competing-risks models."
            )

        if kind == "mean":
            kind = "response_params"
            warnings.warn(
                "'mean' has been replaced by 'response_params' and "
                "is not going to work in the future",
                FutureWarning,
            )
        if kind == "pps":
            kind = "response"
            warnings.warn(
                "'pps' has been replaced by 'response' and is not going to work in the future",
                FutureWarning,
            )

        return self.backend.predict(
            idata=idata,
            data=data,
            include_group_specific=include_group_specific,
            random_seed=random_seed,
            kind=kind,
            inplace=inplace,
            progressbar=progressbar,
        )

    def compute_log_likelihood(self, idata, data=None, inplace=True, progressbar=False):
        """Compute the model's log-likelihood

        Parameters
        ----------
        idata : DataTree
            The `DataTree` instance returned by `.fit()`.
        data : pd.DataFrame or None, optional
            An optional data frame with values for the predictors and the response on which
            the model's log-likelihood function is evaluated.
            If omitted, the original dataset is used.
        inplace : bool, optional
            If `True` it will modify `idata` in-place. Otherwise, it will return a copy of
            `idata` with the `log_likelihood` group added.
        progressbar : bool, optional
            Whether to display the log-likelihood computation progress bar. Defaults to `False`.

        Returns
        -------
        DataTree or None
        """
        self._check_built()
        return self.backend.compute_log_likelihood(
            idata=idata, data=data, inplace=inplace, progressbar=progressbar
        )

    def r2_score(self, idata, summary=True):
        """R² for Bayesian regression models.

        The R², or coefficient of determination, is defined as the proportion of variance
        in the data that is explained by the model. It is computed as the variance of the
        predicted values divided by the variance of the predicted values plus the variance
        of the residuals. For details of the Bayesian R² see [1]_.

        Parameters
        ----------
        idata : DataTree
            The `DataTree` instance returned by `.fit()`. It should contain the
            `posterior_predictive` group, otherwise it will be computed and added to `idata`.
        summary : bool, optional
            If `True`, it returns a summary of the Bayesian R². Otherwise, it returns the
            posterior samples of the Bayesian R².

        Returns
        -------
        pandas.Series
            A series with the following indices:
            r2: mean value for the Bayesian R²
            r2_std: standard deviation of the Bayesian R².

        References
        ----------
        .. [1] Gelman et al. *R-squared for Bayesian regression models*.
            The American Statistician. 73(3) (2019). <https://doi.org/10.1080/00031305.2018.1549100>
        """
        response_name = self.response_term.name
        pred_mean = self.family.likelihood.parent

        if pred_mean not in idata.posterior:
            self.predict(idata, kind="response_params", inplace=True)

        # We should change this to use bayesian_r2 ensuring we pass the correct scale for each
        # family we could use residual_r2 as a fallback for families we don't have implemented
        # yet we may want to have an argument to compute the loo_r2 as well or a separate method
        return residual_r2(idata, pred_mean=pred_mean, obs_name=response_name, summary=summary)

    def compute_log_prior(self, idata, inplace=True):
        """Compute the model's log-prior

        Parameters
        ----------
        idata : DataTree
            The `DataTree` instance returned by `.fit()`.
        inplace : bool, optional
            If `True` it will modify `idata` in-place. Otherwise, it will return a copy of
            `idata` with the `log_prior` group added.

        Returns
        -------
        DataTree or None
        """
        if not self.built:
            self.build()

        return self.backend.compute_log_prior(idata=idata, inplace=inplace)

    def graph(self, formatting="plain", name=None, figsize=None, dpi=300, fmt="png"):
        """Produce a graphviz Digraph from a built Bambi model.

        Requires graphviz, which may be installed most easily with:
        ```cmd
        conda install -c conda-forge python-graphviz
        ```

        Alternatively, you may install the `graphviz` binaries yourself, and then
        `pip install graphviz` to get the python bindings.
        See <http://graphviz.readthedocs.io/en/stable/manual.html> for more information.

        Parameters
        ----------
        formatting : str, optional
            One of `"plain"` or `"plain_with_params"`. Defaults to `"plain"`.
        name : str, optional
            Name of the figure to save. Defaults to `None`, no figure is saved.
        figsize : tuple, optional
            Maximum width and height of figure in inches. Defaults to `None`, the figure size is
            set automatically. If defined and the drawing is larger than the given size, the drawing
            is uniformly scaled down so that it fits within the given size.  Only works if `name`
            is not `None`.
        dpi : int, optional
            Point per inch of the figure to save.
            Defaults to 300. Only works if `name` is not `None`.
        fmt : str, optional
            Format of the figure to save.
            Defaults to `"png"`. Only works if `name` is not `None`.

        Returns
        -------
        graphviz.Digraph
            The graph

        Examples
        --------
        ```python
        model = Model("y ~ x + (1|z)")
        model.fit()
        model.graph()
        ```
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

    @property
    def formula(self):
        return self._formula

    @formula.setter
    def formula(self, value):
        if isinstance(value, str):
            self._formula = Formula(value)
        elif isinstance(value, Formula):
            self._formula = value
        else:
            raise ValueError("'.formula' must be instance of 'str' or 'bambi.Formula'")

    def __str__(self):
        # Empty list with the output parameters
        output_list = []

        # Build header
        parent_name = self.family.likelihood.parent
        formulas = self.formula.get_all_formulas()
        family_name = self.family.name
        parent_parameter = self.parameters[parent_name]

        links = [
            f"{key} = {value.name}"
            for key, value in self.family.link.items()
            if key == parent_name or key in self.conditional_parameters
        ]
        observations = self.response_term.data.shape[0]

        header_dict = {
            "Formula: ": formulas,
            "Family: ": family_name,
            "Link: ": links,
            "Observations: ": str(observations),
            "Priors: ": "",
        }

        width = 16
        spacer = "\n" + " " * width
        for key, value in header_dict.items():
            output_list.append(key.rjust(width) + spacer.join(listify(value)))

        # Build priors section. Make sure the parent parameter goes first.
        priors_dict = {parent_name: make_priors_summary(parent_parameter)}

        for name, parameter in self.conditional_parameters.items():
            if parameter.is_parent:
                continue
            priors_dict[name] = make_priors_summary(parameter)

        if self.marginal_parameters:
            aux_str = "\n".join(
                [prior_repr(parameter) for parameter in self.marginal_parameters.values()]
            )
            aux_str = "Auxiliary parameters\n" + wrapify(indentify(aux_str, 4), 100, 4)
            priors_dict[parent_name] = priors_dict[parent_name] + "\n\n" + aux_str

        for key, value in priors_dict.items():
            priors_dict[key] = indentify(value, 4)

        for key, value in priors_dict.items():
            output_list.append(indentify(f"target = {key}" + "\n" + value, 4))

        if self.backend and self.backend.fit:
            foot_list = [
                "------",
                "* To see a plot of the priors call the .plot_priors() method.",
                "* To see a summary or plot of the posterior pass the object returned by .fit() to "
                "az.summary() or az.plot_trace()",
            ]
            output_list.extend(foot_list)

        return "\n".join(output_list)

    def __repr__(self):
        return self.__str__()

    @property
    def marginal_parameters(self):
        return {k: v for k, v in self.parameters.items() if isinstance(v, MarginalParameter)}

    @property
    def conditional_parameters(self):
        return {k: v for k, v in self.parameters.items() if isinstance(v, ConditionalParameter)}


def with_categorical_cols(data: pd.DataFrame, columns) -> pd.DataFrame:
    """Convert selected columns of a DataFrame to categorical type.

    It converts all object columns plus columns specified in the `columns` argument.
    """
    # Convert 'object' and explicitly asked columns to categorical.
    object_columns = list(data.select_dtypes("object").columns)
    to_convert = list(set(object_columns + listify(columns)))
    if to_convert:
        data = data.copy()  # don't modify original data frame
        data[to_convert] = data[to_convert].apply(lambda x: x.astype("category"))
    return data


def prior_repr(term) -> str:
    """Get a string representation of a Bambi term."""
    return f"{term.name} ~ {term.prior}"


def hsgp_repr(term) -> str:
    """Get a string representation of a Bambi HSGP term."""
    output_list = [
        f"cov: {term.cov}",
        *[f"{key} ~ {value}" for key, value in term.prior.items()],
    ]
    output_list = ["    " + element for element in output_list]
    output_list.insert(0, term.name)
    return "\n".join(output_list)


def make_priors_summary(parameter: ConditionalParameter) -> str:
    """Get a summary of terms and priors in a conditional parameter."""
    # Common effects
    priors_common = [
        prior_repr(term) for term in parameter.common_terms.values() if term.kind != "offset"
    ]
    if parameter.intercept_term:
        priors_common.insert(0, prior_repr(parameter.intercept_term))

    # Group-specific effects
    priors_group = [prior_repr(term) for term in parameter.group_specific_terms.values()]

    # Offsets
    offsets = [f"{term.name} ~ 1" for term in parameter.offset_terms.values()]

    # HSGP
    hsgp = [hsgp_repr(term) for term in parameter.hsgp_terms.values()]

    priors_dict = {
        "Common-level effects": priors_common,
        "Group-level effects": priors_group,
        "Offset effects": offsets,
        "HSGP contributions": hsgp,
    }

    priors_list = []
    for group, priors in priors_dict.items():
        if priors:
            priors_list.append(group + "\n" + wrapify(indentify("\n".join(priors), 4), 100, 4))

    return "\n\n".join(priors_list)
