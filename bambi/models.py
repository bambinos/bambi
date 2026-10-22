import logging
import warnings
from copy import copy, deepcopy
from dataclasses import dataclass
from importlib.metadata import version

import formulae as fm
import numpy as np
import pandas as pd
from arviz_plots import plot_dist
from arviz_stats import residual_r2
from formulae.matrices import ResponseMatrix
from formulae.terms.call import Call
from formulae.terms.call_resolver import LazyVariable

from bambi.backend import PyMCModel
from bambi.config import config
from bambi.defaults import get_builtin_family
from bambi.parameters import (
    Conditional,
    ConditionalCoefficient,
    ConditionalParameter,
    Marginal,
    MarginalCoefficient,
    MarginalParameter,
)
from bambi.families import Family
from bambi.families.builtin import (
    AdjacentCategory,
    Bernoulli,
    ContinuationRatio,
    Cumulative,
    DirichletMultinomial,
    Multinomial,
    OrderedStereotype,
    StoppingRatio,
)
from bambi.families.types import DimType
from bambi.formula import Formula, formula_has_intercept
from bambi.nonlinear import (
    ParameterDependencyGraph,
    SUPPORTED_FUNCTIONS,
    parameter_dependency_order,
    split_predictor,
)
from bambi.scaling import scale_priors
from bambi.terms import ResponseTerm
from bambi.transformations import transformations_namespace
from bambi.utils import (
    as_dataset,
    indentify,
    listify,
    remove_common_intercept,
    wrapify,
)

_log = logging.getLogger("bambi")

ORDINAL_FAMILIES = (
    AdjacentCategory,
    ContinuationRatio,
    Cumulative,
    OrderedStereotype,
    StoppingRatio,
)

__version__ = version("bambi")


class Model:
    """Specification of model class

    Parameters
    ----------
    formula : str or Formula
        A model description written using one or more model formulas. See `Formula` for
        formula syntax and nonlinear coefficient semantics.
    data : pd.DataFrame
        A pandas dataframe containing the data on which the model will be fit, with column
        names matching variables defined in the formula.
    family : str or bambi.Family, optional
        A specification of the model family (analogous to the family object in R). Either
        a string, or an instance of class [](`bambi.Family`). If a string is passed, a
        family with the corresponding name must be defined in the defaults loaded at `Model`
        initialization. Defaults to `"gaussian"`.
    priors : dict, optional
        Optional specification of priors for one or more terms. A dictionary where the keys are
        the names of terms in the model, "common," "group_specific," or the name of a model
        component, and the values are instances of class `Prior` or `dict`. An observational model
        parameter (e.g. "sigma" when it is modeled with a formula) maps to a nested dictionary of
        the same form; a constant component name maps directly to a `Prior`, a number, or an
        array. If priors are unset, use automatic priors inspired by the R rstanarm library.
        Names that don't match any term or component are reported with a warning; set
        `bmb.config["UNUSED_PRIORS"]` to `"error"` or `"ignore"` to change that.
        Bare term priors can be combined with priors nested under the parent component. If both
        specify the same term, the nested parent prior takes precedence.
        For nonlinear models, coefficients without a formula accept a direct prior.
        Coefficients with additive formulas accept a term-prior dictionary.
        Nonlinear coefficients are scalar by default and are shared across response categories.
        A nonlinear coefficient expression can depend on a vector-valued modeled parameter.
    link : str or dict of str to str, optional
        The name of the link function to use. Valid names are `"cloglog"`, `"identity"`,
        `"inverse_squared"`, `"inverse"`, `"log"`, `"logit"`, `"probit"`, and `"softmax"`.
        Not all the link functions can be used with all the families.
        If a dictionary, keys are the names of the target parameters and the values are the names
        of the link functions.
        For nonlinear formulas, each parameter's inverse link is applied once to its complete
        predictor.
    categorical : str or list of str, optional
        The names of any variables to treat as categorical. Can be either a single variable
        name, or a list of names. If categorical is `None`, the data type of the columns in
        the `data` will be used to infer handling. In cases where numeric columns are
        to be treated as categorical (e.g., group specific factors coded as numerical IDs),
        explicitly passing variable names via this argument is recommended.
    potentials : list of 2-tuples, optional
        Optional specification of potentials. A potential is an arbitrary expression added to the
        likelihood, this is generally useful to add constraints to models, that are difficult to
        express otherwise. The first term of a 2-tuple is the name of a variable in the model, the
        second a lambda function expressing the desired constraint.
        If a constraint involves n variables, you can pass n 2-tuples or pass a tuple which first
        element is an n-tuple and second element is a lambda function with n arguments. The number
        and order of the lambda function has to match the number and order of the variable names.
    dropna : bool, optional
        When `True`, rows with any missing values in either the predictors or outcome are
        automatically dropped from the dataset in a listwise manner, optional.
    auto_scale : bool, optional
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
        if not isinstance(data, pd.DataFrame):
            raise ValueError("'data' must be a pandas DataFrame.")

        # attributes that are set later
        self.parameters = {}
        self.built = False  # build()

        self.family = None  # _add_response()
        self.backend = None  # _set_backend()

        self.auto_scale = auto_scale
        self.dropna = dropna
        self.formula = formula
        self.noncentered = noncentered
        self.potentials = potentials
        self.center_predictors = center_predictors

        # Prevent modification of user-suplied priors.
        priors = {} if priors is None else deepcopy(priors)

        # Obtain design matrices and related objects.
        na_action = "drop" if dropna else "error"

        # Handle additional namespaces
        additional_namespace = transformations_namespace.copy()
        if not isinstance(extra_namespace, (type(None), dict)):
            raise ValueError("'namespace' must be a dictionary or None")
        additional_namespace.update(extra_namespace or {})

        # Create family. Sets `family` attribute.
        self._set_family(family, link)

        # Temporary method to handle deprecated count response
        self._convert_deprecated_c_response()

        nonlinear_params = set(self.formula.nlpars)
        likelihood_params = set(self.family.likelihood.params)
        auxiliary_params = set(self.family.auxiliary_parameters)
        data_names = set(data.columns)
        marginal_params = auxiliary_params - set(self.formula.additionals_lhs)
        invalid_lhs = set(self.formula.additionals_lhs) - nonlinear_params - auxiliary_params
        invalid_nlpars = nonlinear_params & (SUPPORTED_FUNCTIONS | {"nl"})

        if nonlinear_params & data_names:
            raise ValueError(
                "Nonlinear parameter names must not also be data columns: "
                f"{sorted(nonlinear_params & data_names)}."
            )

        if nonlinear_params & likelihood_params:
            raise ValueError(
                "Nonlinear parameter names must not be observational model parameter names: "
                f"{sorted(nonlinear_params & likelihood_params)}."
            )

        if invalid_lhs:
            raise ValueError(
                f"Additional formulas contain invalid parameter names: {sorted(invalid_lhs)}."
            )

        if invalid_nlpars:
            raise ValueError(
                "Nonlinear parameter names must not be supported function names: "
                f"{sorted(invalid_nlpars)}."
            )

        # Start processing model formulas
        parent_name = self.family.likelihood.parent
        formulas = {parent_name: self.formula.main} | dict(
            zip(self.formula.additionals_lhs, self.formula.additionals)
        )

        parameter_names = nonlinear_params | (likelihood_params - data_names)
        parameters_config = {}

        for param_name, param_formula in formulas.items():
            linear_formula, nonlinear_expr = split_predictor(
                param_formula.split("~", 1)[-1], parameter_names
            )
            if nonlinear_expr is not None:
                ambiguous = nonlinear_expr.symbols & likelihood_params & data_names
                if ambiguous:
                    raise ValueError(
                        "Observational model parameter names must not also be data columns "
                        f"in nonlinear expressions: {sorted(ambiguous)}."
                    )
            parameters_config[param_name] = {
                "linear": {"formula": linear_formula},
                "nonlinear": {
                    "expr": nonlinear_expr,
                    "data_names": set(),
                    "dependencies": set(),
                },
            }

        self.data = prepare_data(
            data=data,
            formula=self.formula,
            categorical=categorical,
            dropna=dropna,
            parameters_config=parameters_config,
        )

        # Determine nonlinear parameter dependencies (other parameters) and validate data columns
        for parameter in parameters_config.values():
            nl = parameter["nonlinear"]
            if nl["expr"] is not None:
                nl["dependencies"].update(nl["expr"].symbols & parameter_names)
                nl["data_names"].update(nl["expr"].symbols - nl["dependencies"])
                unknown = nl["data_names"] - set(self.data.columns)
                nonnumeric = sorted(
                    name
                    for name in nl["data_names"] - unknown
                    if not pd.api.types.is_numeric_dtype(self.data[name])
                )
                errors = []
                if unknown:
                    errors.append(
                        "No nonlinear parameter formula or data column was found for symbol(s): "
                        f"{sorted(unknown)}."
                    )
                if nonnumeric:
                    errors.append(
                        f"Nonlinear expression data must be numeric. "
                        f"Invalid column(s): {nonnumeric}."
                    )
                if errors:
                    raise ValueError(" ".join(errors))

        # Reject declared non-linear parameters that are not used.
        used_nlpars = {
            dependency
            for parameter in parameters_config.values()
            for dependency in parameter["nonlinear"]["dependencies"]
            if dependency in nonlinear_params
        }
        unused = nonlinear_params - used_nlpars
        if unused:
            raise ValueError(
                "Nonlinear parameter name(s) not used by the expression graph: "
                f"{sorted(unused)}."
            )

        response_name = self.formula.main.partition("~")[0].strip()
        parent_linear_formula = parameters_config[parent_name]["linear"]["formula"]
        response_formula = f"{response_name} ~ {parent_linear_formula or '1'}"

        if isinstance(self.family, ORDINAL_FAMILIES):
            if parent_linear_formula is not None and formula_has_intercept(parent_linear_formula):
                warnings.warn("The intercept is omitted in ordinal families")

            # The temporary intercept constrains categorical predictors, and it's then removed.
            response_formula = response_formula + " + 1"

        response_design = fm.design_matrices(
            response_formula, self.data, na_action, 1, additional_namespace
        )

        if isinstance(self.family, ORDINAL_FAMILIES):
            response_design = remove_common_intercept(response_design)

        self.response_term = ResponseTerm(response_design.response)
        if isinstance(self.family, OrderedStereotype) and len(self.response_term.levels or []) < 3:
            raise ValueError("The 'osm' family requires at least three response categories.")

        self._response_component = _ResponseComponentAdapter(
            self.response_term, response_design.response, self
        )

        if self.response_term.is_competing_risks:
            # Competing risks use a model-local family with cause-specific parameters.
            self.family = copy(self.family)
            self.family.PARAMETERS = {
                name: self.family.get_param_spec(name)._replace(ndim=1, coefs_dim=DimType.RESPONSE)
                for name in self.family.likelihood.params
            }

        dependencies = dict.fromkeys(self.family.likelihood.params, ())
        dependencies.update(dict.fromkeys(self.formula.nlpars, ()))
        for name, parameter_info in parameters_config.items():
            dependencies[name] = tuple(sorted(parameter_info["nonlinear"]["dependencies"]))

        # Ordinal's parent usually depends on auxiliary parameters.
        # We should device a more general mechanism to specify dependencies between parameters.
        # In this case, they're family related. In others, they could be prior related.
        if isinstance(self.family, ORDINAL_FAMILIES):
            dependencies[parent_name] = tuple(
                sorted(set(dependencies[parent_name]) | auxiliary_params)
            )

        # Sorting allows reproducibility
        declaration_order = (
            tuple(sorted(marginal_params))
            + (parent_name,)
            + tuple(sorted(auxiliary_params - marginal_params))
            + tuple(sorted(self.formula.nlpars))
        )
        parameter_order = parameter_dependency_order(dependencies, declaration_order)

        # Build non-linear coefficients and parameters in the appropriate order
        nonlinear_coefficients = {}
        for name in parameter_order:
            if name in marginal_params:
                self.parameters[name] = MarginalParameter(name, priors.get(name), self)
                continue

            if name in nonlinear_params and name not in parameters_config:
                nonlinear_coefficients[name] = MarginalCoefficient(name, priors.get(name), self)
                continue

            parameter_info = parameters_config[name]
            nlexpr = parameter_info["nonlinear"]["expr"]
            linear_formula = parameter_info["linear"]["formula"]

            if linear_formula is None:
                design = None
            elif name == parent_name:
                design = response_design
            else:
                design = fm.design_matrices(
                    linear_formula, self.data, na_action, 1, additional_namespace
                )

            parameter_priors = {}
            if design is not None:
                parameter_priors = priors.get(name, {})
                if not isinstance(parameter_priors, dict):
                    raise ValueError(
                        f"Priors for conditional parameter '{name}' must be a dictionary."
                    )

                if name == parent_name:
                    bare_priors = {
                        key: value for key, value in priors.items() if key not in parameter_names
                    }
                    parameter_priors = bare_priors | parameter_priors

            if name in nonlinear_params:
                nonlinear_coefficients[name] = ConditionalCoefficient(
                    name=name,
                    design=design,
                    priors=parameter_priors,
                    spec=self,
                    expression=nlexpr,
                    data_names=sorted(parameter_info["nonlinear"]["data_names"]),
                    nonlinear_coefficients={
                        dependency: nonlinear_coefficients[dependency]
                        for dependency in dependencies[name]
                        if dependency in nonlinear_params
                    },
                )
            else:
                self.parameters[name] = ConditionalParameter(
                    name=name,
                    design=design,
                    priors=parameter_priors,
                    spec=self,
                    is_parent=name == parent_name,
                    expression=nlexpr,
                    data_names=sorted(parameter_info["nonlinear"]["data_names"]),
                    nonlinear_coefficients={
                        dependency: nonlinear_coefficients[dependency]
                        for dependency in dependencies[name]
                        if dependency in nonlinear_params
                    },
                )

        self.parameter_graph = ParameterDependencyGraph(
            nodes=self.parameters | nonlinear_coefficients,
            dependencies=dependencies,
            order=parameter_order,
        )

        # Validate prior names, now that every component and its terms are known.
        self._check_prior_names(priors)

        # Validate per-parameter noncentered dict, now that all parameters are known.
        if isinstance(self.noncentered, dict):
            valid_parameters = set(self._quantities)
            unknown = set(self.noncentered) - valid_parameters
            if unknown:
                raise ValueError(
                    f"Unknown parameter name(s) in `noncentered`: {sorted(unknown)}. "
                    f"Valid parameter names for this model: {sorted(valid_parameters)}."
                )

        # Build priors
        self._build_priors()

    @property
    def _quantities(self):
        """All observational-model parameters and their shared nonlinear coefficients."""
        return self.parameter_graph.nodes

    @property
    def _marginal_quantities(self):
        """Parameters and coefficients with direct priors."""
        return {
            name: quantity
            for name, quantity in self._quantities.items()
            if isinstance(quantity, Marginal)
        }

    def _quantities_with_terms(self):
        """Return conditional quantities backed by additive designs."""
        return {
            name: quantity
            for name, quantity in self._quantities.items()
            if isinstance(quantity, Conditional) and quantity.design is not None
        }

    def get_terms(self):
        """Return formula terms from all conditional quantities."""
        terms = {}
        for quantity in self._quantities_with_terms().values():
            if quantity.design.common:
                terms.update(quantity.design.common.terms)
            if quantity.design.group:
                terms.update(quantity.design.group.terms)
        return terms

    def get_covariates(self):
        """Return observed covariate names from formulas and expressions."""
        terms = self.get_terms()
        covariates = []
        for term in terms.values():
            if hasattr(term, "components"):
                for component in term.components:
                    # If the component is a function call, look for relevant argument names
                    if isinstance(component, Call):
                        # Add variable names passed as unnamed arguments
                        covariates.extend(
                            arg.name for arg in component.call.args if isinstance(arg, LazyVariable)
                        )
                        # Add variable names passed as named arguments
                        covariates.extend(
                            kwarg_value.name
                            for kwarg_value in component.call.kwargs.values()
                            if isinstance(kwarg_value, LazyVariable)
                        )
                    else:
                        covariates.append(component.name)
            elif hasattr(term, "factor"):
                covariates.extend(list(term.var_names))

        for parameter in self.parameter_graph.expression_nodes.values():
            covariates.extend(parameter.data_names)

        # Don't include non-covariate names (#797)
        covariates = [name for name in covariates if name in self.data]

        return np.unique(covariates)

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
            For other kwargs see the documentation for `pm.sample()`.

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

        # Arguments `common` and `group_specific` only affect the parent parameter.
        parent_name = self.family.likelihood.parent

        parent_parameter = self.parameters[parent_name]
        if parent_parameter.design is None and (common is not None or group_specific is not None):
            raise ValueError(
                "Set priors by coefficient name instead of 'common' or 'group_specific'."
            )
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
            parameters = self._quantities_with_terms() | self._marginal_quantities
            normalized_priors = {name: priors[name] for name in parameters if name in priors}
            if parent_parameter.design is not None:
                parent_priors = {
                    name: prior for name, prior in priors.items() if name not in parameters
                }
                if parent_name in normalized_priors:
                    parent_priors.update(normalized_priors[parent_name])
                normalized_priors[parent_name] = parent_priors

            # Make sure mutation of Prior objects within update_priors does not have side effects.
            normalized_priors = deepcopy(normalized_priors)

            for name, component in parameters.items():
                prior = normalized_priors.get(name)
                if prior is not None:
                    component.update_priors(prior)

        self._build_priors()
        self.built = False

    def _build_priors(self):
        """Carry out all operations related to the construction and/or scaling of priors."""
        # Prepare all priors
        quantities_with_terms = self._quantities_with_terms()
        for parameter in quantities_with_terms.values():
            parameter.build_priors()

        for quantity in self._marginal_quantities.values():
            quantity.build_priors()

        # Scale priors if there is at least one term in the model and auto_scale is True
        if self.auto_scale:
            scale_priors(self)

    def _check_prior_names(self, priors):
        """Report names in `priors` that don't match any term or parameter."""
        behavior = config["UNUSED_PRIORS"]
        if behavior == "ignore":
            return

        quantities_with_terms = self._quantities_with_terms()
        valid = set(self._marginal_quantities) | set(quantities_with_terms)
        parent_parameter = self.parameters[self.family.likelihood.parent]
        if parent_parameter.design is not None:
            valid |= set(parent_parameter.terms) | {"common", "group_specific"}

        unused = []
        for name, value in priors.items():
            if name not in valid:
                unused.append(name)
            elif name in quantities_with_terms:
                if not isinstance(value, dict):
                    raise ValueError(
                        f"Priors for conditional parameter '{name}' must be a dictionary."
                    )
                nested_valid = set(quantities_with_terms[name].terms) | {
                    "common",
                    "group_specific",
                }
                for nested_name in set(value) - nested_valid:
                    unused.append(f"{name}.{nested_name}")

        if not unused:
            return

        message = f"Unused name(s) in `priors`: {sorted(unused)}."
        hint = ' Set `bmb.config["UNUSED_PRIORS"]` to change this.'
        if behavior == "warn":
            warnings.warn(message + " They will be ignored." + hint, UserWarning)
        else:
            raise ValueError(f"{message} Valid names for this model: {sorted(valid)}.{hint}")

    def _set_family(self, family, link):
        """Set the Family of the model

        Parameters
        ----------
        family : str or Family
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

    def _convert_deprecated_c_response(self):
        """Replace the deprecated `c(...)` count response with `counts(...)`."""
        if not isinstance(self.family, (Multinomial, DirichletMultinomial)):
            return

        response = fm.model_description(self.formula.main).response
        if response is None or len(response.term.components) != 1:
            return

        component = response.term.components[0]
        call = getattr(component, "call", None)
        if call is None or call.callee != "c":
            return

        lhs, separator, rhs = self.formula.main.partition("~")
        lhs = lhs.replace("c", "counts", 1)
        self.formula = Formula(
            lhs + separator + rhs, *self.formula.additionals, nlpars=self.formula.nlpars
        )
        warnings.warn(
            f"Using 'c(...)' as the response for the '{self.family.name}' family is deprecated. "
            "Use 'counts(...)' instead.",
            FutureWarning,
            stacklevel=3,
        )

    def set_alias(self, aliases):
        """Set aliases for the terms and auxiliary parameters in the model

        Parameters
        ----------
        aliases : dict
            Map original names to aliases. For distributional and nonlinear models, use nested
            dictionaries keyed by modeled parameter names. Inside each dictionary, map term names
            or the parameter's own name to strings. Response and marginal parameter aliases use
            strings directly, as do marginal coefficients. For example,
            `{"a": {"a": "baseline", "Intercept": "a0"},
            "mu": {"mu": "mean"}, "y": "response"}` aliases a nonlinear coefficient, its
            intercept, the parent, and the response. Formulas and prior dictionaries continue to
            use original names. The model must be rebuilt after setting aliases.

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
        # pylint: disable=too-many-nested-blocks
        if len(self.conditional_parameters) == 1 and not self.parameter_graph.expression_nodes:
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
            modeled_parameters = self._quantities
            for parameter_name, parameter_aliases in aliases.items():
                if parameter_name in self._marginal_quantities:
                    assert isinstance(parameter_aliases, str)
                    self._marginal_quantities[parameter_name].alias = parameter_aliases
                elif parameter_name in (self.response_term.name, self.response_term.full_name):
                    assert isinstance(parameter_aliases, str)
                    self.response_term.alias = parameter_aliases
                else:
                    assert isinstance(parameter_aliases, dict)
                    if parameter_name not in modeled_parameters:
                        missing_names.append(parameter_name)
                        continue
                    parameter = modeled_parameters[parameter_name]
                    for name, alias in parameter_aliases.items():
                        assert isinstance(alias, str), "Alias must be a string"
                        is_used = False

                        if name in getattr(parameter, "terms", {}):
                            parameter.terms[name].alias = alias
                            is_used = True

                        # Useful for non-response conditional parameters
                        if name == parameter.name:
                            parameter.alias = alias
                            is_used = True

                        for term in getattr(parameter, "group_specific_terms", {}).values():
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

    def plot_priors(  # pylint: disable=too-many-positional-arguments
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
            Otherwise, it uses `arviz_base.rcParams["plot.backend"]`.
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

        pc = None
        prior_dt = self.backend.prior_predictive(
            draws=draws,
            var_names=var_names,
            omit_offsets=omit_offsets,
            omit_group_specific=omit_group_specific,
            prior_only=True,
            random_seed=random_seed,
        )

        if prior_dt is not None:
            var_names = list(as_dataset(prior_dt["prior"]).data_vars)
            # Sort variable names so Intercept is in the beginning
            if "Intercept" in var_names:
                var_names.insert(0, var_names.pop(var_names.index("Intercept")))

            pc = plot_dist(
                prior_dt,
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

        return self.backend.prior_predictive(
            draws=draws,
            var_names=var_names,
            omit_offsets=omit_offsets,
            random_seed=random_seed,
        )

    def predict(
        self,
        idata,
        kind="response_params",
        data=None,
        inplace=True,
        include_group_specific=True,
        sample_new_groups=None,
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
        sample_new_groups : bool or None, optional
            Deprecated and has no effect. Bambi automatically determines how to predict for new
            groups based on the type of group. Defaults to `None`.
        random_seed : int, RandomState or Generator, optional
            Seed for the random number generator.
        progressbar : bool, optional
            Whether to display a progress bar. Defaults to `False`.

        Returns
        -------
        DataTree or None
        """
        if sample_new_groups is not None:
            warnings.warn(
                "'sample_new_groups' is deprecated and has no effect. Bambi now automatically "
                "determines how to predict for new groups based on the type of group.",
                FutureWarning,
                stacklevel=2,
            )

        if kind not in (
            "response_params",
            "response",
            "response_conditional",
            "time_and_cause",
        ):
            raise ValueError(
                "'kind' must be one of 'response_params', 'response', 'response_conditional', "
                "or 'time_and_cause'"
            )

        if kind == "time_and_cause" and not self.response_term.is_competing_risks:
            raise ValueError("'kind=time_and_cause' is only available for competing-risks models.")

        if kind == "response_conditional" and not (
            self.response_term.is_censored
            or self.response_term.is_truncated
            or self.response_term.is_competing_risks
        ):
            raise ValueError(
                "'kind=response_conditional' is only available for censored, truncated, "
                "or competing-risks models."
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

        graphviz = self.backend.graph(formatting=formatting)
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
        if self.parameter_graph.expression_nodes:
            quantities_with_terms = self._quantities_with_terms()
            priors_dict = {
                parameter.label: make_priors_summary(parameter)
                for parameter in quantities_with_terms.values()
            }
        else:
            priors_dict = {parent_name: make_priors_summary(parent_parameter)}

            for name, parameter in self.conditional_parameters.items():
                if parameter.is_parent:
                    continue
                priors_dict[name] = make_priors_summary(parameter)

        for quantity in self._marginal_quantities.values():
            if isinstance(quantity, MarginalCoefficient):
                priors_dict[quantity.label] = prior_repr(quantity)

        if self.marginal_parameters:
            aux_str = "\n".join(
                [prior_repr(parameter) for parameter in self.marginal_parameters.values()]
            )
            aux_str = "Auxiliary parameters\n" + wrapify(indentify(aux_str, 4), 100, 4)
            key = parent_parameter.label if self.parameter_graph.expression_nodes else parent_name
            if key in priors_dict:
                priors_dict[key] += "\n\n" + aux_str
            else:
                priors_dict[key] = aux_str

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
    def components(self):
        warnings.warn(
            "'components' is deprecated. Use 'parameters' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.parameters

    @property
    def distributional_components(self):
        warnings.warn(
            "'distributional_components' is deprecated. Use 'conditional_parameters' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.conditional_parameters

    @property
    def constant_components(self):
        warnings.warn(
            "'constant_components' is deprecated. Use 'marginal_parameters' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self.marginal_parameters

    @property
    def response_component(self):
        warnings.warn(
            "'response_component' is deprecated. Use 'response_term' instead.",
            FutureWarning,
            stacklevel=2,
        )
        return self._response_component

    @property
    def marginal_parameters(self):
        return {k: v for k, v in self.parameters.items() if isinstance(v, MarginalParameter)}

    @property
    def conditional_parameters(self):
        """Return likelihood parameters defined by formulas.

        Returns
        -------
        dict of str to ConditionalParameter
            Likelihood parameters keyed by their original names. Intermediate nonlinear
            quantities are excluded.

        Examples
        --------
        For a Gaussian model with an ordinary formula, the mapping contains the formula-defined
        parent `mu`. The unmodeled auxiliary `sigma` remains a marginal parameter.

        >>> import bambi as bmb
        >>> import pandas as pd
        >>> model = bmb.Model("y ~ x", pd.DataFrame({"y": [1, 2], "x": [0, 1]}))
        >>> set(model.conditional_parameters)
        {'mu'}
        """
        return {k: v for k, v in self.parameters.items() if isinstance(v, ConditionalParameter)}


def prepare_data(
    data: pd.DataFrame,
    formula,
    parameters_config: dict,
    categorical: str | list[str] | None = None,
    dropna: bool = False,
) -> pd.DataFrame:
    """Convert categorical columns and align complete observations across model formulas.

    Parameters
    ----------
    data : pandas.DataFrame
        Original model data.
    formula : Formula
        Response and parameter formulas, with or without nonlinear expressions.
    parameters_config : dict
        Parameter names mapped to their linear formulas and nonlinear expressions.
    categorical : str or list of str, optional
        Columns to convert to categorical, in addition to all object columns.
    dropna : bool, optional
        Whether to remove rows with missing values in model inputs instead of raising an error.

    Returns
    -------
    pandas.DataFrame
        Prepared data. The supplied DataFrame is not modified.

    Raises
    ------
    ValueError
        If a required response is not specified, inputs contain missing values with
        `dropna=False`, or no complete observations remain.
    """
    # Convert object columns and requested categorical columns to categorical
    object_columns = list(data.select_dtypes("object").columns)
    to_convert = list(set(object_columns + listify(categorical)))
    if to_convert:
        data = data.copy()
        data[to_convert] = data[to_convert].apply(lambda x: x.astype("category"))

    data_vars = set()
    for parameter in parameters_config.values():
        nlexpr = parameter["nonlinear"]["expr"]
        if nlexpr is not None:
            data_vars.update(nlexpr.symbols)

        linear_formula = parameter["linear"]["formula"]
        if linear_formula is not None:
            data_vars.update(fm.model_description(linear_formula).var_names)

    lhs, separator, _ = formula.main.partition("~")
    response_formula = f"{lhs} ~ 1" if separator else formula.main
    response = fm.model_description(response_formula).response
    if response is None:
        raise ValueError(
            "No outcome variable is set! "
            "Please specify an outcome variable using the formula interface."
        )
    # `var_names` could have multiple names (i.e. counts(y1, y2, y3))
    data_vars.update(response.var_names)

    columns = list(data_vars & set(data.columns))
    incomplete = data[columns].isna().any(axis=1)
    if incomplete.any():
        if not dropna:
            raise ValueError(f"'data' contains {incomplete.sum()} incomplete rows.")
        _log.info(
            "Automatically removing %s/%s rows from the dataset.", incomplete.sum(), len(data)
        )
        data = data.loc[~incomplete].copy()

    if len(data) == 0:
        raise ValueError("'data' does not contain any complete observation.")

    return data


def prior_repr(term) -> str:
    """Get a string representation of a Bambi term."""
    return f"{term.label} ~ {term.prior}"


def hsgp_repr(term) -> str:
    """Get a string representation of a Bambi HSGP term."""
    output_list = [
        f"cov: {term.cov}",
        *[f"{key} ~ {value}" for key, value in term.prior.items()],
    ]
    output_list = ["    " + element for element in output_list]
    output_list.insert(0, term.name)
    return "\n".join(output_list)


def smooth_repr(term) -> str:
    """Format a smooth term with one indented line per prior block."""
    blocks = [
        f"    {name} ~ {term.prior[name]}"
        for name in ("constant", "linear", "curvature")
        if name in term.prior
    ]
    return "\n".join([term.name, *blocks])


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

    # Smooths
    smooths = [smooth_repr(term) for term in parameter.smooth_terms.values()]

    priors_dict = {
        "Common-level effects": priors_common,
        "Group-level effects": priors_group,
        "Offset effects": offsets,
        "HSGP contributions": hsgp,
        "Smooth contributions": smooths,
    }

    priors_list = []
    for group, priors in priors_dict.items():
        if priors:
            priors_list.append(group + "\n" + wrapify(indentify("\n".join(priors), 4), 100, 4))

    return "\n\n".join(priors_list)


@dataclass
class _ResponseComponentAdapter:
    term: ResponseTerm
    response: ResponseMatrix
    spec: Model
