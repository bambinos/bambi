import inspect

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.utils import (
    get_distribution_from_likelihood,
    get_distribution_from_prior,
    get_linkinv,
    make_weighted_distribution,
    GP_KERNELS,
)
from bambi.families.multivariate import MultivariateFamily
from bambi.families.univariate import Categorical, Cumulative, StoppingRatio
from bambi.priors import Prior


class CommonTerm:
    """Representation of a common effects term in PyMC

    It builds the PyMC distribution for a common effects term.

    Parameters
    ----------
    term : bambi.terms.Term
        An object representing a common effects term.
    """

    def __init__(self, term):
        self.term = term
        self.coords = self.term.coords.copy()
        if self.coords and self.term.alias:
            self.coords[self.term.alias + "_dim"] = self.coords.pop(self.term.name + "_dim")

    def build(self, spec):
        """Build term.

        Parameters
        ----------
        spec : bambi.Model
            The model instance.

        Returns
        -------
        coef : pm.Distribution
            A distribution of shape `(1, )`, `(p_j, )`, `(1, K)`, or `(p_j, K)`.
        """
        label = self.name
        kwargs = self.term.prior.args
        distribution = get_distribution_from_prior(self.term.prior)

        # Dims of the term
        term_dims = list(self.coords)

        # Dims of the response variable
        response_dims = []
        if isinstance(spec.family, (MultivariateFamily, Categorical)):
            # NOTE: Very opaque. The dictionary contains at most one element.
            response_dims = list(spec.response_component.term.coords)
            response_dims_n = len(spec.response_component.term.coords[response_dims[0]])

            # Arguments may be of shape (p_j,) but we need them to be of shape (p_j, K)
            # p_j: length of predictor coordinates
            # K: length of response coordinates
            for key, value in kwargs.items():
                # NOTE: The case value.ndim == 0 is handled below
                if value.ndim == 1:
                    kwargs[key] = np.hstack([value[:, np.newaxis]] * response_dims_n)

        if response_dims and term_dims:
            # shape: (p_j, K)
            coef = distribution(label, dims=term_dims + response_dims, **kwargs)
        elif response_dims:
            # shape: (1, K)
            coef = distribution(label, dims=response_dims, **kwargs)[np.newaxis, :]
        elif term_dims:
            # shape: (p_j, )
            coef = distribution(label, dims=term_dims, **kwargs)
        else:
            # shape: (1, )
            coef = pt.atleast_1d(distribution(label, **kwargs))

        return coef

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name


class GroupSpecificTerm:
    """Representation of a group specific effects term in PyMC

    It builds the PyMC distribution for a group-specific effects term.

    Parameters
    ----------
    term : bambi.terms.GroupSpecificTerm
        An object representing a group specific effects term.
    noncentered : bool
        Specifies if we use non-centered parametrization of group-specific effects.
    """

    def __init__(self, term, noncentered):
        self.term = term
        self.noncentered = noncentered

    @property
    def coords(self):
        coords = self.term.coords.copy()
        # If there's no alias, return the coords from the underlying term
        if not self.term.alias:
            return coords

        # If there's an alias, create a coords where the name is based on the alias
        new_coords = {}
        for key, value in coords.items():
            _, kind = key.split("__")
            new_coords[self.term.alias + "__" + kind] = value
        return new_coords

    def build(self, spec):
        """Build term.

        Parameters
        ----------
        spec : bambi.Model
            The model instance.

        Returns
        -------
        coef : pm.Distribution
            A PyMC distribution of shape `(f_j, )`, `(e_j, f_j)`, `(f_j, K)` or `(e_j, f_j, K)`.
        """
        # Dims of the term (factor_dims or factor_dims + expr_dims)
        term_dims = list(self.coords)

        # Dims of the response variable
        response_dims = []
        if isinstance(spec.family, (MultivariateFamily, Categorical)):
            response_dims = list(spec.response_component.term.coords)

        dims = term_dims + response_dims

        # Possible output shapes
        # * (f_j, e_j, K): when factor_dims, expr_dims and response_dims
        # * (f_j, K):      when factor_dims and response_dims
        # * (f_j, e_j):    when factor_dims and expr_dims
        # * (f_j, ):       when factor_dims
        return self.build_distribution(prior=self.term.prior, label=self.name, dims=dims)

    def build_distribution(self, prior, label, dims=None):
        """_summary_

        Parameters
        ----------
        prior : bambi.priors.Prior
            The prior distribution.
        label : str
            Name of the distribution.
        dims : Sequence[str], optional
            Sequence of dimension names, by default None.

        Returns
        -------
        pm.Distribution
            The PyMC distribution for the corresponding group-specific term.
        """
        # Keep all dims except of `"{name}__factor_dim"`.
        # The value in `dims` can be:
        # * The expression is vector-valued:               ["{name}__expr_dim"]
        # * The response is vector-valued:                 ["{response_dim}"]
        # * The expression and response are vector-valued: ["{name}__expr_dim", "{response_dim}"]
        # * The expression is scalar and the response is univariate: None
        if dims is not None:
            hyperparams_dims = [dim for dim in dims if not dim.endswith("__factor_dim")]
        else:
            hyperparams_dims = None

        dist_kwargs = {}
        for key, value in prior.args.items():
            if isinstance(value, Prior):
                hyperparam_key = self.term.hyperprior_alias.get(key, key)
                hyperparam_label = f"{label}_{hyperparam_key}"
                dist_kwargs[key] = self.build_distribution(
                    prior=value,
                    label=hyperparam_label,
                    dims=hyperparams_dims,
                )
            else:
                dist_kwargs[key] = value

        if self.noncentered and any(isinstance(v, pt.TensorVariable) for v in dist_kwargs.values()):
            # non-centered is only relevant when distribution arguments are random variables.
            if (
                prior.name == "Normal"
                and "sigma" in dist_kwargs
                and isinstance(dist_kwargs["sigma"], pt.TensorVariable)
            ):
                sigma = dist_kwargs["sigma"]
                offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=dims)
                return pm.Deterministic(label, offset * sigma, dims=dims)

            raise NotImplementedError(
                "The non-centered parametrization is only supported for Normal priors"
            )

        distribution = get_distribution_from_prior(prior)
        return distribution(label, **dist_kwargs, dims=dims)

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name


class InterceptTerm:
    """Representation of an intercept term in a PyMC model.

    Parameters
    ----------
    term : bambi.terms.Term
        An object representing the intercept.
    """

    def __init__(self, term):
        self.term = term

    def build(self, spec):
        """Build term.

        Parameters
        ----------
        spec : bambi.Model
            The model instance.

        Returns
        -------
        dist : pm.Distribution
            A PyMC distribution of shape `(1, )` or `(1, K)`.
        """
        distribution = get_distribution_from_prior(self.term.prior)
        label = self.name

        if isinstance(spec.family, (MultivariateFamily, Categorical)):
            # shape: (1, K)
            dims = list(spec.response_component.term.coords)
            dist = distribution(label, dims=dims, **self.term.prior.args)[np.newaxis, :]
        else:
            # shape: (1,)
            dist = pt.atleast_1d(distribution(label, **self.term.prior.args))
        return dist

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name


class ResponseTerm:
    """Representation of a response term in a PyMC model

    Parameters
    ----------
    term : bambi.terms.ResponseTerm
        The response term as represented in Bambi.
    family : bambi.famlies.Family
        The model family.
    """

    def __init__(self, term, family):
        self.term = term
        self.family = family

    def build(self, pymc_backend, bmb_model):
        """Create and return the response distribution for the PyMC model

        Parameters
        ----------
        pymc_backend : bambi.backend.PyMCModel
            The object with all the backend information
        bmb_model : bambi.Model
            The Bambi model instance.

        Returns
        -------
        dist : pm.Distribution
            The response distribution.
        """
        data = np.squeeze(self.term.data)
        parent_name = self.family.likelihood.parent

        # Auxiliary parameters and data
        kwargs = {"observed": data, "dims": ("__obs__",)}

        if isinstance(self.family, (MultivariateFamily, Categorical, Cumulative, StoppingRatio)):
            response_term = bmb_model.response_component.term
            response_name = response_term.alias or response_term.name
            dim_name = response_name + "_dim"
            pymc_backend.model.add_coords({dim_name: response_term.levels})
            dims = ("__obs__", dim_name)

            # For multivariate families, the outcome variable has two dimensions too.
            if isinstance(self.family, MultivariateFamily):
                kwargs["dims"] = dims
        else:
            dims = ("__obs__",)

        # Constant parameters. No link function is used.
        for name, component in pymc_backend.constant_components.items():
            kwargs[name] = component.output

        # Distributional parameters. A link function is used.
        for name, component in pymc_backend.distributional_components.items():
            bmb_component = bmb_model.components[name]
            aliased_name = bmb_component.alias or bmb_component.name
            linkinv = get_linkinv(self.family.link[name], pymc_backend.INVLINKS)

            # Transform the linear predictor of the parent parameter (usually the mean)
            if name == parent_name and hasattr(self.family, "transform_backend_eta"):
                output = self.family.transform_backend_eta(component.output, kwargs)
            else:
                output = component.output

            kwargs[name] = pm.Deterministic(aliased_name, linkinv(output), dims=dims)

        # Build the response distribution
        dist = self.build_distribution(kwargs, pymc_backend)

        return dist

    def build_distribution(self, kwargs, pymc_backend):
        # Get likelihood distribution
        distribution = get_distribution_from_likelihood(self.family.likelihood)

        # Families can implement specific transformations of parameters that are passed to the
        # likelihood function
        if hasattr(self.family, "transform_backend_kwargs"):
            kwargs = self.family.transform_backend_kwargs(kwargs)

        kwargs = self.robustify_dims(pymc_backend, kwargs)

        # Handle censoring
        if self.term.is_censored:
            dims = kwargs.pop("dims", None)
            data_matrix = kwargs.pop("observed")

            # Get values of the response variable
            observed = np.squeeze(data_matrix[:, 0])

            # Get censoring codes
            censoring_code = np.squeeze(data_matrix[:, 1])

            is_left_censored = censoring_code == -1
            is_right_censored = censoring_code == 1

            lower = np.where(is_left_censored, observed, -np.inf)
            upper = np.where(is_right_censored, observed, np.inf)
            stateless_dist = distribution.dist(**kwargs)
            dist_rv = pm.Censored(
                self.name, stateless_dist, lower=lower, upper=upper, observed=observed, dims=dims
            )

        # Handle truncation
        elif self.term.is_truncated:
            dims = kwargs.pop("dims", None)
            data_matrix = kwargs.pop("observed")

            # Get values of the response variable
            observed = np.squeeze(data_matrix[:, 0])

            # Get truncation values
            lower = np.squeeze(data_matrix[:, 1])
            upper = np.squeeze(data_matrix[:, 2])

            # Handle 'None' and scalars appropriately
            if np.all(lower == -np.inf):
                lower = None
            elif np.all(lower == lower[0]):
                lower = lower[0]

            if np.all(upper == np.inf):
                upper = None
            elif np.all(upper == upper[0]):
                upper = upper[0]

            stateless_dist = distribution.dist(**kwargs)
            dist_rv = pm.Truncated(
                self.name, stateless_dist, lower=lower, upper=upper, observed=observed, dims=dims
            )

        # Handle constrained responses (through truncated distributions)
        elif self.term.is_constrained:
            dims = kwargs.pop("dims", None)
            data_matrix = kwargs.pop("observed")

            # Get values of the response variable
            observed = np.squeeze(data_matrix[:, 0])

            # Get truncation values
            lower = np.squeeze(data_matrix[:, 1])
            upper = np.squeeze(data_matrix[:, 2])

            # Handle 'None' and scalars appropriately
            if np.all(lower == -np.inf):
                lower = None
            elif np.all(lower == lower[0]):
                lower = lower[0]

            if np.all(upper == np.inf):
                upper = None
            elif np.all(upper == upper[0]):
                upper = upper[0]

            stateless_dist = distribution.dist(**kwargs)
            dist_rv = pm.Truncated(
                self.name, stateless_dist, lower=lower, upper=upper, observed=observed, dims=dims
            )

        # Handle weighted responses
        elif self.term.is_weighted:
            dims = kwargs.pop("dims", None)
            data_matrix = kwargs.pop("observed")

            # Get values of the response variable
            observed = np.squeeze(data_matrix[:, 0])

            # Get weights
            weights = np.squeeze(data_matrix[:, 1])

            # Get a weighted version of the response distribution
            weighted_dist = make_weighted_distribution(distribution)
            dist_rv = weighted_dist(self.name, weights, **kwargs, observed=observed, dims=dims)
        # All of the other response kinds are "not special" and thus are handled the same way
        else:
            dist_rv = distribution(self.name, **kwargs)

        return dist_rv

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name

    def robustify_dims(self, pymc_backend, kwargs):
        # It's possible the observed for the response is multidimensional,
        # but there's a single linear predictor because the family is not multivariate.
        # In this case, we add extra dimensions to avoid having shape mismatch between the data
        # and the shape implied by the `dims` we pass.

        if (
            self.term.is_censored
            or self.term.is_truncated
            or self.term.is_weighted
            or self.term.is_constrained
        ):
            return kwargs

        dims, data = kwargs["dims"], kwargs["observed"]
        dims_n = len(dims)
        ndim_diff = data.ndim - dims_n

        # TO DO: Test with multinomial regression, shouldn't be added?
        if ndim_diff > 0:
            for i in range(ndim_diff):
                axis = dims_n + i
                name = f"{self.name}_extra_dim_{i}"
                values = np.arange(np.size(data, axis=axis))
                pymc_backend.model.add_coords({name: values})
                dims = dims + (name,)
        kwargs["dims"] = dims
        return kwargs


class HSGPTerm:
    """A term that is compiled to a HSGP term in PyMC

    This instance contains information of a bambi.HSGPTerm and knows how to build the distributions
    in PyMC that represent the HSGP latent approximation.

    Parameters
    ----------
    term : bambi.terms.HSGPTerm
        An object representing a Bambi hsgp term.
    """

    def __init__(self, term):
        self.term = term
        self.coords = self.term.coords.copy()

        # Coordinates for the variable
        if not self.term.iso and self.term.shape[1] > 1:
            self.coords[f"{self.name}_var"] = np.arange(self.term.shape[1])

        if self.coords and self.term.alias:
            self.coords[f"{self.term.alias}_weights_dim"] = self.coords.pop(
                f"{self.term.name}_weights_dim"
            )
            if self.term.by_levels is not None:
                self.coords[f"{self.term.alias}_by"] = self.coords.pop(f"{self.term.name}_by")

    def build(self, spec):
        # Get the name of the term
        label = self.name

        # Get the covariance functions (it's possibly more than one)
        covariance_functions = self.get_covariance_functions()

        # Prepare dims
        coeff_dims = (f"{label}_weights_dim",)
        contribution_dims = ("__obs__",)

        # Data may be scaled so the maximum Euclidean distance between two points is 1
        if self.term.scale_predictors:
            data = self.term.data_centered / self.term.maximum_distance
        else:
            data = self.term.data_centered

        # Build HSGP and store it in the term.
        if self.term.by_levels is not None:
            coeff_dims = coeff_dims + (f"{label}_by",)
            phi_list, sqrt_psd_list = [], []
            self.term.hsgp = {}
            # Because of the filter in the loop, it will be as if the observations were sorted
            # using the values of the 'by' variable.
            # This approach helps especially when there are many groups, which causes many zeros
            # with other approaches (until PyMC and us have better support for sparse matrices)
            indexes_to_unsort = self.term.by.argsort(kind="mergesort").argsort(kind="mergesort")
            for i, level in enumerate(self.term.by_levels):
                cov_func = covariance_functions[i]
                # Notes:
                # 'm' doesn't change by group
                # We need to use list() in 'm' and 'L' because arrays are not instance of Sequence
                hsgp = pm.gp.HSGP(
                    m=list(self.term.m),
                    L=list(self.term.L[i]),
                    drop_first=self.term.drop_first,
                    cov_func=cov_func,
                )
                # Notice we pass all the values, for all the groups.
                # Then we only keep the ones for the corresponding group.
                phi, sqrt_psd = hsgp.prior_linearized(data[self.term.by == i])
                sqrt_psd_list.append(sqrt_psd)
                phi_list.append(phi.eval())

                # Store it for later usage
                self.term.hsgp[level] = hsgp
            sqrt_psd = pt.stack(sqrt_psd_list, axis=1)
        else:
            (cov_func,) = covariance_functions
            self.term.hsgp = pm.gp.HSGP(
                m=list(self.term.m),
                L=list(self.term.L[0]),
                drop_first=self.term.drop_first,
                cov_func=cov_func,
            )
            # Get prior components
            phi, sqrt_psd = self.term.hsgp.prior_linearized(data)
            phi = phi.eval()

        # Build weights coefficient
        # Handle the case where the outcome is multivariate
        if isinstance(spec.family, (MultivariateFamily, Categorical)):
            # Append the dims of the response variables to the coefficient and contribution dims
            # In general:
            # coeff_dims: ('weights_dim', ) -> ('weights_dim', f'{response}_dim')
            # contribution_dims: ('__obs__', ) -> ('__obs__', f'{response}_dim')
            response_dims = tuple(spec.response_component.term.coords)
            coeff_dims = coeff_dims + response_dims
            contribution_dims = contribution_dims + response_dims

            # Append a dimension to sqrt_psd: ('weights_dim', ) -> ('weights_dim', 1)
            sqrt_psd = sqrt_psd[:, np.newaxis]

        if self.term.centered:
            coeffs = pm.Normal(f"{label}_weights", sigma=sqrt_psd, dims=coeff_dims)
        else:
            coeffs_raw = pm.Normal(f"{label}_weights_raw", dims=coeff_dims)
            coeffs = pm.Deterministic(f"{label}_weights", coeffs_raw * sqrt_psd, dims=coeff_dims)

        # Build deterministic for the HSGP contribution
        # If there are groups, we do as many dot products as groups
        if self.term.by_levels is not None:
            contribution_list = []
            for i in range(len(self.term.by_levels)):
                contribution_list.append(phi_list[i] @ coeffs[:, i])
            # We need to unsort the contributions so they match the original data
            contribution = pt.concatenate(contribution_list)[indexes_to_unsort]
        # If there are no groups, it's a single dot product
        else:
            contribution = pt.dot(phi, coeffs)  # "@" operator is not working as expected

        output = pm.Deterministic(label, contribution, dims=contribution_dims)
        return output

    def get_covariance_functions(self):
        """Construct and return the covariance function

        This method uses the name of the covariance function to retrieve a callable that
        returns a GP kernel and the name of its parameters. Then it looks for values for the
        parameters in the dictionary of priors of the term (building PyMC distributions as needed)
        and finally it determines the value of 'input_dim' if that is required by the callable
        that produces the covariance function. If that is the case, 'input_dim' is set to the
        dimensionality of the GP component -- the number of columns in the data.

        Returns
        -------
        Sequence[pm.gp.Covariance]
            A covariance function that can be used with a GP in PyMC.
        """

        # Get the callable that creates the function
        cov_dict = GP_KERNELS[self.term.cov]
        create_covariance_function = cov_dict["fn"]
        param_names = cov_dict["params"]
        params = {}

        # Set dimensions and behavior for priors that are actually fixed (floats or ints)
        if self.term.by_levels is not None and not self.term.share_cov:
            dims = (f"{self.name}_by",)
            recycle = True
        else:
            dims = None
            recycle = False

        # Build priors and parameters
        for param_name in param_names:
            prior = self.term.prior[param_name]
            param_dims = dims
            if isinstance(prior, Prior):
                distribution = get_distribution_from_prior(prior)
                # varying lengthscale parameter
                if param_name == "ell" and not self.term.iso and self.term.shape[1] > 1:
                    if param_dims is not None:
                        param_dims = (f"{self.name}_var",) + param_dims
                    else:
                        param_dims = (f"{self.name}_var",)
                value = distribution(f"{self.name}_{param_name}", **prior.args, dims=param_dims)
            else:
                # If it's not a distribution, but a scalar...
                if recycle:
                    value = (prior,) * self.term.groups_n
                else:
                    value = prior
            params[param_name] = value

        if "input_dim" in list(inspect.signature(create_covariance_function).parameters):
            if self.term.groups_n > 1 and not self.term.share_cov:
                params["input_dim"] = np.repeat(self.term.shape[1], self.term.groups_n)
            else:
                params["input_dim"] = self.term.shape[1]

        if self.term.groups_n == 1 or self.term.share_cov:
            covariance_function = create_covariance_function(**params)
            output = [covariance_function] * self.term.groups_n
        else:
            output = []
            for i, _ in enumerate(self.term.by_levels):
                params_level = {}
                for key, value in params.items():
                    if value[..., i].ndim == 0 and isinstance(value, np.ndarray):
                        entry = value[..., i].item()
                    else:
                        entry = value[..., i]
                    params_level[key] = entry
                covariance_function = create_covariance_function(**params_level)
                output.append(covariance_function)
        return output

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name
