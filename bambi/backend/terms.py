import inspect

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from bambi.backend.utils import (
    has_hyperprior,
    get_distribution_from_prior,
    get_distribution_from_likelihood,
    get_linkinv,
    GP_KERNELS,
)
from bambi.families.multivariate import MultivariateFamily, Multinomial, DirichletMultinomial
from bambi.families.univariate import Categorical
from bambi.priors import Prior
from bambi.utils import get_aliased_name


class CommonTerm:
    """Representation of a common effects term in PyMC

    An object that builds the PyMC distribution for a common effects term. It also contains the
    coordinates that we then add to the model.

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
        data = self.term.data
        label = self.name
        args = self.term.prior.args
        distribution = get_distribution_from_prior(self.term.prior)

        # Dims of the response variable
        response_dims = []
        if isinstance(spec.family, (MultivariateFamily, Categorical)):
            response_dims = list(spec.response_component.response_term.coords)
            response_dims_n = len(spec.response_component.response_term.coords[response_dims[0]])
            # Arguments may be of shape (a,) but we need them to be of shape (a, b)
            # a: length of predictor coordinates
            # b: length of response coordinates
            for key, value in args.items():
                if value.ndim == 1:
                    args[key] = np.hstack([value[:, np.newaxis]] * response_dims_n)

        dims = list(self.coords) + response_dims
        if dims:
            coef = distribution(label, dims=dims, **args)
        else:
            if data.ndim == 1:
                shape = None
            elif data.shape[1] == 1:
                shape = None
            else:
                shape = data.shape[1]
            coef = distribution(label, shape=shape, **args)
            coef = pt.atleast_1d(coef)  # If only a single numeric column it won't be 1D

        # Prepends one dimension if response is multivariate and the predictor is 1D
        if response_dims and len(dims) == 1:
            coef = coef[np.newaxis, :]

        return coef, data

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name


class GroupSpecificTerm:
    """Representation of a group specific effects term in PyMC

    Creates an object that builds the PyMC distribution for a group specific effect. It also
    contains the coordinates that we then add to the model.

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
        self.coords = self.get_coords()

    def build(self, spec):
        label = self.name
        kwargs = self.term.prior.args
        predictor = np.squeeze(self.term.predictor)

        # Dims of the response variable (e.g. categorical)
        response_dims = []
        if isinstance(spec.family, (MultivariateFamily, Categorical)):
            response_dims = list(spec.response_component.response_term.coords)

        dims = list(self.coords) + response_dims
        # Squeeze ensures we don't have a shape of (n, 1) when we mean (n, )
        # This happens with categorical predictors with two levels and intercept.
        coef = self.build_distribution(self.term.prior, label, dims=dims, **kwargs).squeeze()
        coef = coef[self.term.group_index]

        return coef, predictor

    def get_coords(self):
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

    def build_distribution(self, prior, label, **kwargs):
        """Build and return a PyMC Distribution."""
        distribution = get_distribution_from_prior(prior)

        if "dims" in kwargs:
            group_dim = [dim for dim in kwargs["dims"] if dim.endswith("__expr_dim")]
            kwargs = {
                k: self.expand_prior_args(k, v, label, dims=group_dim) for (k, v) in kwargs.items()
            }
        else:
            kwargs = {k: self.expand_prior_args(k, v, label) for (k, v) in kwargs.items()}

        if self.noncentered and has_hyperprior(kwargs):
            sigma = kwargs["sigma"]
            offset = pm.Normal(label + "_offset", mu=0, sigma=1, dims=kwargs["dims"])
            return pm.Deterministic(label, offset * sigma, dims=kwargs["dims"])
        return distribution(label, **kwargs)

    def expand_prior_args(self, key, value, label, **kwargs):
        # kwargs are used to pass 'dims' for group specific terms.
        if isinstance(value, Prior):
            # If there's an alias for the hyperprior, use it.
            key = self.term.hyperprior_alias.get(key, key)
            return self.build_distribution(value, f"{label}_{key}", **value.args, **kwargs)
        return value

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
        An object representing the intercept. This has ``.kind == "intercept"``
    """

    def __init__(self, term):
        self.term = term

    def build(self, spec):
        dist = get_distribution_from_prior(self.term.prior)
        label = self.name
        # Prepends one dimension if response is multivariate
        if isinstance(spec.family, (MultivariateFamily, Categorical)):
            dims = list(spec.response_component.response_term.coords)
            dist = dist(label, dims=dims, **self.term.prior.args)[np.newaxis, :]
        else:
            dist = dist(label, **self.term.prior.args)
        return dist

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name


class ResponseTerm:
    """Representation of a response term in a PyMC model.

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
        """Create and return the response distribution for the PyMC model.

        Parameters
        ----------
        pymc_backend : bambi.backend.PyMCModel
            The object with all the backend information
        bmb_model : bambi.Model
            The Bambi model instance

        Returns
        -------
        dist : pm.Distribution
            The response distribution
        """
        data = np.squeeze(self.term.data)
        parent = self.family.likelihood.parent

        # Auxiliary parameters
        kwargs = {}

        # Constant parameters. No link function is used.
        for name, component in pymc_backend.constant_components.items():
            kwargs[name] = component.output

        # Distributional parameters. A link funciton is used.
        dims = (f"{self.name}_obs",)
        for name, component in pymc_backend.distributional_components.items():
            bmb_component = bmb_model.components[name]
            if bmb_component.response_term:  # The response is added later
                continue
            aliased_name = (
                bmb_component.alias if bmb_component.alias else bmb_component.response_name
            )
            linkinv = get_linkinv(self.family.link[name], pymc_backend.INVLINKS)
            kwargs[name] = pm.Deterministic(aliased_name, linkinv(component.output), dims=dims)

        # Add observed and dims
        kwargs["observed"] = data
        kwargs["dims"] = dims

        # The linear predictor for the parent parameter (usually the mean)
        eta = pymc_backend.distributional_components[self.term.name].output

        if hasattr(self.family, "transform_backend_eta"):
            eta = self.family.transform_backend_eta(eta, kwargs)

        # Take the inverse link function that maps from linear predictor to the parent of likelihood
        linkinv = get_linkinv(self.family.link[parent], pymc_backend.INVLINKS)

        # Add parent parameter after the applying the linkinv transformation
        kwargs[parent] = linkinv(eta)

        # Build the response distribution
        dist = self.build_response_distribution(kwargs, pymc_backend)

        return dist

    def build_response_distribution(self, kwargs, pymc_backend):
        # Get likelihood distribution
        distribution = get_distribution_from_likelihood(self.family.likelihood)

        # Families can implement specific transformations of parameters that are passed to the
        # likelihood function
        if hasattr(self.family, "transform_backend_kwargs"):
            kwargs = self.family.transform_backend_kwargs(kwargs)

        kwargs = self.robustify_dims(pymc_backend, kwargs)

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
        else:
            dist_rv = distribution(self.name, **kwargs)

        return dist_rv

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name

    def robustify_dims(self, pymc_backend, kwargs):
        # It's possible the observed for the response is multidimensional, but there's a single
        # linear predictor because the family is not multivariate.
        # In this case, we add extra dimensions to avoid having shape mismatch between the data
        # and the shape implied by the `dims` we pass.

        # Don't do it for the Multinomial families (it's an exception)
        if isinstance(self.family, (Multinomial, DirichletMultinomial)):
            return kwargs

        if self.term.is_censored:
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

    def build(self, bmb_model):
        # Get the name of the term
        label = self.name

        # Get the covariance functions (it's possibly more than one)
        covariance_functions = self.get_covariance_functions()

        # Get dimension name for the response
        response_name = get_aliased_name(bmb_model.response_component.response_term)

        # Prepare dims
        coeff_dims = (f"{label}_weights_dim",)
        contribution_dims = (f"{response_name}_obs",)

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
            A covariance function that can be used with a GP in PyMC
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
