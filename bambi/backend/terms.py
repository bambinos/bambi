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
from bambi.families.multivariate import MultivariateFamily
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
            new_coords[self.term.alias + kind] = value
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

        # The linear predictor for the parent parameter (usually the mean)
        nu = pymc_backend.distributional_components[self.term.name].output

        if hasattr(self.family, "transform_backend_nu"):
            nu = self.family.transform_backend_nu(nu, data)

        # Add auxiliary parameters
        kwargs = {}

        # Constant parameters. No link function is used.
        for name, component in pymc_backend.constant_components.items():
            kwargs[name] = component.output

        # Distributional parameters. A link funciton is used.
        response_aliased_name = get_aliased_name(self.term)
        dims = [response_aliased_name + "_obs"]
        for name, component in pymc_backend.distributional_components.items():
            bmb_component = bmb_model.components[name]
            if bmb_component.response_term:  # The response is added later
                continue
            aliased_name = (
                bmb_component.alias if bmb_component.alias else bmb_component.response_name
            )
            linkinv = get_linkinv(self.family.link[name], pymc_backend.INVLINKS)
            kwargs[name] = pm.Deterministic(
                f"{response_aliased_name}_{aliased_name}", linkinv(component.output), dims=dims
            )

        # Take the inverse link function that maps from linear predictor to the parent of likelihood
        linkinv = get_linkinv(self.family.link[parent], pymc_backend.INVLINKS)

        # Add parent parameter and observed data. We don't need to pass dims.
        kwargs[parent] = linkinv(nu)
        kwargs["observed"] = data

        # Build the response distribution
        dist = self.build_response_distribution(kwargs)

        return dist

    def build_response_distribution(self, kwargs):
        # Get likelihood distribution
        distribution = get_distribution_from_likelihood(self.family.likelihood)

        # Families can implement specific transformations of parameters that are passed to the
        # likelihood function
        if hasattr(self.family, "transform_backend_kwargs"):
            kwargs = self.family.transform_backend_kwargs(kwargs)

        return distribution(self.name, **kwargs)

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name


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

        # TODO: Is this enough? What if we have more than a single coord?
        #       We already have some coords that correspond to the 'weights' or basis functions.
        #       There could be another dimension which would be given by a categorical variable.
        #       I'm not sure if it should be handled via the 'by' argument, or as an interaction.
        if self.coords and self.term.alias:
            self.coords[self.term.alias + "_dim"] = self.coords.pop(self.term.name + "_dim")

    def build(self, pymc_backend, bmb_model):  # TODO: Handle when there's a 'by' argument
        label = self.name

        # Get the covariance function
        cov_func = self.get_cov_func()

        # This handles univariate and multivariate cases.
        pymc_backend.model.add_coords({f"{label}_weights_dim": np.arange(np.prod(self.term.m))})

        # Get dimension name for the response
        response_name = get_aliased_name(bmb_model.response_component.response_term)

        # Prepare dims
        coeff_dims = (f"{label}_weights_dim",)
        contribution_dims = (f"{response_name}_obs",)

        # Build HSGP and store it in the term.
        if self.term.by is not None:
            # TODO: How to take different covariance functions, while also preserving
            #       the ability to use the same when needed?
            flatten_coeffs = True
            levels = np.unique(self.term.by)
            coeff_dims = coeff_dims + (f"{label}_by",)
            pymc_backend.model.add_coords({f"{label}_by": levels})
            phi_list = []
            sqrt_psd_list = []
            self.term.hsgp = {}
            for i, level in enumerate(levels):
                hsgp = pm.gp.HSGP(
                    m=list(self.term.m),  # Doesn't change by group
                    L=list(self.term.L[i]),  # 1d array is not a Sequence
                    drop_first=self.term.drop_first,
                    cov_func=cov_func,
                )
                phi, sqrt_psd = hsgp.prior_linearized(self.term.data_centered)
                phi = phi.eval()
                phi[self.term.by != level] = 0
                sqrt_psd_list.append(sqrt_psd)
                phi_list.append(phi)

                # Store it for later usage
                self.term.hsgp[level] = hsgp

            sqrt_psd = pt.stack(sqrt_psd_list, axis=1)
            phi = np.hstack(phi_list)
        else:
            flatten_coeffs = False
            self.term.hsgp = pm.gp.HSGP(
                m=list(self.term.m),
                L=list(self.term.L[0]),
                drop_first=self.term.drop_first,
                cov_func=cov_func,
            )
            # Get prior components
            phi, sqrt_psd = self.term.hsgp.prior_linearized(self.term.data_centered)
            phi = phi.eval()

        # Build weights coefficient
        if self.term.centered:
            coeffs = pm.Normal(f"{label}_weights", sigma=sqrt_psd, dims=coeff_dims)
        else:
            coeffs_raw = pm.Normal(f"{label}_weights_raw", dims=coeff_dims)
            coeffs = pm.Deterministic(f"{label}_weights", coeffs_raw * sqrt_psd, dims=coeff_dims)

        # Build deterministic for the HSGP contribution
        if flatten_coeffs:
            coeffs = coeffs.T.flatten()  # Equivalent to .flatten("F")
        output = pm.Deterministic(label, phi @ coeffs, dims=contribution_dims)
        return output

    def get_cov_func(self):
        """Construct and return the covariance function

        This method uses the name of the covariance function to retrieve a callable that
        returns a GP kernel and the name of its parameters. Then it looks for values for the
        parameters in the dictionary of priors of the term (building PyMC distributions as needed)
        and finally it determines the value of 'input_dim' if that is required by the callable
        that produces the covariance function. If that is the case, 'input_dim' is set to the
        dimensionality of the GP component -- the number of columns in the data.

        Returns
        -------
        pm.gp.Covariance
            A covariance function that can be used with a GP in PyMC
        """
        # Get the callable that creates the function
        cov_dict = GP_KERNELS[self.term.cov[0]]  # FIXME
        create_cov_function = cov_dict["fn"]
        names = cov_dict["params"]
        params = {}

        # Build priors
        for name in names:
            prior = self.term.prior[name]
            if isinstance(prior, Prior):
                distribution = get_distribution_from_prior(prior)
                value = distribution(f"{self.name}_{name}", **prior.args)
            else:
                value = prior
            params[name] = value

        if "input_dim" in list(inspect.signature(create_cov_function).parameters):
            params["input_dim"] = self.term.shape[1]

        return create_cov_function(**params)

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name
