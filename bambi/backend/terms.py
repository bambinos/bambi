import numpy as np
import pymc as pm

import pytensor.tensor as pt

from bambi.backend.utils import has_hyperprior, get_distribution
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
        dist = self.term.prior.name
        args = self.term.prior.args
        distribution = get_distribution(dist)

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
        dist = self.term.prior.name
        kwargs = self.term.prior.args
        predictor = np.squeeze(self.term.predictor)

        # Dims of the response variable (e.g. categorical)
        response_dims = []
        if isinstance(spec.family, (MultivariateFamily, Categorical)):
            response_dims = list(spec.response_component.response_term.coords)

        dims = list(self.coords) + response_dims
        # Squeeze ensures we don't have a shape of (n, 1) when we mean (n, )
        # This happens with categorical predictors with two levels and intercept.
        coef = self.build_distribution(dist, label, dims=dims, **kwargs).squeeze()
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

    def build_distribution(self, dist, label, **kwargs):
        """Build and return a PyMC Distribution."""
        dist = get_distribution(dist)

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
        return dist(label, **kwargs)

    def expand_prior_args(self, key, value, label, **kwargs):
        # kwargs are used to pass 'dims' for group specific terms.
        if isinstance(value, Prior):
            # If there's an alias for the hyperprior, use it.
            key = self.term.hyperprior_alias.get(key, key)
            return self.build_distribution(value.name, f"{label}_{key}", **value.args, **kwargs)
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
        dist = get_distribution(self.term.prior.name)
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
        if self.family.likelihood.dist:
            dist = self.family.likelihood.dist
        else:
            dist = get_distribution(self.family.likelihood.name)

        # Families can implement specific transformations of parameters that are passed to the
        # likelihood function
        if hasattr(self.family, "transform_backend_kwargs"):
            kwargs = self.family.transform_backend_kwargs(kwargs)

        return dist(self.name, **kwargs)

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name


def get_linkinv(link, invlinks):
    """Get the inverse of the link function as needed by PyMC

    Parameters
    ----------
    link : bmb.Link
        A link function object. It may contain the linkinv function that the backend uses.
    invlinks : dict
        Keys are names of link functions. Values are the built-in link functions.

    Returns
    -------
        callable
        The link function
    """
    # If the name is in the backend, get it from there
    if link.name in invlinks:
        invlink = invlinks[link.name]
    # If not, use whatever is in `linkinv_backend`
    else:
        invlink = link.linkinv_backend
    return invlink
