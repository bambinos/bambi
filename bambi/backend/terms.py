import numpy as np
import pymc as pm

import pytensor.tensor as pt

from bambi.backend.utils import has_hyperprior, get_distribution
from bambi.families.multivariate import MultivariateFamily
from bambi.priors import Prior


class CommonTerm:
    """Representation of a common effects term in PyMC

    An object that builds the PyMC distribution for a common effects term. It also contains the
    coordinates that we then add to the model.

    Parameters
    ----------
    term: bambi.terms.Term
        An object representing a common effects term.
    """

    def __init__(self, term):
        self.term = term
        self.coords = self.term.coords.copy()
        # Make sure we use the alias, if there's one
        # NOTE: Could be handled in the term??
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
        if isinstance(spec.family, MultivariateFamily):
            response_dims = list(spec.response.coords)
            response_dims_n = len(spec.response.coords[response_dims[0]])
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
    term: bambi.terms.GroupSpecificTerm
        An object representing a group specific effects term.
    noncentered: bool
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
        if isinstance(spec.family, MultivariateFamily):
            response_dims = list(spec.response.coords)

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
    term: bambi.terms.Term
        An object representing the intercept. This has ``.kind == "intercept"``
    """

    def __init__(self, term):
        self.term = term

    def build(self, spec):
        dist = get_distribution(self.term.prior.name)
        label = self.name
        # Prepends one dimension if response is multivariate
        if isinstance(spec.family, MultivariateFamily):
            dims = list(spec.response.coords)
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

    def build(self, nu, invlinks):
        """Create and return the response distribution for the PyMC model.

        nu : pytensor.tensor.var.TensorVariable
            The linear predictor in the PyMC model.
        invlinks : dict
            A dictionary where names are names of inverse link functions and values are functions
            that can operate with PyTensor tensors.
        """
        data = np.squeeze(self.term.data)

        # Take the inverse link function that maps from linear predictor to the mean of likelihood
        if self.family.link.name in invlinks:
            linkinv = invlinks[self.family.link.name]
        else:
            linkinv = self.family.link.linkinv_backend

        if hasattr(self.family, "transform_backend_nu"):
            nu = self.family.transform_backend_nu(nu, data)

        # Add mean parameter and observed data
        kwargs = {self.family.likelihood.parent: linkinv(nu), "observed": data}

        # Add auxiliary parameters
        kwargs = self.build_auxiliary_parameters(kwargs)

        # Build the response distribution
        dist = self.build_response_distribution(kwargs)

        return dist

    def build_auxiliary_parameters(self, kwargs):
        # Build priors for the auxiliary parameters in the likelihood (e.g. sigma in Gaussian)
        if self.family.likelihood.priors:
            for key, value in self.family.likelihood.priors.items():

                # Use the alias if there's one
                if key in self.family.aliases:
                    label = self.family.aliases[key]
                else:
                    label = f"{self.name}_{key}"

                if isinstance(value, Prior):
                    dist = get_distribution(value.name)
                    kwargs[key] = dist(label, **value.args)
                else:
                    kwargs[key] = value
        return kwargs

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
