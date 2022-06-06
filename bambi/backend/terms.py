import numpy as np
import pymc as pm
import aesara.tensor as at

from bambi.backend.utils import has_hyperprior, get_distribution
from bambi.families.multivariate import Categorical, Multinomial
from bambi.families.univariate import Beta, Binomial, Gamma
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
        self.coords = self.get_coords()

    def build(self, spec):
        data = self.term.data
        label = self.name
        dist = self.term.prior.name
        args = self.term.prior.args
        distribution = get_distribution(dist)

        # Dims of the response variable
        response_dims = []
        if isinstance(spec.family, (Categorical, Multinomial)):
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
            coef = distribution(label, shape=data.shape[1], **args)

        # Prepends one dimension if response is multivariate categorical and predictor is 1d
        if response_dims and len(dims) == 1:
            coef = coef[np.newaxis, :]

        return coef, data

    def get_coords(self):
        coords = {}
        if self.term.categorical:
            name = self.name + "_dim"
            coords[name] = self.term.term.levels
        # Not categorical but multi-column, like when we use splines
        elif self.term.data.shape[1] > 1:
            name = self.name + "_dim"
            coords[name] = list(range(self.term.data.shape[1]))
        return coords

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
        predictor = self.term.predictor.squeeze()

        # Dims of the response variable (e.g. categorical)
        response_dims = []
        if isinstance(spec.family, (Categorical, Multinomial)):
            response_dims = list(spec.response.coords)

        dims = list(self.coords) + response_dims
        # Squeeze ensures we don't have a shape of (n, 1) when we mean (n, )
        # This happens with categorical predictors with two levels and intercept.
        coef = self.build_distribution(dist, label, dims=dims, **kwargs).squeeze()
        coef = coef[self.term.group_index]

        return coef, predictor

    def get_coords(self):
        coords = {}

        # Use the name of the alias if there's an alias
        if self.term.alias:
            expr, factor = self.term.alias, self.term.alias
        else:
            expr, factor = self.term.name.split("|")

        # The group is always a coordinate we add to the model.
        coords[factor + "__factor_dim"] = self.term.groups

        if self.term.categorical:
            levels = self.term.term.expr.levels
            coords[expr + "__expr_dim"] = levels
        elif self.term.predictor.ndim == 2 and self.term.predictor.shape[1] > 1:
            coords[expr + "__expr_dim"] = [str(i) for i in range(self.term.predictor.shape[1])]
        return coords

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
        # Pre-pends one dimension if response is multi-categorical
        if isinstance(spec.family, (Categorical, Multinomial)):
            dims = list(spec.response.coords)
            dist = dist(label, dims=dims, **self.term.prior.args)[np.newaxis, :]
        else:
            # NOTE: Intercept only models with shape=1 don't work anymore
            #       It seems that 'shape=1' is not needed anymore?
            # dist = dist(label, shape=1, **self.term.prior.args)
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

        nu : aesara.tensor.var.TensorVariable
            The linear predictor in the PyMC model.
        invlinks : dict
            A dictionary where names are names of inverse link functions and values are functions
            that can operate with Aesara tensors.
        """
        data = self.term.data.squeeze()

        # Take the inverse link function that maps from linear predictor to the mean of likelihood
        if self.family.link.name in invlinks:
            linkinv = invlinks[self.family.link.name]
        else:
            linkinv = self.family.link.linkinv_backend

        # Add column of zeros to the linear predictor for the reference level (the first one)
        if isinstance(self.family, (Categorical, Multinomial)):
            # Make sure intercept-only models work
            nu = np.ones((data.shape[0], 1)) * nu
            nu = at.concatenate([np.zeros((data.shape[0], 1)), nu], axis=1)

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
        dist = get_distribution(self.family.likelihood.name)

        # Handle some special cases
        if isinstance(self.family, Beta):
            # Beta distribution in PyMC uses alpha and beta, but we have mu and kappa.
            # alpha = mu * kappa
            # beta = (1 - mu) * kappa
            alpha = kwargs["mu"] * kwargs["kappa"]
            beta = (1 - kwargs["mu"]) * kwargs["kappa"]
            return dist(self.name, alpha=alpha, beta=beta, observed=kwargs["observed"])

        if isinstance(self.family, Binomial):
            successes = kwargs["observed"][:, 0].squeeze()
            trials = kwargs["observed"][:, 1].squeeze()
            return dist(self.name, p=kwargs["p"], observed=successes, n=trials)

        if isinstance(self.family, Gamma):
            # Gamma distribution is specified using mu and sigma, but we request prior for alpha.
            # We build sigma from mu and alpha.
            sigma = kwargs["mu"] / (kwargs["alpha"] ** 0.5)
            return dist(self.name, mu=kwargs["mu"], sigma=sigma, observed=kwargs["observed"])

        if isinstance(self.family, Multinomial):
            n = kwargs["observed"].sum(axis=1)
            return dist(self.name, p=kwargs["p"], observed=kwargs["observed"], n=n)

        return dist(self.name, **kwargs)

    @property
    def name(self):
        if self.term.alias:
            return self.term.alias
        return self.term.name
