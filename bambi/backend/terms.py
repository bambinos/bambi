import numpy as np
import pymc3 as pm
import theano.tensor as tt

from bambi.backend.utils import has_hyperprior, get_distribution
from bambi.families.multivariate import Categorical
from bambi.families.univariate import Beta, Binomial, Gamma
from bambi.priors import Prior


class CommonTerm:
    """Represetation of a common effects term in PyMC3

    An object that builds the PyMC3 distribution for a common effects term. It also contains the
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
        label = self.term.name
        dist = self.term.prior.name
        args = self.term.prior.args
        distribution = get_distribution(dist)

        # Dims of the response variable (e.g. categorical family)
        response_dims = []
        if spec.response.categorical and not spec.response.binary:
            response_dims = list(spec.response.pymc_coords)

        dims = list(self.coords) + response_dims
        if dims:
            coef = distribution(label, dims=dims, **args)
        else:
            coef = distribution(label, shape=data.shape[1], **args)

        # Pre-pends one dimension if response is multi-categorical
        if response_dims:
            # If response is multivariate and predictor is univariate (not categorical/basis)
            if len(dims) == 1:
                coef = coef[np.newaxis, :]
        return coef, data

    def get_coords(self):
        coords = {}
        if self.term.categorical:
            name = self.term.name + "_coord"
            levels = self.term.term_dict["levels"]
            if self.term.kind == "interaction":
                coords[name] = levels
            elif self.term.term_dict["encoding"] == "full":
                coords[name] = levels
            else:
                coords[name] = levels[1:]
        else:
            # Not categorical but multi-column, like when we use splines
            if self.term.data.shape[1] > 1:
                name = self.term.name + "_coord"
                coords[name] = list(range(self.term.data.shape[1]))
        return coords


class GroupSpecificTerm:
    """Represetation of a group specific effects term in PyMC3

    Creates an object that builds the PyMC3 distribution for a group specific effect. It also
    contains the coordinates that we then add to the model.

    Parameters
    ----------
    term: bambi.terms.GroupSpecificTerm
        An object representing a group specific effects term.
    """

    def __init__(self, term, noncentered):
        self.term = term
        self.noncentered = noncentered
        self.coords = self.get_coords()

    def build(self, spec):
        label = self.term.name
        dist = self.term.prior.name
        kwargs = self.term.prior.args
        predictor = self.term.predictor.squeeze()

        # Dims of the response variable (e.g. categorical family)
        response_dims = []
        if spec.response.categorical and not spec.response.binary:
            response_dims = list(spec.response.pymc_coords)

        dims = list(self.coords) + response_dims
        # dims = list(self.coords)
        coef = self.build_distribution(dist, label, dims=dims, **kwargs)
        coef = coef[self.term.group_index]

        return coef, predictor

    def get_coords(self):
        coords = {}
        # The group is always a coordinate we add to the model.
        expr, factor = self.term.name.split("|")
        coords[factor + "_coord_group_factor"] = self.term.groups

        if self.term.categorical:
            name = expr + "_coord_group_expr"
            levels = self.term.term["levels"]
            if self.term.kind == "interaction":
                coords[name] = levels
            elif self.term.term["encoding"] == "full":
                coords[name] = levels
            else:
                coords[name] = levels[1:]
        return coords

    def build_distribution(self, dist, label, **kwargs):
        """Build and return a PyMC3 Distribution."""
        dist = get_distribution(dist)

        if "dims" in kwargs:
            group_dim = [dim for dim in kwargs["dims"] if dim.endswith("_group_expr")]
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
            return self.build_distribution(value.name, f"{label}_{key}", **value.args, **kwargs)
        return value


class InterceptTerm:
    """Representation of an intercept term in a PyMC3 model.

    Parameters
    ----------
    term: bambi.terms.Term
        An object representing the intercept. This has ``.kind == "intercept"``
    """

    def __init__(self, term):
        self.term = term

    def build(self, spec):
        dist = get_distribution(self.term.prior.name)
        # Pre-pends one dimension if response is multi-categorical
        if spec.response.categorical and not spec.response.binary:
            dims = list(spec.response.pymc_coords)
            dist = dist(self.term.name, dims=dims, **self.term.prior.args)[np.newaxis, :]
        else:
            dist = dist(self.term.name, shape=1, **self.term.prior.args)
        return dist


class ResponseTerm:
    """Representation of a response term in a PyMC3 model.

    Parameters
    ----------
    term: bambi.terms.ResponseTerm
        The response term as represented in Bambi.
    family: bambi.famlies.Family
        The model family.
    """

    def __init__(self, term, family):
        self.term = term
        self.family = family

    def build(self, nu, invlinks):
        """Create and return the response distribution for the PyMC3 model.

        nu: theano.tensor.var.TensorVariable
            The linear predictor in the PyMC3 model.
        invlinks: dict
            A dictionary where names are names of inverse link functions and values are functions
            that can operate with Theano tensors.
        """
        data = self.term.data.squeeze()
        name = self.term.name

        if self.family.link.name in invlinks:
            linkinv = invlinks[self.family.link.name]
        else:
            linkinv = self.family.link.linkinv_backend

        # Add column of zeros, for the reference level, which is the first one.
        if isinstance(self.family, Categorical):
            nu = tt.concatenate([np.zeros((data.shape[0], 1)), nu], axis=1)

        # Build priors for the auxiliary parameters in the likelihood (e.g. sigma in Gaussian)
        likelihood = self.family.likelihood
        kwargs = {likelihood.parent: linkinv(nu), "observed": data}
        if likelihood.priors:
            for key, value in likelihood.priors.items():
                if isinstance(value, Prior):
                    dist = get_distribution(value.name)
                    kwargs[key] = dist(f"{name}_{key}", **value.args)
                else:
                    kwargs[key] = value

        # Get likelihood distribution
        dist = get_distribution(likelihood.name)

        if isinstance(self.family, Beta):
            # Beta distribution is specified using alpha and beta, but we have mu and kappa.
            # alpha = mu * kappa
            # beta = (1 - mu) * kappa
            alpha = kwargs["mu"] * kwargs["kappa"]
            beta = (1 - kwargs["mu"]) * kwargs["kappa"]
            return dist(name, alpha=alpha, beta=beta, observed=kwargs["observed"])

        if isinstance(self.family, Binomial):
            successes = data[:, 0].squeeze()
            trials = data[:, 1].squeeze()
            return dist(name, p=kwargs["p"], observed=successes, n=trials)

        if isinstance(self.family, Gamma):
            # Gamma distribution is specified using mu and sigma, but we request prior for alpha.
            # We build sigma from mu and alpha.
            sigma = kwargs["mu"] / (kwargs["alpha"] ** 0.5)
            return dist(name, mu=kwargs["mu"], sigma=sigma, observed=kwargs["observed"])

        return dist(name, **kwargs)
