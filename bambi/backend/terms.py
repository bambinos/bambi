import pymc3 as pm

from bambi.backend.utils import has_hyperprior, get_distribution
from bambi.priors import Prior


class CommonTerm:
    """
    term: bambi.terms.Term
    """

    def __init__(self, term):
        self.term = term
        self.coords = self.get_coords()

    def build(self):
        data = self.term.data
        label = self.term.name
        dist = self.term.prior.name
        args = self.term.prior.args
        distribution = get_distribution(dist)
        if self.coords:
            dims = list(self.coords.keys())
            coef = distribution(label, dims=dims, **args)
        else:
            coef = distribution(label, shape=data.shape[1], **args)
        return coef, data

    def get_coords(self):
        coords = {}
        if self.term.categorical:
            name = self.term.name + "_coord"
            levels = self.term.term_dict["levels"]
            if self.term.type == "interaction":
                coords[name] = levels
            elif self.term.term_dict["encoding"] == "full":
                coords[name] = levels
            else:
                coords[name] = levels[1:]
        return coords


class GroupSpecificTerm:
    """
    term: bambi.terms.GroupSpecificTerm
    """

    def __init__(self, term, noncentered):
        self.term = term
        self.noncentered = noncentered
        self.coords = self.get_coords()

    def build(self):
        label = self.term.name
        dist = self.term.prior.name
        kwargs = self.term.prior.args
        predictor = self.term.predictor.squeeze()
        dims = list(self.coords.keys())
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
            if self.term.type == "interaction":
                coords[name] = levels
            elif self.term.term["encoding"] == "full":
                coords[name] = levels
            else:
                coords[name] = levels[1:]
        return coords

    def build_distribution(self, dist, label, **kwargs):
        """Build and return a PyMC3 Distribution for a group specific term."""
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
    def __init__(self, term):
        self.term = term

    def build(self):
        dist = get_distribution(self.term.prior.name)
        return dist(self.term.name, shape=1, **self.term.prior.args)


class ResponseTerm:
    def __init__(self, term, family):
        self.term = term
        self.family = family

    def build(self, nu, invlinks):
        # nu is the linear predictor
        # invlinks is a dictionary with inverse link functions
        data = self.term.data.squeeze()
        name = self.term.name

        if self.family.link.name in invlinks:
            linkinv = invlinks[self.family.link.name]
        else:
            linkinv = self.family.link.linkinv_backend

        likelihood = self.family.likelihood
        dist = get_distribution(likelihood.name)
        kwargs = {likelihood.parent: linkinv(nu), "observed": data}
        if likelihood.priors:
            for key, value in likelihood.priors.items():
                if isinstance(value, Prior):
                    _dist = get_distribution(value.name)
                    kwargs[key] = _dist(f"{name}_{key}", **value.args)
                else:
                    kwargs[key] = value

        if self.family.name == "beta":
            # Beta distribution is specified using alpha and beta, but we have mu and kappa.
            # alpha = mu * kappa
            # beta = (1 - mu) * kappa
            alpha = kwargs["mu"] * kwargs["kappa"]
            beta = (1 - kwargs["mu"]) * kwargs["kappa"]
            return dist(name, alpha=alpha, beta=beta, observed=kwargs["observed"])

        if self.family.name == "binomial":
            successes = data[:, 0].squeeze()
            trials = data[:, 1].squeeze()
            return dist(name, p=kwargs["p"], observed=successes, n=trials)

        if self.family.name == "gamma":
            # Gamma distribution is specified using mu and sigma, but we request prior for alpha.
            # We build sigma from mu and alpha.
            sigma = kwargs["mu"] / (kwargs["alpha"] ** 0.5)
            return dist(name, mu=kwargs["mu"], sigma=sigma, observed=kwargs["observed"])

        return dist(name, **kwargs)
