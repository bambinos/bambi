from .prior import Prior

from ..utils import multilinify, spacify

DISTRIBUTIONS = {
    "Normal": {"params": ("mu", "sigma"), "parent": "mu", "args": ("sigma",)},
    "Bernoulli": {"params": ("p",), "parent": "p", "args": None},
    "Beta": {"params": ("mu", "kappa"), "parent": "mu", "args": ("kappa",)},
    "Binomial": {"params": ("p",), "parent": "p", "args": None},
    "Poisson": {"params": ("mu",), "parent": "mu", "args": None},
    "StudentT": {"params": ("mu", "lam"), "args": ("lam", "nu")},
    "NegativeBinomial": {"params": ("mu", "alpha"), "parent": "mu", "args": ("alpha",)},
    "Gamma": {"params": ("mu", "alpha"), "parent": "mu", "args": ("alpha",)},
    "Wald": {"params": ("mu", "lam"), "parent": "mu", "args": ("lam",)},
}


class Likelihood:
    """Representation of a Likelihood function for a Bambi model.

    'parent' must not be in 'kwargs'. 'parent' is inferred from the 'name' if it is a known name

    Parameters
    ----------
    name: str
        Name of the likelihood function. Must be a valid PyMC3 distribution name.
    parent: str
        Optional specification of the name of the mean parameter in the likelihood.
        This is the parameter whose transformation is modeled by the linear predictor.
    kwargs
        Keyword arguments that indicate prior distributions for auxiliary parameters in the
        likelihood.
    """

    DISTRIBUTIONS = DISTRIBUTIONS

    def __init__(self, name, parent=None, pps=None, **kwargs):
        if name in self.DISTRIBUTIONS:
            self.name = name
            self.parent = self._get_parent(parent)
            self.priors = self._check_priors(kwargs)
            self.pps = pps
        else:
            # On your own risk
            self.name = name
            # Check priors passed are in fact of class Prior
            check_all_are_priors(kwargs)
            self.priors = kwargs
            self.parent = parent
            self.pps = pps

    def _get_parent(self, parent):
        if parent is None:
            parent = self.DISTRIBUTIONS[self.name]["parent"]
        elif parent not in self.DISTRIBUTIONS[self.name]["params"]:
            raise ValueError(
                f"'{parent}' is not a valid parameter for the likelihood '{self.name}'"
            )
        return parent

    def _check_priors(self, priors):
        args = self.DISTRIBUTIONS[self.name]["args"]

        # The function requires priors but none were passed
        if priors == {} and args is not None:
            raise ValueError(f"'{self.name}' requires priors for the parameters {args}.")

        # The function does not require priors, but at least one was passed
        if priors != {} and args is None:
            raise ValueError(f"'{self.name}' does not require any additional prior.")

        # The function requires priors, priors were passed, but they differ from the required
        if priors and args:
            difference = set(args) - set(priors)
            if len(difference) > 0:
                raise ValueError(f"'{self.name}' misses priors for the parameters {difference}")

            # And check priors passed are in fact of class Prior
            check_all_are_priors(priors)

        return priors

    def __str__(self):
        args = [f"name: {self.name}", f"parent: {self.parent}", f"priors: {self.priors}"]
        return f"{self.__class__.__name__}({spacify(multilinify(args))}\n)"

    def __repr__(self):
        return self.__str__()


def check_all_are_priors(priors):
    if any(not isinstance(prior, (Prior, int, float)) for prior in priors.values()):
        raise ValueError("Prior distributions must be a 'Prior' instance or a numeric value")
