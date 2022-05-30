from collections import namedtuple

from bambi.priors import Prior
from bambi.utils import multilinify, spacify


DistSettings = namedtuple("DistSettings", ["params", "parent", "args"])

DISTRIBUTIONS = {
    "Bernoulli": DistSettings(params=("p",), parent="p", args=None),
    "Beta": DistSettings(params=("mu", "kappa"), parent="mu", args=("kappa",)),
    "Binomial": DistSettings(params=("p",), parent="p", args=None),
    "Categorical": DistSettings(params=("p",), parent="p", args=None),
    "Gamma": DistSettings(params=("mu", "alpha"), parent="mu", args=("alpha",)),
    "Multinomial": DistSettings(params=("p",), parent="p", args=None),
    "Normal": DistSettings(params=("mu", "sigma"), parent="mu", args=("sigma",)),
    "NegativeBinomial": DistSettings(params=("mu", "alpha"), parent="mu", args=("alpha",)),
    "Poisson": DistSettings(params=("mu",), parent="mu", args=None),
    "StudentT": DistSettings(params=("mu", "sigma"), parent="mu", args=("sigma", "nu")),
    "VonMises": DistSettings(params=("mu", "kappa"), parent="mu", args=("kappa",)),
    "Wald": DistSettings(params=("mu", "lam"), parent="mu", args=("lam",)),
}


class Likelihood:
    """Representation of a Likelihood function for a Bambi model.

    Notes:
    * ``parent`` must not be in ``kwargs``.
    * ``parent`` is inferred from the ``name`` if it is a known name

    Parameters
    ----------
    name: str
        Name of the likelihood function. Must be a valid PyMC distribution name.
    parent: str
        Optional specification of the name of the mean parameter in the likelihood.
        This is the parameter whose transformation is modeled by the linear predictor.
    kwargs:
        Keyword arguments that indicate prior distributions for auxiliary parameters in the
        likelihood.
    """

    DISTRIBUTIONS = DISTRIBUTIONS

    def __init__(self, name, parent=None, **kwargs):
        if name in self.DISTRIBUTIONS:
            self.name = name
            self.parent = parent
            self.priors = self._check_priors(kwargs)
        else:
            # On your own risk
            self.name = name
            # Check priors passed are in fact of class Prior
            check_all_are_priors(kwargs)
            self.priors = kwargs
            self.parent = parent

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, x):
        # Checks are made when using a known distribution
        if self.name in self.DISTRIBUTIONS:
            if x is None:
                x = self.DISTRIBUTIONS[self.name].parent
            elif x not in self.DISTRIBUTIONS[self.name].params:
                raise ValueError(f"'{x}' is not a valid parameter for the likelihood '{self.name}'")
        # Otherwise, no check is done. At your own risk!
        self._parent = x

    def _check_priors(self, priors):
        args = self.DISTRIBUTIONS[self.name].args

        # The function requires priors but none were passed
        if priors == {} and args is not None:
            raise ValueError(f"'{self.name}' requires priors for the parameters {args}.")

        # The function does not require priors, but at least one was passed
        if priors != {} and args is None:
            raise ValueError(f"'{self.name}' does not require any additional priors.")

        # The function requires priors, priors were passed, but they differ from the required
        if priors and args:
            difference = set(args) - set(priors)
            if len(difference):
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
    """Checks if values in the supplied dictionary are all valid prior objects

    An object is a valid prior if
    * It is an instance of bambi.priors.Prior
    * It is a number

    Parameters
    ----------
    priors: dict
        A dictionary whose values are tested to be valid priors
    """
    if any(not isinstance(prior, (Prior, int, float)) for prior in priors.values()):
        raise ValueError("Prior distributions must be a 'Prior' instance or a numeric value")
