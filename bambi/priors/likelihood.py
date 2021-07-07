from .prior import Prior

DISTRIBUTIONS = {
    "Normal": {"params": ("mu", "sigma"), "parent": "mu", "args": ("sigma",)},
    "Bernoulli": {"params": ("p",), "parent": "p", "args": None},
    "Poisson": {"params": ("mu",), "parent": "mu", "args": None},
    "StudentT": {"params": ("mu", "lam"), "args": ("lam",)},
    "NegativeBinomial": {"params": ("mu", "alpha"), "parent": "mu", "args": ("alpha",)},
    "Gamma": {"params": ("mu", "alpha"), "parent": "mu", "args": ("alpha",)},
    "Wald": {"params": ("mu", "lam"), "parent": "mu", "args": ("lam",)},
}


class Likelihood:
    """Representation of a Likelihood function for a Bambi model.

    'parent' must not be in 'priors'. 'parent' is inferred from the 'name' if it is a known name

    Parameters
    ----------
    name: str
        Name of the likelihood function. Must be a valid PyMC3 distribution name.
    priors: dict
        Optional dictionary containing prior distributions for auxiliary parameters in the
        likelihood.
    parent: str
        Optional specification of the name of the mean parameter in the likelihood.
        This is the parameter whose transformation is modeled by the linear predictor.
    """

    DISTRIBUTIONS = DISTRIBUTIONS

    def __init__(self, name, priors=None, parent=None):
        if name in self.DISTRIBUTIONS:
            self.name = name
            self.parent = self._get_parent(parent)
            self.priors = self._check_priors(priors)
        else:
            # On your own risk
            self.name = name
            self.priors = priors
            self.parent = parent

    def _get_parent(self, parent):
        if parent is None:
            parent = self.DISTRIBUTIONS[self.name]["parent"]
        elif parent not in self.DISTRIBUTIONS[self.name]["params"]:
            raise ValueError(
                f"'{parent}'' is not a valid parameter for the likelihood '{self.name}'"
            )
        return parent

    def _check_priors(self, priors):
        args = self.DISTRIBUTIONS[self.name]["args"]

        # Make sure 'priors' is of the expected type
        if not (priors is None or isinstance(priors, dict)):
            raise ValueError("The 'priors' argument must be a dictionary or None.")

        # The function requires priors but none were passed
        if priors is None and args is not None:
            raise ValueError(f"'{self.name}' requires priors for the parameters {args}.")

        # The function does not require priors, but at least one was passed
        if priors is not None and args is None:
            raise ValueError(f"'{self.name}' does not require any additional prior.")

        # The function requires priors, priors were passed, but they differ from the required
        if priors and args:
            difference = set(args) - set(priors)
            if len(difference) > 0:
                raise ValueError(f"'{self.name}' misses priors for the parameters {difference}")

            # And check priors passed are in fact of class Prior
            if any(not isinstance(prior, Prior) for prior in priors.values()):
                raise ValueError("Prior distributions must be of class 'Prior'")

        return priors
