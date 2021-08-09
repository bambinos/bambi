import numpy as np

from scipy import special

from ..utils import multilinify, spacify


def force_within_unit_interval(x):
    """Make sure data in unit interval is in (0, 1)"""
    eps = np.finfo(float).eps
    x[x == 0] = eps
    x[x == 1] = 1 - eps
    return x


def force_greater_than_zero(x):
    """Make sure data in positive reals is in (0, infty)"""
    eps = np.finfo(float).eps
    x[x == 0] = eps
    return x


def cloglog(mu):
    """Cloglog function that ensures the input is greater than 0."""
    mu = force_greater_than_zero(mu)
    return np.log(-np.log(1 - mu))


def invcloglog(eta):
    """Inverse of the cloglog function that ensures result is in (0, 1)"""
    result = 1 - np.exp(-np.exp(eta))
    return force_within_unit_interval(result)


def probit(mu):
    """Probit function that ensures the input is in (0, 1)"""
    mu = force_within_unit_interval(mu)
    return 2 ** 0.5 * special.erfinv(2 * mu - 1)  # pylint: disable=no-member


def invprobit(eta):
    """Inverse of the probit function that ensures result is in (0, 1)"""
    result = 0.5 + 0.5 * special.erf(eta / 2 ** 0.5)  # pylint: disable=no-member
    return force_within_unit_interval(result)


def expit(eta):
    """Expit function that ensures result is in (0, 1)"""
    result = special.expit(eta)  # pylint: disable=no-member
    result = force_within_unit_interval(result)
    return result


def logit(mu):
    """Logit function that ensures the input is in (0, 1)"""
    mu = force_within_unit_interval(mu)
    return special.logit(mu)  # pylint: disable=no-member


# linkfun: These are g. They map the response to the linear predictor scale.
# linkinv: These are g^(-1). They map the linear predictor to the response scale.
# fmt: off
LINKS = {
    "cloglog": {
        "link": cloglog,
        "linkinv": invcloglog
    },
    "identity": {
        "link": lambda mu: mu,
        "linkinv": lambda eta: eta
    },
    "inverse_squared": {
        "link": lambda mu: 1 / mu ** 2,
        "linkinv": lambda eta: 1 / np.sqrt(eta)
    },
    "inverse": {
        "link": cloglog,
        "linkinv": invcloglog
    },
    "log": {
        "link": np.log,
        "linkinv": np.exp
    },
    "logit": {
        "link": logit,
        "linkinv": expit
    },
    "probit": {
        "link": probit,
        "linkinv": invprobit
    }
}
# fmt: on


class Link:
    """Representation of link function.

    This object actually contains two main functions. One is the link function itself, the function
    that maps values in the response scale to the linear predictor, and the other is the inverse
    of the link function, that maps values of the linear predictor to the response scale.

    The great majority of users will never interact with this class unless they want to create
    a custom ``Family`` with a custom ``Link``. This is automatically handled for all the built-in
    families.

    Parameters
    ----------
    name: str
        The name of the link function. If it is a known name, it's not necessary to pass any
        other arguments because functions are already defined internally. If not known, all of
        ``link``, ``linkinv`` and ``linkinv_backend`` must be specified.
    link: function
        A function that maps the response to the linear predictor. Known as 'g()' in GLM jargon.
        Does not need to be specified when ``name`` is a known name.
    linkinv: function
        A function that maps the linear predictor to the response. Known as 'g()^-1' in GLM jargon.
        Does not need to be specified when ``name`` is a known name.
    linkinv_backend: function
        Same than ``linkinv`` but must be something that works with PyMC3 backend (i.e. it must
        work with Aesara tensors). Does not need to be specified when ``name`` is a known name.

    """

    def __init__(self, name, link=None, linkinv=None, linkinv_backend=None):
        self.name = name
        self.link = link
        self.linkinv = linkinv
        self.linkinv_backend = linkinv_backend

        if name in LINKS:
            self.link = LINKS[name]["link"]
            self.linkinv = LINKS[name]["linkinv"]
        else:
            if not link or not linkinv or not linkinv_backend:
                raise ValueError(
                    f"Link name '{name}' is not supported and at least one of 'link', "
                    "'linkinv' or 'linkinv_backend' are unespecified."
                )

    def __str__(self):
        args = [f"name: {self.name}", f"link: {self.link}", f"linkinv: {self.linkinv}"]
        return f"{self.__class__.__name__}({spacify(multilinify(args))}\n)"

    def __repr__(self):
        return self.__str__()
