import numpy as np

from scipy import special
from statsmodels.genmod import families as sm_families

STATSMODELS_FAMILIES = {
    "bernoulli": sm_families.Binomial,
    "gamma": sm_families.Gamma,
    "gaussian": sm_families.Gaussian,
    "wald": sm_families.InverseGaussian,
    "negativebinomial": sm_families.NegativeBinomial,
    "poisson": sm_families.Poisson,
}

STATSMODELS_LINKS = {
    "identity": sm_families.links.identity(),
    "logit": sm_families.links.logit(),
    "probit": sm_families.links.probit(),
    "cloglog": sm_families.links.cloglog(),
    "inverse": sm_families.links.inverse_power(),
    "inverse_squared": sm_families.links.inverse_squared(),
    "log": sm_families.links.log(),
}


def cloglog(x):
    """Cloglog function that ensures result is in (0, 1)"""
    result = 1 - np.exp(-np.exp(x))
    result = force_within_unit_interval(result)
    return result


def probit(x):
    """Probit function that ensures result is in (0, 1)"""
    result = 0.5 + 0.5 * special.erf(x / 2 ** 0.5)
    result = force_within_unit_interval(result)
    return result


def expit(x):
    """Expit function that ensures result is in (0, 1)"""
    result = special.expit(x)
    result = force_within_unit_interval(result)
    return result


def force_within_unit_interval(x):
    eps = np.finfo(float).eps
    x[x == 0] = eps
    x[x == 1] = 1 - eps
    return x


class Family:
    """A specification of model family.

    Parameters
    ----------
    name : str
        Family name.
    likelihood: Likelihood
        A ``Likelihood`` instace specifying the model likelihood function.
    link : str or function
        The name of the link function, or the function itself, transforming the linear model
        prediction to the mean parameter of the likelihood. If a function, it must be able to
        operate over theano tensors rather than numpy arrays.
    """

    LINKS = {
        "cloglog": cloglog,
        "identity": lambda x: x,
        "inverse_squared": lambda x: 1 / np.sqrt(x),
        "inverse": lambda x: 1 / x,
        "log": np.exp,
        "logit": expit,
        "probit": probit,
    }

    def __init__(self, name, likelihood, link):
        self.smlink = None
        self.link = None
        self._link = None
        self.name = name
        self.likelihood = likelihood
        self.smfamily = STATSMODELS_FAMILIES.get(name, None)
        self._set_link(link)

    def _set_link(self, link):
        """Set new link function.

        It updates both ``self.link`` (a string or function passed to the backend) and
        ``self.smlink`` (the link instance for the statsmodel family).
        """
        if isinstance(link, str):
            if link in self.LINKS:
                self.link = link
                self._link = self.LINKS[link]
                self.smlink = STATSMODELS_LINKS.get(link, None)
            else:
                raise ValueError(f"Link name '{link}' is not supported.")
        # Things like classes are still callable, so this is not ideal.
        # But it is hard to check whether something is a function OR something like
        # 'tt.nnet.sigmoid'. These return False for inspect.isfunction().
        elif callable(link):
            self.link = link
            self._link = link
        else:
            raise ValueError("'link' must be a string or a function.")

    def __str__(self):
        if self.link is None:
            msg = "No family set"
        else:
            msg_list = [f"Response distribution: {self.likelihood.name}", f"Link: {self.link}"]
            if self.likelihood.priors:
                priors_msg = "\n  ".join([f"{k} ~ {v}" for k, v in self.likelihood.priors.items()])
                msg_list += [f"Priors:\n  {priors_msg}"]
            msg = "\n".join(msg_list)
        return msg

    def __repr__(self):
        return self.__str__()
