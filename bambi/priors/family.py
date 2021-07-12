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

    LINKS = ("identity", "log", "logit", "probit", "cloglog", "inverse", "inverse_squared")

    def __init__(self, name, likelihood, link):
        self.smlink = None
        self.name = name
        self.likelihood = likelihood
        self.link = link
        self.smfamily = STATSMODELS_FAMILIES.get(name, None)

    def _set_link(self, link):
        """Set new link function.

        It updates both ``self.link`` (a string or function passed to the backend) and
        ``self.smlink`` (the link instance for the statsmodel family).
        """
        if isinstance(link, str):
            if link in self.LINKS:
                self.link = link
                self.smlink = STATSMODELS_LINKS.get(link, None)
            else:
                raise ValueError(f"Link name '{link}' is not supported.")
        # Things like classes are still callable, so this is not ideal.
        # But it is hard to check whether something is a function OR something like
        # 'tt.nnet.sigmoid'. These return False for inspect.isfunction().
        elif callable(link):
            self.link = link
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
