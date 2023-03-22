from bambi.defaults.distributions import SETTINGS_DISTRIBUTIONS
from bambi.defaults.hsgp import HSGP_COV_PARAMS_DEFAULT_PRIORS

from bambi.families import Likelihood
from bambi.priors import Prior


def generate_prior(dist, **kwargs):
    """Generate a Prior distribution.

    The parameter ``kwargs`` is used to pass hyperpriors that are assigned to the parameters of
    the prior to be built.

    Parameters
    ----------
    dist: str, int, float
        If a string, it is the name of the prior distribution with default values taken from
        ``SETTINGS_DISTRIBUTIONS``. If a number, it is a factor used to scale the standard deviation
        of the priors generated automatically by Bambi.

    Raises
    ------
    ValueError
        If ``dist`` is not a string or a number.

    Returns
    -------
    Prior
        The Prior instance.
    """
    if isinstance(dist, str):
        prior = Prior(dist, **SETTINGS_DISTRIBUTIONS[dist])
        if kwargs:
            prior.update(**{k: generate_prior(v) for k, v in kwargs.items()})
    elif isinstance(dist, (int, float)):
        prior = dist
    else:
        raise ValueError("'dist' must be the name of a distribution or a numeric value.")
    return prior


def generate_prior_hsgp(cov_name: str):
    """Generate a prior configuration for an HSGP term

    The 'prior' for the HSGP term refers to a dictionary of priors. This dictionary contains
    Prior instances for the parameters of the covariance function.

    Parameters
    ----------
    cov_name : str
        The name of a covariance function

    Returns
    -------
    dict[str, Prior]
        The priors for the parameters of the covariance function
    """
    config = HSGP_COV_PARAMS_DEFAULT_PRIORS[cov_name]
    priors = {}
    for param, dist in config.items():
        priors[param] = Prior(dist, **SETTINGS_DISTRIBUTIONS[dist])
    return priors


def get_default_prior(term_type, **kwargs):
    """Generate a Prior based on the default settings

    The following summarises default priors for each type of term:

    * intercept: Normal prior.
    * common: Normal prior.
    * intercept_flat: Uniform prior.
    * common_flat: Uniform prior.
    * group_specific: Normal prior where its sigma has a HalfNormal hyperprior.
    * group_specific_flat: Normal prior where its sigma has a HalfFlat hyperprior.
    * hsgp: The priors depend on the value passed to `kwargs["cov_func"]`.
        See `HSGP_COV_PARAMS_DEFAULT_PRIORS`.

    Parameters
    ----------
    term_type : str
        The type of the term for which the default prior is wanted.

    Raises
    ------
    ValueError
        If ``term_type`` is not within the values listed above.

    Returns
    -------
    prior: Prior
        The instance of Prior according to the ``term_type``.
    """
    if term_type in ["intercept", "common"]:
        prior = generate_prior("Normal")
    elif term_type in ["intercept_flat", "common_flat"]:
        prior = generate_prior("Flat")
    elif term_type == "group_specific":
        prior = generate_prior("Normal", sigma="HalfNormal")
    elif term_type == "group_specific_flat":
        prior = generate_prior("Normal", sigma="HalfFlat")
    elif term_type == "hsgp":
        prior = generate_prior_hsgp(kwargs["cov_func"])
    else:
        raise ValueError("Unrecognized term type.")
    return prior


def generate_likelihood(name, params, parent):
    """Generate a Likelihood instance.

    Parameters
    ----------
    name: str
        The name of the likelihood function.
    args: dict
        Indicates the auxiliary parameters and the values for their default priors. The keys are the
        names of the parameters and the values are passed to ``generate_prior()`` to obtain the
        actual instance of ``bambi.Prior``.
    parent: str
        The name of the parent parameter. In other words, the name of the mean parameter in the
        likelihood function.

    Returns
    -------
    bambi.Likelihood
        The likelihood instance.
    """
    return Likelihood(name, params, parent)


def generate_family(name, likelihood, link, family, default_priors=None):
    """Generate a Bambi family.

    Parameters
    ----------
    name : str
        The name of the family.
    likelihood: bambi.Likelihood
        A representation of the likelihood function that corresponds to the family being created.
    link : bambi.Link
        A representation of the link function that corresponds to the family being created.
    family : subclass of bambi.Family
        A subclass of bambi.Family that generates the instance of the desired family.
    default_priors : dict
        Default priors for non-parent parameters.

    Returns
    -------
    bambi.Family
        The family instance.
    """
    likelihood = generate_likelihood(**likelihood)
    family = family(name, likelihood, link)
    if default_priors:
        family.set_default_priors({k: generate_prior(v) for k, v in default_priors.items()})
    return family
