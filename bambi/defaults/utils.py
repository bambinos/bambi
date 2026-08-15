from bambi.defaults.hsgp import HSGP_COV_PARAMS_DEFAULT_PRIORS

from bambi.families.likelihood import Likelihood
from bambi.priors.prior import Prior


def _build_prior_from_spec(spec):
    """Build a Prior instance from a dictionary specification."""
    if not isinstance(spec, dict) or "name" not in spec:
        raise ValueError(
            "Prior specification must be a dictionary containing at least a 'name' key."
        )

    kwargs = {}
    for key, value in spec.items():
        if key == "name":
            continue
        if isinstance(value, dict) and "name" in value:
            kwargs[key] = _build_prior_from_spec(value)
        else:
            kwargs[key] = value

    return Prior(spec["name"], **kwargs)


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
    dict of str to Prior
        The priors for the parameters of the covariance function
    """
    if cov_name not in HSGP_COV_PARAMS_DEFAULT_PRIORS:
        available = sorted(HSGP_COV_PARAMS_DEFAULT_PRIORS)
        raise ValueError(
            f"'{cov_name}' is not a valid HSGP covariance function. "
            f"Available options are: {available}."
        )

    config = HSGP_COV_PARAMS_DEFAULT_PRIORS[cov_name]
    priors = {}
    for param, prior_spec in config.items():
        priors[param] = _build_prior_from_spec(prior_spec)
    return priors


def get_default_prior(term_type, **kwargs):
    """Generate a Prior based on the default settings

    The following summarises default priors for each type of term:

    - intercept: Normal prior.
    - common: Normal prior.
    - intercept_flat: Uniform prior.
    - common_flat: Uniform prior.
    - group_specific: Normal prior where its sigma has a HalfNormal hyperprior.
    - group_specific_flat: Normal prior where its sigma has a HalfFlat hyperprior.
    - hsgp: The priors depend on the value passed to `kwargs["cov_func"]`.
        See `HSGP_COV_PARAMS_DEFAULT_PRIORS`.

    Parameters
    ----------
    term_type : str
        The type of the term for which the default prior is wanted.

    Raises
    ------
    ValueError
        If `term_type` is not one of the values listed above.

    Returns
    -------
    prior : Prior
        The instance of Prior according to the `term_type`.
    """
    if term_type in ["intercept", "common"]:
        prior = Prior("Normal", mu=0, sigma=1)
    elif term_type in ["intercept_flat", "common_flat"]:
        prior = Prior("Flat")
    elif term_type == "group_specific":
        prior = Prior("Normal", mu=0, sigma=Prior("HalfNormal", sigma=1))
    elif term_type == "group_specific_flat":
        prior = Prior("Normal", mu=0, sigma=Prior("HalfFlat"))
    elif term_type == "hsgp":
        prior = generate_prior_hsgp(kwargs["cov_func"])
    else:
        raise ValueError("Unrecognized term type.")
    return prior


def generate_family(name, likelihood, link, family, default_priors=None):
    """Generate a Bambi family.

    Parameters
    ----------
    name : str
        The name of the family.
    likelihood : bambi.Likelihood
        A representation of the likelihood function that corresponds to the family being created.
    link : bambi.Link
        A representation of the link function that corresponds to the family being created.
    family : subclass of bambi.Family
        A subclass of bambi.Family that generates the instance of the desired family.
    default_priors : dict or None, optional
        Default priors for non-parent parameters.

    Returns
    -------
    bambi.Family
        The family instance.
    """
    likelihood = Likelihood(**likelihood)
    family = family(name, likelihood, link)
    if default_priors:
        family.set_default_priors({k: _build_prior_from_spec(v) for k, v in default_priors.items()})
    return family
