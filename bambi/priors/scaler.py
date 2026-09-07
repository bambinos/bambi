import warnings

import numpy as np

from bambi.families.builtin import (
    Bernoulli,
    Binomial,
    Cumulative,
    Gaussian,
    StudentT,
    VonMises,
)
from bambi.parameters import MarginalParameter
from bambi.priors.prior import Prior

_NORMAL_STD = 2.5


def _safe_inverse_std(x, context, axis=None):
    x_std = np.std(x, axis=axis)
    zero_std = np.isclose(x_std, 0)

    if np.any(zero_std):
        if np.ndim(x_std) == 0:
            details = "std=0"
            x_std = 1.0
        else:
            zero_count = int(np.sum(zero_std))
            details = f"{zero_count} column(s) with std=0"
            x_std = np.where(zero_std, 1.0, x_std)

        warnings.warn(
            f"Detected {details} while scaling priors for {context}. "
            "Using std=1 for zero-variance data to avoid division by zero.",
            RuntimeWarning,
            stacklevel=2,
        )

    return 1 / x_std


def _is_bernoulli_and_sigmoid_like(model):
    is_bernoulli_like = isinstance(model.family, (Bernoulli, Binomial))
    if not is_bernoulli_like:
        return False

    is_sigmoid_like = getattr(model.family.link.get("p"), "name", None) in ("logit", "probit")
    return is_sigmoid_like


def _get_normal_intercept_stats(common_terms, common_priors, response_mean, response_std):
    """Compute the mean and scale for a Normal intercept prior."""
    mu = response_mean
    sigma = _NORMAL_STD * response_std

    # Only adjust sigma if there is at least one Normal prior for a common term.
    if common_priors:
        sigmas = np.hstack([prior["sigma"] for prior in common_priors.values()])
        x_mean = np.hstack([common_terms[term].data.mean(axis=0) for term in common_priors])
        sigma = (sigma**2 + np.dot(sigmas**2, x_mean**2)) ** 0.5
    return mu, sigma


def _get_normal_slope_sigma(x, response_std):
    """Compute the scale for a Normal slope prior."""
    inv_x_std = _safe_inverse_std(x, context="a slope term")
    return _NORMAL_STD * response_std * inv_x_std


def _get_common_term_stats(term, model, response_std):
    is_sigmoid_like = _is_bernoulli_and_sigmoid_like(model)
    is_interaction = term.kind == "interaction"
    is_categorical = term.categorical
    all_categoric_interaction = is_interaction and all(
        component.kind == "categoric" for component in term.term.components
    )

    if term.data.ndim == 1:
        mu = 0
        if not is_sigmoid_like:
            sigma = _get_normal_slope_sigma(term.data, response_std)
        elif all_categoric_interaction or is_categorical:
            sigma = 1
        else:
            sigma = _safe_inverse_std(term.data, axis=0, context=f"common term '{term.name}'")

        return mu, sigma

    # 2D term
    n_cols = term.data.shape[1]
    mu = np.zeros(n_cols)

    if not is_sigmoid_like:
        sigma = np.array([_get_normal_slope_sigma(col, response_std) for col in term.data.T])
        return mu, sigma

    if all_categoric_interaction or is_categorical:
        sigma = np.ones(n_cols)
    elif is_interaction:
        # Use std of the marginal numerical variable
        shared_sigma = _safe_inverse_std(
            np.sum(term.data, axis=1),
            context=f"interaction term '{term.name}' (marginal numerical variable)",
        )
        sigma = np.full(n_cols, shared_sigma)
    else:
        sigma = _safe_inverse_std(term.data, axis=0, context=f"common term '{term.name}'")

    return mu, sigma


def _scale_marginal_parameters(model, response_std):
    """Scale priors for response parameters when the family requires it."""
    if isinstance(model.family, (Gaussian, StudentT)):
        sigma = model.parameters["sigma"]
        if isinstance(sigma, MarginalParameter) and getattr(sigma.prior, "auto_scale", False):
            sigma.prior = Prior("HalfStudentT", nu=4, sigma=response_std)
    elif isinstance(model.family, VonMises):
        kappa = model.parameters["kappa"]
        if isinstance(kappa, MarginalParameter) and getattr(kappa.prior, "auto_scale", False):
            kappa.prior = Prior("HalfStudentT", nu=4, sigma=response_std)
    elif isinstance(model.family, Cumulative):
        threshold = model.parameters["threshold"]
        is_constant = isinstance(threshold, MarginalParameter)
        is_normal = threshold.prior.name == "Normal"
        auto_scale = getattr(threshold.prior, "auto_scale", False)
        if is_constant and is_normal and auto_scale:
            response_level_n = len(model.response_term.levels)
            mu = np.round(np.linspace(-2, 2, num=response_level_n - 1), 2)
            threshold.prior.update(mu=mu, sigma=1, transform="ordered")


def _scale_common_normal(model, term, response_std, common_priors):
    mu, sigma = _get_common_term_stats(term, model, response_std)
    common_priors[term.name] = {"mu": mu, "sigma": sigma}
    term.prior.update(mu=mu, sigma=sigma)


def _scale_intercept_normal(model, term, intercept_stats):
    if _is_bernoulli_and_sigmoid_like(model):
        mu, sigma = 0, 1.5
    else:
        mu, sigma = intercept_stats

    term.prior.update(mu=mu, sigma=sigma)


def _scale_group_specific_half_normal(term, intercept_stats, response_std):
    if term.kind == "intercept":
        _, sigma = intercept_stats
    else:
        # Recreate the corresponding common effect data.
        if len(term.predictor.shape) == 2:
            data_as_common = term.predictor
        else:
            data_as_common = term.predictor[:, None]
        sigma = np.zeros(data_as_common.shape[1])
        for i, value in enumerate(data_as_common.T):
            sigma[i] = _get_normal_slope_sigma(value, response_std)
    term.prior.args["sigma"].update(sigma=np.squeeze(np.atleast_1d(sigma)))


def scale_priors(model):
    main_parameter = model.parameters[model.family.likelihood.parent]

    has_intercept = main_parameter.intercept_term is not None
    common_terms = main_parameter.common_terms
    common_priors = {}

    if isinstance(model.family, (Gaussian, StudentT)):
        response_mean = np.mean(model.response_term.data)
        response_std = np.std(model.response_term.data)
    else:
        response_mean = 0
        response_std = 1

    # Scale marginal parameters
    _scale_marginal_parameters(model, response_std)

    # Scale common terms.
    for term in main_parameter.common_terms.values():
        auto_scale = getattr(term.prior, "auto_scale", False)
        is_normal = getattr(term.prior, "name", None) == "Normal"
        if auto_scale and is_normal:
            _scale_common_normal(model, term, response_std, common_priors)

    # Statistics common to both intercept and group-specific terms
    intercept_stats = _get_normal_intercept_stats(
        common_terms, common_priors, response_mean, response_std
    )

    # Scale intercept.
    if has_intercept:
        term = main_parameter.intercept_term
        auto_scale = getattr(term.prior, "auto_scale", False)
        is_normal = getattr(term.prior, "name", None) == "Normal"
        if auto_scale and is_normal:
            _scale_intercept_normal(model, term, intercept_stats)

    # Scale group-specific terms.
    for term in main_parameter.group_specific_terms.values():
        auto_scale = getattr(term.prior, "auto_scale", False)
        is_half_normal = getattr(term.prior.args.get("sigma"), "name", None) == "HalfNormal"
        if auto_scale and is_half_normal:
            _scale_group_specific_half_normal(term, intercept_stats, response_std)
