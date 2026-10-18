import bambi as bmb
import numpy as np

from scipy import special, stats

from pytensor.graph.traversal import ancestors


def assert_ip_dlogp(model: bmb.Model) -> bool:
    dlogp = model.backend.model.compile_dlogp()
    ip = model.backend.model.initial_point()
    assert not np.isinf(dlogp(ip)).any()


def graph_contains_op(output, op_type):
    return any(
        isinstance(node.owner.op, op_type) for node in ancestors([output]) if node.owner is not None
    )


def ordered_beta_reference_logp(value, mu, kappa, c0, c1):
    """Ordered beta log-density, computed independently with SciPy."""
    eta = special.logit(mu)
    p0 = special.expit(c0 - eta)
    p1 = special.expit(eta - c1)
    with np.errstate(divide="ignore"):
        interior = np.log1p(-(p0 + p1)) + stats.beta(mu * kappa, (1 - mu) * kappa).logpdf(value)
    return np.where(value == 0, np.log(p0), np.where(value == 1, np.log(p1), interior))


def zero_one_inflated_beta_reference_logp(value, mu, kappa, zoi, coi):
    """Zero-one-inflated beta log-density, computed independently with SciPy."""
    with np.errstate(divide="ignore"):
        interior = np.log1p(-zoi) + stats.beta(mu * kappa, (1 - mu) * kappa).logpdf(value)
        log_p0 = np.log(zoi) + np.log1p(-coi)
        log_p1 = np.log(zoi) + np.log(coi)
    return np.where(value == 0, log_p0, np.where(value == 1, log_p1, interior))
