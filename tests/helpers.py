from functools import partial

import numpy as np
import pandas as pd
import pymc as pm
from pytensor.graph.traversal import ancestors

import bambi as bmb
from bambi.families.types import DimType, ParamSpec, ResponseType


def assert_ip_dlogp(model: bmb.Model) -> bool:
    dlogp = model.backend.model.compile_dlogp()
    ip = model.backend.model.initial_point()
    assert not np.isinf(dlogp(ip)).any()


def graph_contains_op(output, op_type):
    return any(
        isinstance(node.owner.op, op_type) for node in ancestors([output]) if node.owner is not None
    )


def normal_prior(sigma=2):
    return bmb.Prior("Normal", mu=0, sigma=sigma)


def linear_data(size=30):
    x = np.linspace(-1, 1, size)
    return pd.DataFrame({"x": x, "z": x**2, "y": 1 + 2 * x})


def exponential_formula(group_specific=False):
    a_formula = "a ~ 1 + (1 | group)" if group_specific else "a ~ 1 + z"
    return bmb.Formula(
        "y ~ a + b * exp(-k * x)",
        a_formula,
        nlpars=("a", "b", "k"),
    )


def exponential_priors(group_specific=False):
    a_priors = {"Intercept": normal_prior(), "z": normal_prior()}
    if group_specific:
        a_priors = {
            "Intercept": normal_prior(),
            "1|group": bmb.Prior("Normal", mu=0, sigma=bmb.Prior("HalfNormal", sigma=1)),
        }
    return {
        "a": a_priors,
        "b": normal_prior(),
        "k": normal_prior(),
    }


def weighted_nonlinear_data():
    return pd.DataFrame(
        {
            "x": [-1.0, 0.0, 0.5, 1.0],
            "z": [0.0, 1.0, 0.5, -0.5],
            "y": [-0.8, 0.4, 1.3, 2.5],
            "w": [0.5, 2.0, 1.5, 3.0],
        }
    )


def weighted_nonlinear_model(data=None, dropna=False):
    formula = bmb.Formula("weighted(y, w) ~ a + b * x", "a ~ 1 + z", nlpars=("a", "b"))
    priors = {
        "a": {"Intercept": normal_prior(), "z": normal_prior()},
        "b": normal_prior(),
        "sigma": bmb.Prior("HalfNormal", sigma=1),
    }
    return bmb.Model(
        formula,
        weighted_nonlinear_data() if data is None else data,
        priors=priors,
        center_predictors=False,
        dropna=dropna,
    )


def censored_nonlinear_data(statuses=("none", "right", "none", "right")):
    return pd.DataFrame(
        {
            "x": [-1.0, 0.0, 0.5, 1.0][: len(statuses)],
            "z": [0.0, 1.0, 0.5, -0.5][: len(statuses)],
            "y": [-0.8, 0.4, 1.3, 2.5][: len(statuses)],
            "status": list(statuses),
        }
    )


def censored_nonlinear_model(data, dropna=False):
    formula = bmb.Formula("censored(y, status) ~ a + rate * x", "a ~ 1 + z", nlpars=("a", "rate"))
    priors = {
        "a": {"Intercept": normal_prior(), "z": normal_prior()},
        "rate": normal_prior(),
        "sigma": bmb.Prior("HalfNormal", sigma=1),
    }
    return bmb.Model(formula, data, priors=priors, center_predictors=False, dropna=dropna)


def competing_risks_nonlinear_data():
    return pd.DataFrame(
        {
            "time": [1.0, 2.0, 1.5, 3.0, 2.5, 4.0, 1.2, 2.2],
            "status": ["event", "right", "event", "right", "event", "right", "event", "event"],
            "cause": [
                "cause_a",
                "none",
                "cause_b",
                "none",
                "cause_a",
                "none",
                "cause_b",
                "cause_a",
            ],
            "x": [-1.0, -0.4, 0.0, 0.2, 0.7, 1.3, 0.5, -0.2],
            "z": [0.0, 1.0, 0.2, -0.4, 0.3, 1.1, -0.3, 0.6],
        }
    )


class VectorNormalFamily(bmb.Family):
    DATA_TYPE = ResponseType.NUMERIC
    RESPONSE_NDIM = 1
    PARAMETERS = {"mu": ParamSpec(links=["identity"], ndim=1, coefs_dim=DimType.RESPONSE)}


VECTOR_NORMAL_COV = np.array([[1.0, 0.2], [0.2, 2.0]])


def vector_normal_family():
    likelihood = bmb.Likelihood(
        "VectorNormal",
        params=["mu"],
        parent="mu",
        dist=partial(pm.MvNormal, cov=VECTOR_NORMAL_COV),
    )
    return VectorNormalFamily("vector_normal", likelihood, "identity")


def vector_normal_data(size=6):
    x = np.linspace(-1, 1, size)
    z = np.linspace(0.2, 1.1, size)
    y1 = 0.2 + 0.4 * x - 0.1 * z
    y2 = -0.3 + 0.2 * x + 0.3 * z
    return pd.DataFrame({"x": x, "z": z, "y1": y1, "y2": y2})


def join_vector_response(first, second):
    return np.column_stack((first, second))


class VectorStudentTFamily(bmb.Family):
    RESPONSE_NDIM = 1
    PARAMETERS = {
        "mu": ParamSpec(links=["identity"], ndim=1, coefs_dim=DimType.RESPONSE),
        "nu": ParamSpec(links=["log"]),
    }


def vector_student_t_family():
    likelihood = bmb.Likelihood(
        "VectorStudentT",
        params=["mu", "nu"],
        parent="mu",
        dist=partial(pm.MvStudentT, scale=VECTOR_NORMAL_COV),
    )
    return VectorStudentTFamily("vector_student_t", likelihood, {"mu": "identity", "nu": "log"})
