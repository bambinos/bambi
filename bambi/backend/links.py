import numpy as np
import pytensor.tensor as pt


def probit(x):
    """Probit function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = 0.5 + 0.5 * pt.erf(x / pt.sqrt(2))
    result = pt.switch(pt.eq(result, 0), eps, result)
    result = pt.switch(pt.eq(result, 1), 1 - eps, result)

    return result


def cloglog(x):
    """Cloglog function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = 1 - pt.exp(-pt.exp(x))
    result = pt.switch(pt.eq(result, 0), eps, result)
    result = pt.switch(pt.eq(result, 1), 1 - eps, result)

    return result


def logit(x):
    """Logit function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = pt.sigmoid(x)
    result = pt.switch(pt.eq(result, 0), eps, result)
    result = pt.switch(pt.eq(result, 1), 1 - eps, result)
    return result


def identity(x):
    return x


def inverse_squared(x):
    return pt.reciprocal(pt.sqrt(x))


def arctan_2(x):
    return 2 * pt.arctan(x)
