import numpy as np
import aesara.tensor as at


def probit(x):
    """Probit function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = 0.5 + 0.5 * at.erf(x / at.sqrt(2))
    result = at.switch(at.eq(result, 0), eps, result)
    result = at.switch(at.eq(result, 1), 1 - eps, result)

    return result


def cloglog(x):
    """Cloglog function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = 1 - at.exp(-at.exp(x))
    result = at.switch(at.eq(result, 0), eps, result)
    result = at.switch(at.eq(result, 1), 1 - eps, result)

    return result


def logit(x):
    """Logit function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = at.sigmoid(x)
    result = at.switch(at.eq(result, 0), eps, result)
    result = at.switch(at.eq(result, 1), 1 - eps, result)
    return result


def identity(x):
    return x


def inverse_squared(x):
    return at.inv(at.sqrt(x))


def arctan_2(x):
    return 2 * at.arctan(x)
