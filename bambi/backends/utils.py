import numpy as np
import theano.tensor as tt


def probit(x):
    """Probit function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = 0.5 + 0.5 * tt.erf(x / tt.sqrt(2))
    result = tt.switch(tt.eq(result, 0), eps, result)
    result = tt.switch(tt.eq(result, 1), 1 - eps, result)

    return result


def cloglog(x):
    """Cloglog function that ensures result is in (0, 1)"""
    eps = np.finfo(float).eps
    result = 1 - tt.exp(-tt.exp(x))
    result = tt.switch(tt.eq(result, 0), eps, result)
    result = tt.switch(tt.eq(result, 1), 1 - eps, result)

    return result
