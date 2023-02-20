import numpy as np

from formulae.transforms import register_stateful_transform


def c(*args):  # pylint: disable=invalid-name
    """Concatenate columns into a 2D NumPy Array"""
    return np.column_stack(args)


def censored(*args):
    """Construct array for censored response

    The `args` argument must be of length 2 or 3.
    If it is of length 2, the first value has the values of the variable and the second value
    contains the censoring statuses.

    If it is of length 3, the first value represents either the value of the variable or the lower
    bound (depending on whether it's interval censoring or not). The second value represents the
    upper bound, only if it's interval censoring, and the third argument contains the censoring
    statuses.

    Valid censoring statuses are

    * "left": left censoring
    * "none": no censoring
    * "right": right censoring
    * "interval": interval censoring

    Interval censoring is supported by this function but not supported by PyMC, so Bambi
    does not support interval censoring for now.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) or (n, 3). The first case applies when a single value argument is
        passed, and the second case applies when two values are passed.
    """
    status_mapping = {"left": -1, "none": 0, "right": 1, "interval": 2}

    if len(args) == 2:
        left, status = args
        right = None
    elif len(args) == 3:
        left, right, status = args
    else:
        raise ValueError("'censored' needs 2 or 3 argument values.")

    assert len(left) == len(status)

    if right is not None:
        right = np.asarray(right)
        assert len(left) == len(right)
        assert (right > left).all(), "Upper bound must be larger than lower bound"

    assert all(s in status_mapping for s in status), f"Statuses must be in {list(status_mapping)}"
    status = np.asarray([status_mapping[s] for s in status])

    if right is not None:
        result = np.column_stack([left, right, status])
    else:
        result = np.column_stack([left, status])

    return result


censored.__metadata__ = {"kind": "censored"}

# pylint: disable = invalid-name
@register_stateful_transform
class HSGP:
    __transform_name__ = "hsgp"

    def __init__(self):
        self.by = None
        self.m = None
        self.L = None
        self.c = None
        self.by = None
        self.cov = None
        self.drop_first = None
        self.centered = None
        self.mean = None
        self.params_set = False

    # pylint: disable = redefined-outer-name
    def __call__(
        self, *x, m, L=None, c=None, by=None, cov="ExpQuad", drop_first=False, centered=False
    ):
        values = np.column_stack(x)
        if not self.params_set:
            if (L is None and c is None) or (L is not None and c is not None):
                raise ValueError("Provide one of `c` or `L`")
            self.m = m
            self.L = L
            self.c = c
            self.by = by
            self.cov = cov
            self.drop_first = drop_first
            self.centered = centered
            self.mean = np.mean(values, axis=0)
            self.params_set = True
        return values


# These functions are made available in the namespace where the model formula is evaluated
extra_namespace = {
    "c": c,
    "censored": censored,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "exp": np.exp,
    "exp2": np.exp2,
    "abs": np.abs,
}
