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
        """_summary_

        Parameters
        ----------
        m : int, Sequence[int], Sequence[Sequence[int]], ndarray
        L : int, Sequence[int], Sequence[Sequence[int]], ndarray, optional
            _description_, by default None
        c : int, Sequence[int], Sequence[Sequence[int]], ndarray, optional
            _description_, by default None
        by : _type_, optional
            _description_, by default None
        cov : str, optional
            _description_, by default "ExpQuad"
        drop_first : bool, optional
            _description_, by default False
        centered : bool, optional
            _description_, by default False

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            _description_
        """
        values = np.column_stack(x)
        self.by = np.asarray(by) if by is not None else by # can change with new data
        if not self.params_set:
            if (L is None and c is None) or (L is not None and c is not None):
                raise ValueError("Provide one of `c` or `L`")
            self.m = m
            self.L = L
            self.c = c
            self.cov = cov
            self.drop_first = drop_first
            self.centered = centered
            self.mean = mean_by_group(values, self.by)
            self.params_set = True
        return values

    @staticmethod
    def reconciliate_shape(value, variables_n: int, groups_n: int):
        """Reshapes a value considering the number of variables and groups

        Parameter values such as `m`, `L`, and `c` may be different for the different variables and
        groups. Internally, the shape of these objects is always `(groups_n, variables_n)`.
        This method contains the logic used to map user supplied values, which may be of different
        shape and nature, into an object of shape `(groups_n, variables_n)`.

        The behavior of the method depends on the type of `value` in the following way. 
        If value is of type...
        * `int`: the same value is recycled for all variables and groups.
        * `Sequence[int]`: it represents the values by variable and it is recycled for all groups.
        * `Sequence[Sequence[int]]`: it represents the values by variable and by group and thus
        no recycling applies. Must be of shape `(groups_n, variables_n)`.
        * `ndarray`:
            * If one dimensional, it behaves as `Sequence[int]`
            * If two dimensional, it behaves as `Sequence[Sequence[int]]`
        """
        value = np.asarray(value)
        shape = value.shape
        if len(shape) == 0:
            output = np.tile(value, (groups_n, variables_n)).tolist()
        elif len(shape) == 1:
            output = np.tile(value, (groups_n, 1)).tolist()
        elif len(shape) == 2:
            assert shape == (groups_n, variables_n)
            output = value.tolist()
        return output


def mean_by_group(values, group):
    if group is None:
        return np.mean(values, axis=0)
    levels = np.unique(group)
    means = np.zeros((len(levels), values.shape[1]))
    for i, level in enumerate(levels):
        means[i] = np.mean(values[group == level], axis=0)
    return means


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
