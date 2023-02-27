from typing import Any

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
        self.m = None
        self.L = None
        self.c = None
        self.by = None
        self.cov = None
        self.share_cov = None
        self.drop_first = None
        self.centered = None
        self.mean = None
        self.params_set = False
        self.variables_n = None
        self.groups_n = None

    # pylint: disable = redefined-outer-name
    def __call__(
        self,
        *x,
        m,
        L=None,
        c=None,
        by=None,
        cov="ExpQuad",
        share_cov=True,
        drop_first=False,
        centered=False,
    ):
        """_summary_

        See `pymc.gp.HSGP` for more details about the parameters `m`, `L`, `c`, and `drop_first`.

        Parameters
        ----------
        m : int, Sequence[int], Sequence[Sequence[int]], ndarray
            The number of basis vectors. See `HSGP.reconciliate_shape` to see how it is
            broadcasted/recycled.
        L : float, Sequence[float], Sequence[Sequence[float]], ndarray, optional
            The boundary of the variable space. See `HSGP.reconciliate_shape` to see how it is
            broadcasted/recycled. Defaults to `None`.
        c : float, Sequence[float], Sequence[Sequence[float]], ndarray, optional
            The proportion extension factor. Se `HSGP.reconciliate_shape` to see how it is
            broadcasted/recycled. Defaults to `None`.
        by : array-like, optional
            The values of a variable to group by. It is used to create a HSGP term by group.
            Defaults to `None`.
        cov : str, Sequence[str], optional
            The name of the covariance function to use. If it is a sequence, each element is
            the name of the covariance function for every group. Defaults to "ExpQuad".
        share_cov : bool, optional
            Whether to share the same covariance function for every group. Defaults to `True`.
        drop_first : bool, optional
            Whether to ignore the first basis vector or not. Defaults to `False`.
        centered : bool, optional
            Whether to use the centered or the non-centered parametrization. Defaults to `False`.

        TODO: Assert original shapes of 'c', 'L' and 'm'.

        Returns
        -------
        _type_
            _description_

        Raises
        ------
        ValueError
            When both `L` and `c` are `None` or when both of them are not `None` at the same time.
        """
        values = np.column_stack(x)
        self.by = np.asarray(by) if by is not None else by  # can change with new data

        if not self.params_set:
            if (L is None and c is None) or (L is not None and c is not None):
                raise ValueError("Provide one of `c` or `L`")

            # Number of variables and number of groups
            self.variables_n = values.shape[1]
            self.groups_n = 1 if self.by is None else len(np.unique(self.by))

            # The number of basis functions cannot vary by level of the grouping variable
            # It makes the implementation simpler and... why would you do that?!
            self.m = self.recycle_parameter(m, self.variables_n, 1)

            if L is not None:
                L = self.recycle_parameter(L, self.variables_n, self.groups_n)
            if c is not None:
                c = self.recycle_parameter(c, self.variables_n, self.groups_n)

            # As many covariance functions as groups.
            # If `share_cov` is `True`, the first value in `cov` will be used, no matter how many
            # groups there are.
            if isinstance(cov, str):
                cov = (cov,) * self.groups_n

            self.L = L
            self.c = c
            self.cov = cov
            self.share_cov = share_cov
            self.drop_first = drop_first
            self.centered = centered
            self.mean = mean_by_group(values, self.by)
            self.params_set = True

        return values

    @staticmethod
    def recycle_parameter(value, variables_n: int, groups_n: int) -> list[list[Any]]:
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
            output = np.tile(value, (groups_n, variables_n))
        elif len(shape) == 1:
            output = np.tile(value, (groups_n, 1))
        elif len(shape) == 2:
            assert shape == (groups_n, variables_n)
            output = value
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
