import numpy as np
import pandas as pd

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
class HSGP:  # pylint: disable = too-many-instance-attributes
    __transform_name__ = "hsgp"

    def __init__(self):
        self.m = None
        self.L = None
        self.c = None
        self.by_levels = None
        self.cov = None
        self.share_cov = None
        self.scale = None
        self.iso = None
        self.drop_first = None
        self.centered = None
        self.mean = None
        self.maximum_distance = None
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
        scale=None,
        iso=True,
        drop_first=False,
        centered=False,
    ):
        """Evaluate the values and set internal parameters

        See `pymc.gp.HSGP` for more details about the parameters `m`, `L`, `c`, and `drop_first`.

        Parameters
        ----------
        m : int, Sequence[int], ndarray
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
        cov : str, optional
            The name of the covariance function to use. Defaults to "ExpQuad".
        share_cov : bool, optional
            Whether to share the same covariance function for every group. Defaults to `True`.
        scale : bool, optional
            When `True`, the predictors are be rescaled such that the largest Euclidean
            distance between two points is 1. This adjustment often improves the sampling speed and
            convergence. The rescaling also impacts the estimated length-scale parameters,
            which will resemble those of the scaled predictors rather than the original predictors
            when `scale` is `True`. Defaults to `None`, which means the behavior depends on whether
            custom priors are passed or not. If custom priors are used, `None` is translated to
            `False`. If automatic priors are used, `None` is translated to `True`.
        iso : bool, optional
            Determines whether to use an isotropic or non-isotropic Gaussian Process.
            If isotropic, the same level of smoothing is applied to all predictors,
            while non-isotropic GPs allow different levels of smoothing for individual predictors.
            This parameter is ignored if only one predictor is supplied. Defaults to `True`.
        drop_first : bool, optional
            Whether to ignore the first basis vector or not. Defaults to `False`.
        centered : bool, optional
            Whether to use the centered or the non-centered parametrization. Defaults to `False`.

        Returns
        -------
        values
            A NumPy array of shape (observations_n, variables_n) or
            (observations_n, variables_n + 1) if `by` is not None.

        Raises
        ------
        ValueError
            When both `L` and `c` are `None` or when both of them are not `None` at the same time.
        """
        values = np.column_stack(x)

        if by is not None:
            # Generate indexes according to the original 'by_levels'
            if self.params_set:
                by_indexes = pd.Categorical(by, categories=self.by_levels).codes
            # Determine unique levels and store them, only for the first time
            else:
                by_levels, by_indexes = np.unique(by, return_inverse=True)
                self.by_levels = by_levels
        else:
            by_indexes = None

        if not self.params_set:
            if (L is None and c is None) or (L is not None and c is not None):
                raise ValueError("Provide one of 'c' or 'L'")

            # Number of variables and number of groups
            self.variables_n = values.shape[1]
            self.groups_n = 1 if self.by_levels is None else len(self.by_levels)

            m = np.asarray(m)
            if not (m.ndim == 0 or m.shape == (self.variables_n,)):
                raise ValueError(
                    "'m' must be scalar or a sequence with length equal to the number of variables"
                )

            # The number of basis functions cannot vary by level of the grouping variable
            # It makes the implementation simpler and... why would you do that?!
            self.m = self.recycle_parameter(m, self.variables_n, 1)

            if L is not None:
                L = self.recycle_parameter(L, self.variables_n, self.groups_n)
            if c is not None:
                c = self.recycle_parameter(c, self.variables_n, self.groups_n)

            self.L = L
            self.c = c
            self.cov = cov
            self.share_cov = share_cov
            self.scale = scale
            self.iso = iso
            self.drop_first = drop_first
            self.centered = centered
            self.mean = mean_by_group(values, by)
            self.maximum_distance = np.max(get_distance(values))
            self.params_set = True

        if by_indexes is not None:
            # The indexes of the 'by' variable is the last column of the matrix returned
            # Note this would certainly cast variables from int to float
            # So we must take care of it when using the indexes in 'by'
            values = np.column_stack([values, by_indexes])

        return values

    @staticmethod
    def recycle_parameter(value, variables_n: int, groups_n: int):
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
            if shape != (variables_n,):
                raise ValueError("1D sequences must be of shape (variables_n, )")
            output = np.tile(value, (groups_n, 1))
        elif len(shape) == 2:
            if shape != (groups_n, variables_n):
                raise ValueError("2D sequences must be of shape (groups_n, variables_n)")
            output = value
        return output


def as_matrix(x):
    """Converts array to matrix

    Parameters
    ----------
    x : np.ndarray
        Array

    Returns
    -------
    np.ndarray
        A two dimensional array

    Raises
    ------
    ValueError
        If the input has more than two dimensions
    """
    x = np.atleast_1d(x)
    if x.ndim == 1:
        return x[:, np.newaxis]
    elif x.ndim > 2:
        raise ValueError("'x.ndim' cannot be > 2")
    return x


def mean_by_group(values, group):
    """Compute the mean value by group

    Parameters
    ----------
    values : np.ndarray
        A 2 dimensional array. Rows indicate observations and columns indicate different variables.
    group : sequence
        A sequence that indicates to which group each observation belongs to. If `None`, then
        no group exists.

    Returns
    -------
    np.ndarray
        An array with the mean values for all the variables, per group, if there's a group.
        It's of shape (groups_n, variables_n).
    """
    if group is None:
        return np.mean(values, axis=0)
    levels = np.unique(group)
    means = np.zeros((len(levels), values.shape[1]))
    for i, level in enumerate(levels):
        means[i] = np.mean(values[group == level], axis=0)
    return means


def get_distance(x):
    """Computes the Euclidean distance between observations

    The input is an array of shape `(n, p)` where rows represent observations and columns represent
    variables. The output is an array of shape `(n, n)` where the values represent the Euclidean
    distance between observations considering all the `p` variables.
    """
    x = as_matrix(x)
    out = 0
    for i in range(x.shape[1]):
        out = out + np.subtract.outer(x[:, i], x[:, i]) ** 2
    return np.sqrt(out)


# These functions are made available in the namespace where the model formula is evaluated
transformations_namespace = {
    "c": c,
    "censored": censored,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "exp": np.exp,
    "exp2": np.exp2,
    "abs": np.abs,
}
