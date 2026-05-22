import numpy as np
import pandas as pd

from formulae.transforms import register_stateful_transform


def c(*args):
    """Concatenate columns into a 2D NumPy Array."""
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

    Valid censoring statuses are:

    - "left": left censoring
    - "none": no censoring
    - "right": right censoring
    - "interval": interval censoring

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


def truncated(x, lb=None, ub=None):
    """Construct array for a truncated response

    Parameters
    ----------
    x : np.ndarray
        The values of the truncated variable.
    lb : int, float, np.ndarray
        A number or an array indicating the lower truncation bound.
    ub : int, float, np.ndarray
        A number or an array indicating the upper truncation bound.

    Returns
    -------
    np.ndarray
        Array of shape (n, 3). The first column contains the values of the variable,
        the second column the values for the lower bound, and the third variable
        the values for the upper bound.
    """
    x = np.asarray(x)

    if x.ndim != 1:
        raise ValueError("'truncated' only works with 1-dimensional arrays")

    if lb is None and ub is None:
        raise ValueError("'lb' and 'ub' cannot both be None")

    # Process lower bound so we get a 1d array with the adequate values
    if lb is not None:
        lower = np.asarray(lb)
        if lower.ndim == 0:
            lower = np.full(len(x), lower)
        elif lower.ndim == 1:
            assert len(lower) == len(x), "The length of 'lb' must be equal to the one of 'x'"
        else:
            raise ValueError("'lb' must be 0 or 1 dimensional.")
    else:
        lower = np.full(len(x), -np.inf)

    # Process upper bound so we get a 1d array with the adequate values
    if ub is not None:
        upper = np.asarray(ub)
        if upper.ndim == 0:
            upper = np.full(len(x), upper)
        elif upper.ndim == 1:
            assert len(upper) == len(x), "The length of 'ub' must be equal to the one of 'x'"
        else:
            raise ValueError("'ub' must be 0 or 1 dimensional.")
    else:
        upper = np.full(len(x), np.inf)

    # Construct output matrix
    result = np.column_stack([x, lower, upper])

    return result


truncated.__metadata__ = {"kind": "truncated"}


def constrained(x, lb=None, ub=None):
    """Construct an array for a constrained response

    It's exactly like truncated, but it's interpreted by Bambi in a different way as this
    one truncates/constrains the bounds of a probability distribution, while `truncated()` is
    interpreted as the missing data mechanism.

    `lb` and `ub` can only be scalar values.
    """
    if not (lb is None or isinstance(lb, (int, float))):
        raise ValueError("'lb' must be None or scalar.")

    if not (ub is None or isinstance(ub, (int, float))):
        raise ValueError("'ub' must be None or scalar.")
    return truncated(x, lb, ub)


constrained.__metadata__ = {"kind": "constrained"}


def weighted(x, weights):
    """Construct array for a weighted response

    Parameters
    ----------
    x : np.ndarray
        The values of the truncated variable.
    weights : np.ndarray
        The weight of each value in `x`.

    Returns
    ------
    np.ndarray
        Array of shape (n, 2). The first column contains the values of the `x` array and the second
        contains the values of `weights`.
    """
    x = np.asarray(x)
    weights = np.asarray(weights)

    if any(weights < 0):
        raise ValueError("Weights must be positive.")

    return np.column_stack([x, weights])


weighted.__metadata__ = {"kind": "weighted"}


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
            The number of basis vectors. See `HSGP.reconciliate_shape`? to see how it is
            broadcasted/recycled.
        L : float, Sequence[float], Sequence[Sequence[float]], ndarray, optional
            The boundary of the variable space. See `HSGP.reconciliate_shape` to see how it is
            broadcasted/recycled. Defaults to `None`.
        c : float, Sequence[float], Sequence[Sequence[float]], ndarray, optional
            The proportion extension factor. Se `HSGP.reconciliate_shape` to see how it is
            broadcasted/recycled. Defaults to `None`.
        by : array-like, optional
            The values of a variable to group by. It is used to create an HSGP term by group.
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
            when `scale` is `True`. Defaults to `None`, which means the behavior depends on
            whether custom priors are passed or not. If custom priors are used, `None` is
            translated to `False`. If automatic priors are used, `None` is translated to
            `True`.
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
            (observations_n, variables_n + 1) if `by` is not `None`.

        Raises
        ------
        ValueError
            When both `L` and `c` are `None` or when both of them are not `None` at the
            same time.
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

        - `int`: the same value is recycled for all variables and groups.
        - `Sequence[int]`: it represents the values by variable and it is recycled for all groups.
        - `Sequence[Sequence[int]]`: it represents the values by variable and by group and thus
        no recycling applies. Must be of shape `(groups_n, variables_n)`.
        - `ndarray`:
            - If one dimensional, it behaves as `Sequence[int]`
            - If two dimensional, it behaves as `Sequence[Sequence[int]]`
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
        else:
            raise ValueError(f"Wrong shape: {shape}")
        return output


@register_stateful_transform
class Monotonic:
    """Stateful transform for monotonic effects ``mo(x)``.

    Mirrors brms' ``mo()``: the user passes an ordered predictor (integer or ordered
    categorical), and the linear predictor contribution becomes ``b * D * sum(zeta[1:x])``
    where ``zeta`` is a length-``D`` simplex (``D = K - 1`` for ``K`` categories) and ``b``
    is a scalar slope. See Bürkner & Charpentier (2020).

    On the first call (during ``Model`` construction), this transform records the levels
    of the predictor and returns an ``(n, 1)`` array of zero-indexed integer codes. On
    subsequent calls (e.g. during ``evaluate_new_data`` for prediction) it re-encodes
    the input against the stored levels and raises on unseen categories/values.
    """

    __transform_name__ = "mo"

    def __init__(self):
        self.levels = None
        self.min_value = None
        self.kind = None  # "ordered" or "integer"
        self.K = None  # number of distinct categories
        self.D = None  # K - 1 (length of the simplex)
        self.params_set = False

    def __call__(self, x, id=None):  # pylint: disable=redefined-builtin
        # ``id`` is accepted for forward compatibility with brms' shared-simplex
        # mechanism but is not used in this MVP.
        del id

        if isinstance(x, pd.Series):
            values = x
        else:
            values = pd.Series(np.asarray(x))

        if self.params_set:
            codes = self._encode(values)
        else:
            codes = self._fit_and_encode(values)

        return codes.reshape(-1, 1).astype("float64")

    def _fit_and_encode(self, values):
        dtype = values.dtype
        if isinstance(dtype, pd.CategoricalDtype):
            if not dtype.ordered:
                raise ValueError(
                    "'mo()' requires an ordered categorical predictor. "
                    "Use 'pd.Categorical(..., ordered=True)' or pass an integer predictor."
                )
            self.kind = "ordered"
            self.levels = np.asarray(dtype.categories)
            codes = values.cat.codes.to_numpy()
        elif pd.api.types.is_integer_dtype(values):
            self.kind = "integer"
            uniques = np.sort(values.dropna().unique())
            self.min_value = int(uniques[0])
            self.levels = uniques
            codes = values.to_numpy() - self.min_value
        else:
            raise ValueError(
                "'mo()' requires an integer or ordered categorical predictor; "
                f"got dtype {dtype!r}."
            )

        if (codes < 0).any():
            raise ValueError("'mo()' received negative or missing values in its predictor.")

        self.K = int(len(self.levels))
        if self.K < 2:
            raise ValueError("'mo()' requires a predictor with at least 2 distinct values.")
        self.D = self.K - 1
        self.params_set = True
        return codes.astype("int64")

    def _encode(self, values):
        if self.kind == "ordered":
            recoded = pd.Categorical(values, categories=self.levels, ordered=True)
            codes = recoded.codes
            if (codes == -1).any():
                bad = np.array(values)[codes == -1]
                raise ValueError(
                    f"'mo()' received unseen categories at prediction time: {sorted(set(bad))}"
                )
            return codes.astype("int64")
        # integer kind
        codes = values.to_numpy() - self.min_value
        max_code = self.D
        if (codes < 0).any() or (codes > max_code).any():
            bad = values[(codes < 0) | (codes > max_code)].unique()
            raise ValueError(
                f"'mo()' received values outside the range seen at fit time: {sorted(bad)}"
            )
        return codes.astype("int64")


def as_matrix(x):
    """Converts array to matrix

    Parameters
    ----------
    x : np.ndarray
        Array.

    Returns
    -------
    np.ndarray
        A two dimensional array.

    Raises
    ------
    ValueError
        If the input has more than two dimensions.
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
        A sequence that indicates to which group each observation belongs to.
        If `None`, then no group exists.

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
    "constrained": constrained,
    "truncated": truncated,
    "weighted": weighted,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "exp": np.exp,
    "exp2": np.exp2,
    "abs": np.abs,
}
