import numpy as np
import pandas as pd

from formulae.transforms import CyclicCubicSpline, NaturalCubicSpline, register_stateful_transform


class SmoothTransform:
    """Parent class for penalized smooths.

    `unpenalized_prior_keys` names the null-space coefficients in their order in the
    random-effects basis before centering. Centering removes leading null-space directions.
    """

    unpenalized_prior_keys = ()


@register_stateful_transform
class CRSpline(SmoothTransform, NaturalCubicSpline):
    """Natural cubic spline as a random-effects term.

    The first `null_space_dimension` columns are unpenalized,
    while remaining columns have an identity curvature penalty.
    Overrides formulae's `cr` transform to return the basis in random-effects coordinates.

    When `by` is set, fit a separate basis to each observed group, using that group's knots,
    boundaries, centering constraint and linear standardization.
    Explicit knots and boundaries apply to every group.

    Attributes
    ----------
    by_levels : np.ndarray or None
        If `by` is specified, it contains the observed group levels. Otherwise `None`.
        Order is preserved when `by` is ordered. If not, levels are sorted alphabetically.
    by_indexes : np.ndarray or None
        If `by` is specified, group index for each observation. Otherwise `None`.
    shared : bool
        Whether curvature prior parameters are shared across groups.
    group_splines : list of CRSpline or None
        If `by` is specified, fitted group-specific transforms in `by_levels` order.
        Otherwise `None`.
    basis_dimension : int or None
        Number of columns per group after centering.
    unpenalized_prior_keys : tuple of str
        Names of the unpenalized directions before centering: `("constant", "linear")`.
        Centering removes the constant direction.
    """

    __transform_name__ = "cr"
    unpenalized_prior_keys = ("constant", "linear")

    def __init__(self):
        super().__init__()
        self.by_levels = None
        self.by_indexes = None
        self.shared = False
        self.group_splines = None
        self.basis_dimension = None

    def __call__(
        self,
        x: pd.Series,
        df: int | None = None,
        knots: np.ndarray | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
        center: bool = True,
        by: pd.Series | None = None,
        shared: bool = False,
    ):
        """Evaluate the spline basis in random-effects coordinates.

        Parameters
        ----------
        x : pd.Series
            Numeric predictor.
        df : int, optional
            Basis size per group after centering. Defaults to 10 when knots are omitted.
        knots : array-like, optional
            Interior knots. By default, use quantiles of unique values.
        lower_bound : float, optional
            Knots lower boundary. By default, use data minimum.
        upper_bound : float, optional
            Knots upper boundary. By default, use data maximum.
        center : bool, optional
            Constrain the smooth's contribution to sum to zero. Defaults to True.
        by : pd.Series, optional
            Categorical grouping variable. Defaults to None, for a single smooth.
        shared : bool, optional
            Share all random parameters of the curvature prior across groups. Defaults to False.
            Ignored without `by`.

        Returns
        -------
        np.ndarray
            Basis matrix with one row per observation. With `by`, columns form consecutive
            blocks in `by_levels` order. Each row is zero outside its group's block.

        Notes
        -----
        For grouped smooths, supply `by` when evaluating new observations.
        """
        if by is None:
            return super().__call__(x, df, knots, lower_bound, upper_bound, center)

        if not self.params_set:
            if not (
                isinstance(by.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(by.dtype)
                or pd.api.types.is_string_dtype(by.dtype)
            ):
                raise ValueError("'by' must be categorical.")

            if pd.isna(by).any():
                raise ValueError("'by' cannot contain missing values.")

            if isinstance(by.dtype, pd.CategoricalDtype) and by.dtype.ordered:
                self.by_levels = np.asarray(
                    [level for level in by.dtype.categories if level in set(by)]
                )
            else:
                self.by_levels = np.unique(by)

            self.by_indexes = pd.Categorical(by, categories=self.by_levels).codes
            self.shared = shared
            self._center = bool(center)
            self.group_splines = []
            x = np.asarray(x)
            by = np.asarray(by)

            for level in self.by_levels:
                spline = CRSpline()
                basis = spline(x[by == level], df, knots, lower_bound, upper_bound, center)
                self.group_splines.append(spline)

            self.basis_dimension = basis.shape[1]
            self.params_set = True
        return self.eval(x, by)

    def eval(self, x, by=None):
        # When `by` is used, we evaluate each group separately by calling `eval` on the
        # corresponding group-specific spline. Each of those calls follows the non-grouped
        # path at the bottom, so `to_random` is applied once for each group.
        # When `by` is not used, there is only one evaluation and `to_random` is applied once.
        if self.by_levels is not None:
            if by is None:
                raise ValueError("Supply 'by' when evaluating a grouped smooth.")

            by = np.asarray(by)
            x = np.asarray(x)
            indexes = pd.Categorical(by, categories=self.by_levels).codes
            if np.any(indexes < 0):
                unknown = pd.unique(by[indexes < 0]).tolist()
                raise ValueError(f"Unknown level(s) in smooth 'by': {unknown}.")

            # Group-major blocks keep formulae's design matrix two-dimensional.
            basis = np.zeros((len(x), len(self.by_levels), self.basis_dimension))
            for i, spline in enumerate(self.group_splines):
                rows = indexes == i
                if np.any(rows):
                    basis[rows, i, :] = spline.eval(x[rows])
            return basis.reshape((len(x), -1))

        return self.to_random(super().eval(x))


@register_stateful_transform
class CCSpline(SmoothTransform, CyclicCubicSpline):
    """Cyclic cubic spline as a random-effects term.

    The cyclic spline has a constant null-space direction when uncentered and no null-space
    directions when centered. The remaining columns have an identity curvature penalty.
    Overrides formulae's `cc` transform to return the basis in random-effects coordinates.
    Its values and first two derivatives match at the period boundaries.

    When `by` is set, fit a separate basis to each observed group, using that group's knots,
    boundaries and centering constraint. The period is the same for every group.
    Explicit knots and the lower boundary apply to every group.

    Attributes
    ----------
    by_levels : np.ndarray or None
        If `by` is specified, it contains the observed group levels. Otherwise `None`.
        Order is preserved when `by` is ordered. If not, levels are sorted.
    by_indexes : np.ndarray or None
        If `by` is specified, group index for each training observation. Otherwise `None`.
    shared : bool
        Whether curvature prior parameters are shared across groups.
    group_splines : list of CCSpline or None
        If `by` is specified, fitted group-specific transforms in `by_levels` order.
        Otherwise `None`.
    basis_dimension : int or None
        Number of columns per group after centering. `None` without `by`.
    period : float or None
        Length of the cycle. For grouped smooths, access the period through
        `group_splines`; this attribute is `None`.
    unpenalized_prior_keys : tuple of str
        Names of the unpenalized directions before centering: `("constant",)`.
        Centering removes the constant direction.
    """

    __transform_name__ = "cc"
    unpenalized_prior_keys = ("constant",)

    def __init__(self):
        super().__init__()
        self.by_levels = None
        self.by_indexes = None
        self.shared = False
        self.group_splines = None
        self.basis_dimension = None

    def __call__(
        self,
        x: pd.Series,
        period: float,
        df: int | None = None,
        knots: np.ndarray | None = None,
        lower_bound: float | None = None,
        center: bool = True,
        by: pd.Series | None = None,
        shared: bool = False,
    ):
        """Evaluate the spline basis in random-effects coordinates.

        Parameters
        ----------
        x : pd.Series
            Numeric periodic predictor.
        period : float
            Length of the cycle. Must be finite and greater than zero.
        df : int, optional
            Basis size per group after centering. Defaults to 10 when knots are omitted.
            If only `knots` are supplied, the basis size is inferred from them.
        knots : array-like, optional
            Interior knots, strictly between the period boundaries. By default, use
            quantiles of unique observed phases within the period.
        lower_bound : float, optional
            Start of the cycle. Defaults to the data minimum, separately within each group
            when `by` is set. The upper boundary is `lower_bound + period`.
        center : bool, optional
            Constrain the smooth's contribution to sum to zero over the training observations,
            separately within each group when `by` is set. Defaults to True.
        by : pd.Series, optional
            Categorical grouping variable. Defaults to None, for a single smooth.
        shared : bool, optional
            Share all random parameters of the curvature prior across groups. Defaults to False.
            Ignored without `by`.

        Returns
        -------
        np.ndarray
            Basis matrix with one row per observation. With `by`, columns form consecutive
            blocks in `by_levels` order. Each row is zero outside its group's block.

        Notes
        -----
        Values outside the period boundaries are wrapped into the original interval.
        For grouped smooths, supply `by` when evaluating new observations.
        """
        if by is None:
            return super().__call__(x, period, df, knots, lower_bound, center)

        if not self.params_set:
            if not (
                isinstance(by.dtype, pd.CategoricalDtype)
                or pd.api.types.is_object_dtype(by.dtype)
                or pd.api.types.is_string_dtype(by.dtype)
            ):
                raise ValueError("'by' must be categorical.")

            if pd.isna(by).any():
                raise ValueError("'by' cannot contain missing values.")

            if isinstance(by.dtype, pd.CategoricalDtype) and by.dtype.ordered:
                self.by_levels = np.asarray(
                    [level for level in by.dtype.categories if level in set(by)]
                )
            else:
                self.by_levels = np.unique(by)

            self.by_indexes = pd.Categorical(by, categories=self.by_levels).codes
            self.shared = shared
            self._center = bool(center)
            self.group_splines = []
            x = np.asarray(x)
            by = np.asarray(by)

            for level in self.by_levels:
                spline = CCSpline()
                basis = spline(x[by == level], period, df, knots, lower_bound, center)
                self.group_splines.append(spline)

            self.basis_dimension = basis.shape[1]
            self.params_set = True
        return self.eval(x, by)

    def eval(self, x, by=None):
        if self.by_levels is not None:
            if by is None:
                raise ValueError("Supply 'by' when evaluating a grouped smooth.")

            by = np.asarray(by)
            x = np.asarray(x)
            indexes = pd.Categorical(by, categories=self.by_levels).codes
            if np.any(indexes < 0):
                unknown = pd.unique(by[indexes < 0]).tolist()
                raise ValueError(f"Unknown level(s) in smooth 'by': {unknown}.")

            basis = np.zeros((len(x), len(self.by_levels), self.basis_dimension))
            for i, spline in enumerate(self.group_splines):
                rows = indexes == i
                if np.any(rows):
                    basis[rows, i, :] = spline.eval(x[rows])
            return basis.reshape((len(x), -1))

        return self.to_random(super().eval(x))


def c(*args):
    """Concatenate columns into a 2D NumPy Array."""
    return np.column_stack(args)


def counts(*args, n=None):
    """Construct an array of counts for a multinomial response.

    Parameters
    ----------
    *args : array-like
        Count columns, one for each category.
    n : int, array-like, optional
        The total number of counts per observation. When omitted, it is computed by summing the
        count columns.
    """
    data = np.column_stack(args)
    totals = data.sum(axis=1)

    if n is not None:
        n = np.asarray(n)
        if n.ndim > 1:
            raise ValueError("'n' must be a scalar or a 1-dimensional array.")
        if n.ndim == 1 and len(n) != len(data):
            raise ValueError("The length of 'n' must be equal to the number of observations.")
        if not np.all(totals == n):
            raise ValueError("The counts in each row must sum to 'n'.")

    return data


counts.__metadata__ = {"kind": "counts"}


def censored(x, status):
    """Construct array for censored response

    The first value has the values of the variable and the second value contains the censoring
    statuses.

    Valid censoring statuses are:

    - "left": left censoring
    - "none": no censoring
    - "right": right censoring

    Returns
    -------
    np.ndarray
        Array of shape (n, 2). The first column contains the values of the variable and the second
        column contains the censoring statuses.
    """
    status_mapping = {"left": -1, "none": 0, "right": 1}

    assert len(x) == len(status)

    assert all(s in status_mapping for s in status), f"Statuses must be in {list(status_mapping)}"
    status = np.asarray([status_mapping[s] for s in status])
    return np.column_stack([x, status])


censored.__metadata__ = {"kind": "censored"}


@register_stateful_transform
class CompetingRisks:
    """Competing-risks response with separate time-status and cause encodings.

    `status` describes the information available about the event time and must be one of
    `"event"` or `"right"`. They are internally encoded as 0 and 1, respectively.
    `cause` identifies the event type independently of `status`.
    Known causes are encoded deterministically in sorted order as 1, 2, ...;
    `"none"` is encoded as 0.

    An exact event requires a known cause. A right-censored observation requires `cause="none"`.
    Left censoring is not supported.

    Returns
    -------
    np.ndarray
        An array of shape `(n, 3)` containing event/censoring times, integer status codes, and
        integer cause codes.
    """

    __transform_name__ = "risks"
    __metadata__ = {"kind": "risks"}

    def __init__(self):
        self.cause_codes = None

    def __call__(self, y, status, cause):
        y = np.asarray(y)
        status = np.asarray(status)
        cause = np.asarray(cause)

        if y.ndim != 1 or status.ndim != 1 or cause.ndim != 1:
            raise ValueError("'risks' inputs must be one-dimensional.")

        if len(y) != len(status) or len(y) != len(cause):
            raise ValueError("'y', 'status', and 'cause' must have the same length.")

        if pd.isna(status).any():
            raise ValueError("'status' cannot contain missing values.")

        if np.any(status == "left"):
            raise ValueError("Left censoring is not supported for competing-risks responses.")

        if pd.isna(cause).any():
            raise ValueError("'cause' cannot contain missing values. Use 'none' instead.")

        status_codes = {"event": 0, "right": 1}
        unknown_statuses = set(status) - set(status_codes)
        if unknown_statuses:
            raise ValueError(
                "'status' must contain only 'event' or 'right'; "
                f"got {sorted(unknown_statuses, key=repr)!r}."
            )

        cause_is_none = cause == "none"
        is_event = status == "event"
        is_right = status == "right"
        if np.any(is_event & cause_is_none):
            raise ValueError("'cause' must not be 'none' when status is 'event'.")

        if np.any(is_right & ~cause_is_none):
            raise ValueError("'cause' must be 'none' when status is 'right'.")

        if self.cause_codes is None:
            cause_levels = sorted(set(cause[~cause_is_none]), key=repr)
            if not cause_levels:
                raise ValueError("A competing-risks response requires at least one observed cause.")
            self.cause_codes = {level: code for code, level in enumerate(cause_levels, start=1)}

        unknown_causes = set(cause[~cause_is_none]) - set(self.cause_codes)
        if unknown_causes:
            raise ValueError(
                f"Unknown competing-risks cause level(s): {sorted(unknown_causes, key=repr)}"
            )

        # `column_stack` may promote codes to float; the backend casts them back to integers.
        status_out = np.array([status_codes[level] for level in status], dtype=int)
        cause_out = np.zeros(len(cause), dtype=int)
        cause_out[~cause_is_none] = [self.cause_codes[level] for level in cause[~cause_is_none]]
        return np.column_stack([y, status_out, cause_out])


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
    one truncates/constraints the bounds of a probability distribution, while `truncated()` is
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
    if x.ndim > 2:
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
    "cr": CRSpline,
    "c": c,
    "counts": counts,
    "censored": censored,
    "risks": CompetingRisks,
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
