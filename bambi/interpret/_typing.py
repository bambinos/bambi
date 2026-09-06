from typing import Callable, Sequence
from numpy import ndarray
from pandas import Series
from xarray import DataArray

type Scalar = int | float | str
type ConditionalValues = Scalar | Sequence[Scalar] | ndarray | Series

# Strategy type: given a Series, produce default values as a Series
type DefaultStrategy = Callable[[Series], Series]

# A comparison function performs an operation (op) between a reference and
# a contrast DataArray and returns a result DataArray
type ComparisonFunc = Callable[[DataArray, DataArray], DataArray]

# A slope function scales the raw derivative (dydx) given the evaluation point x
# and the response y, and returns a scaled DataArray
type SlopeFunc = Callable[[DataArray, DataArray, DataArray], DataArray]
