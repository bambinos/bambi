import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_categorical_dtype


def listify(obj):
    """Wrap all non-list or tuple objects in a list.

    Provides a simple way to accept flexible arguments.
    """
    if obj is None:
        return []
    else:
        return obj if isinstance(obj, (list, tuple, type(None))) else [obj]


def get_bernoulli_data(data):
    """Checks and converts a pandas.Series of numerics/string/object/categoric"""
    # If numeric, must be 0-1
    if is_numeric_dtype(data):
        if not all(data.isin([0, 1])):
            raise ValueError("Numeric response must be all 0 and 1 for 'bernoulli' family.")
    # If string/object, convert to 0-1 using first value as reference
    elif is_string_dtype(data) or is_categorical_dtype(data):
        data = pd.Series(np.where(data.values == data.values[0], 1, 0))
    else:
        raise ValueError("Response variable is of the wrong type.")
    return data
