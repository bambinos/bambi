import re
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
        success = 1
    # If string/object/categorical, convert to 0-1 using first value as reference
    elif is_string_dtype(data) or is_categorical_dtype(data):
        success = data.values[0]
        data = pd.Series(np.where(data.values == success, 1, 0))
    else:
        raise ValueError("Response variable is of the wrong type.")
    return data, success


def extract_label(string, kind="common"):
    """Receives a level as returned by patsy and returns a cleaned version of it.
    common and group specific effects are handled differently because their representations differ.

    Parameters
    ----------
    string: str
        A string containing a label, as returned by patsy.
    kind: str
        Indicates if this is a common or group specific effect. Must be 'common' or 'group_specific'

    Returns
    -------
    out: cleaned string with label
    """
    out = re.search(r"\[(.+)\]", str(string))
    if out is None:
        return string
    else:
        out = out.group(1)
    if kind == "common":
        out = re.sub(r"^(.*?)\.", "", out)

    return out
