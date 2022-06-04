import numpy as np


def get_unique_levels(x):
    if hasattr(x, "dtype") and hasattr(x.dtype, "categories"):
        levels = list(x.dtype.categories)
    else:
        levels = np.unique(x)
    return levels
