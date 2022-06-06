import numpy as np


def get_unique_levels(x):
    if hasattr(x, "dtype") and hasattr(x.dtype, "categories"):
        levels = list(x.dtype.categories)
    else:
        levels = np.unique(x)
    return levels


def get_group_offset(n, lower=0.05, upper=0.4):
    # Complementary log log function, scaled.
    # See following code to have an idea of how this function looks like
    # lower, upper = 0.05, 0.4
    # x = np.linspace(2, 9)
    # y = get_group_offset(x, lower, upper)
    # fig, ax = plt.subplots(figsize=(8, 5))
    # ax.plot(x, y)
    # ax.axvline(2, color="k", ls="--")
    # ax.axhline(lower, color="k", ls="--")
    # ax.axhline(upper, color="k", ls="--")
    intercept, slope = 3.25, 1
    return lower + np.exp(-np.exp(intercept - slope * n)) * (upper - lower)
