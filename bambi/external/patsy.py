import numpy as np

class Ignore_NA(object):
    """
    Custom patsy.missing.NAAction class to force Patsy to ignore missing values.
    See Patsy code/API for NAAction documentation.
    """
    def __init__(self, on_NA="ignore", NA_types=["None", "NaN"]):
        self.on_NA = on_NA
        if isinstance(NA_types, str):
            raise ValueError("NA_types should be a list of strings")
        self.NA_types = tuple(NA_types)

    def is_categorical_NA(self, obj):
        if "NaN" in self.NA_types and safe_scalar_isnan(obj):
            return True
        if "None" in self.NA_types and obj is None:
            return True
        return False

    def is_numerical_NA(self, arr):
        mask = np.zeros(arr.shape, dtype=bool)
        if "NaN" in self.NA_types:
            mask |= np.isnan(arr)
        if mask.ndim > 1:
            mask = np.any(mask, axis=1)
        return mask

    def handle_NA(self, values, is_NAs, origins):
        return values
