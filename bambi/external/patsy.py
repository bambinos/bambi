from __future__ import absolute_import
import numpy as np
import re
from patsy.util import safe_scalar_isnan
from patsy import PatsyError


class Custom_NA(object):
    """
    Custom patsy.missing.NAAction class. Similar to patsy drop/raise defaults,
    but changes the raised message and logs which rows (if any) are dropped.
    See Patsy code/API for NAAction documentation.
    """
    def __init__(self, dropna=False, NA_types=["None", "NaN"]):
        self.dropna = dropna
        if isinstance(NA_types, str):
            raise ValueError("NA_types should be a list of strings")
        self.NA_types = tuple(NA_types)
        self.completes = []

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
        if not self.dropna:
            return self._handle_raise(values, is_NAs, origins)
        elif self.dropna:
            return self._handle_drop(values, is_NAs, origins)
        else:
            return values

    def _handle_raise(self, values, is_NAs, origins):
        for is_NA, origin in zip(is_NAs, origins):
            if np.any(is_NA):
                msg = "Missing values detected. If you want rows with missing "\
                      "values to be automatically deleted in a list-wise " \
                      "manner (not recommended), please set dropna=True in " \
                      "the bambi Model initialization."
                raise PatsyError(msg, origin)
        return values

    def _handle_drop(self, values, is_NAs, origins):
        if not len(is_NAs): return values
        total_mask = np.zeros(is_NAs[0].shape[0], dtype=bool)
        for is_NA in is_NAs:
            total_mask |= is_NA
        good_mask = ~total_mask
        self.completes.append(np.where(good_mask)[0])
        # "..." to handle 1- versus 2-dim indexing
        return [v[good_mask, ...] for v in values]


def rename_columns(columns, name_lists):
    ''' Renames numerical indices in column names returned by patsy dmatrix /
    dmatrices calls based on the corresponding string levels.
    Args:
        columns (list): List of cols from dmatrix's .design_info.column_names
        name_lists (list): List of lists, where the i'th list gives the set of
            level names for the i'th index that needs to be replaced.
    Example:
        ts = ['threecats[1]:subjects[4]', 'threecats[0]:subjects[3]']
        rename_columns(ts, (['a', 'b', 'c'], ['d1', 'd2', 'd3', 'd4', 'd5']))
        # Returns ['threecats[b]:subjects[d5]', 'threecats[a]:subjects[d4]']
    '''
    def _replace(c, args):
        grps = re.findall('([^\]]*)(\[(\d+)\])', c)
        for i, (prefix, box, ind) in enumerate(grps):
            c = c.replace(prefix + box, prefix + '[%s]' % args[i][int(ind)])
        return c
    return [_replace(c, name_lists) for c in columns]
