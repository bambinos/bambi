import pandas as pd
import numpy as np
# import xarray as xr
from six import string_types
from collections import OrderedDict
import transformations
from utils import listify
import warnings


class Model(object):

    def __init__(self, data):
        '''
        Args:
            dataset (DataFrame): the pandas DF containing the data to use.
        '''
        self.data = data
        if 'intercept' not in self.data.columns:
            self.data['intercept'] = 1
        elif self.data['intercept'].nunique() > 1:
            warnings.warn("The input dataset contains an existing column named"
                          " 'intercept' that has more than one unique value. "
                          "Note that this may cause unexpected behavior if "
                          "intercepts are added to the model via add_term() "
                          "calls.")
        self.reset()

    def reset(self):
        self.model = None
        self.cache = OrderedDict()
        self.contrasts = OrderedDict()
        self.X = OrderedDict()
        self.y = None

    def build(self):
        ''' Build the PyMC3 model. '''
        pass

    def run(self):
        ''' Run the MCMC sampler. '''
        if self.model is None:
            self.build()

    def set_y(self, label):
        ''' Set the outcome variable. '''
        if self.y is not None:
            self.X[self.y.label] = self.y
            self.y = self.X.pop(label)

    def add_term(self, *args, **kwargs):
        term = Term(self, *args, **kwargs)
        self.cache[term.hash] = term
        self.X[term.label] = term

    def add_contrast(self, *args, **kwargs):
        pass

    def get_cached_term(self, variable, categorical):
        key = hash((tuple(listify(variable)), categorical))
        if key not in self.cache:
            self.cache[key] = Term(self, variable, categorical=categorical)
        return self.cache[key]

    def transform(self, terms, operations, groupby=None, *args, **kwargs):
        for t in listify(terms):
            for op in listify(operations):
                self.X[t].transform(op, groupby, *args, **kwargs)


def transformer(func):
    def wrapper(self, *args, **kwargs):
       self.transform(func.__name__, *args, **kwargs)
    return wrapper


class Term(object):

    def __init__(self, model, variable, label=None, data=None,
                 categorical=False, random=False, split_by=None,
                 operations=None, plot=False, drop_first=False, **kwargs):

        self.model = model
        self.variable = listify(variable)
        self.label = label or '_'.join(self.variable)
        self.operations = []
        self.categorical = categorical
        self.random = random
        self.split_by = split_by
        self.data_source = data or self.model.data
        self.drop_first = drop_first
        self.level_map = None
        self.hash = hash((tuple(self.variable), categorical))

        # Load data
        self._setup()

        if operations is not None:
            for oper in listify(operations):
                self._apply_operation(oper)

    def _setup(self):

        data = self.data_source[self.variable].copy()

        if self.categorical:
            # Handle multiple variables; will fail gracefully if only 1 exists
            try:
                data = data.stack()
            except: pass
            n_cols = data.nunique()
            levels = data.unique()
            mapping = OrderedDict(zip(levels, list(range(n_cols))))
            self.level_map = mapping
            recoded = data.loc[:, self.variable].replace(mapping).values
            data = pd.get_dummies(recoded, drop_first=self.drop_first)

        else:
            if  len(self.variable) > 1:
                raise ValueError("Adding a list of terms is only "
                        "supported for categorical variables "
                        "(e.g., random factors).")
            data = data.convert_objects(convert_numeric=True)

        if self.split_by is None:
            self.values = data.values
        else:
            split_dm = self.model.get_cached_term(self.split_by, True).values
            self.values = np.einsum('ab,ac->abc', data.values, split_dm)

        self.data = data

    def transform(self, operation, groupby=None, copy=False,
                         *args, **kwargs):
        var = self.variable
        if not callable(operation):
            operation = getattr(transformations, operation)
        if groupby is not None:
            groups = self.data_source[groupby]
            self.data = self.data.groupby(groups)[var].apply(operation, *args, **kwargs)
        else:
            self.data = operation(self.data[var], *args, **kwargs)
        self.operations.append(operation.__name__)


class ModelResults(object):
    pass
