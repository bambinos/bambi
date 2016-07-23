import pandas as pd
import numpy as np
# import xarray as xr
from six import string_types
from collections import OrderedDict
import bambi.transformations as tr
from bambi.utils import listify, transformer
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
        ''' Reset instance attributes to initial state. '''
        self.model = None
        # self.cache = OrderedDict()
        self.contrasts = OrderedDict()
        self.terms = OrderedDict()
        self.y = None

    def build(self):
        ''' Build the PyMC3 model. '''
        pass

    # def run(self):
    #     ''' Run the MCMC sampler. '''
    #     if self.model is None:
    #         self.build()

    def set_y(self, label):
        ''' Set the outcome variable. '''
        if self.y is not None:
            self.terms[self.y.label] = self.y
            self.y = self.terms.pop(label)

    def add_term(self, variable, data=None, label=None,
                 categorical=False, random=False, split_by=None,
                 transformations=None, drop_first=False):
        ''' Create a new Term and add it to the current Model. All positional
        and keyword arguments are passed directly to the Term initializer. '''

        if data is None:
            data = self.data

        # Extract splitting variable
        if split_by is not None:
            if split_by in self.terms:
                split_by = self.terms[split_by]
            else:
                split_by = Term(split_by, self.data, categorical=True)
            # split_by = self.get_cached_term(split_by, True).values

        term = Term(variable, data, label, categorical, random, split_by,
                    transformations, drop_first)
        # self.cache[term.hash] = term
        self.terms[term.label] = term

    # def add_contrast(self, *args, **kwargs):
    #     pass

    # def get_cached_term(self, variable, categorical):
    #     ''' Retrieve a Term from the cache based on variable name and
    #     categorical status. '''
    #     key = hash((tuple(listify(variable)), categorical))
    #     if key not in self.cache:
    #         self.cache[key] = Term(variable, self.data, categorical=categorical)
    #     return self.cache[key]

    def transform(self, terms, transformations, groupby=None, *args, **kwargs):
        ''' Apply one or more data transformations to one or more Terms.
        Args:
            terms (str, list): The label(s) of one or more Terms to transform.
            transformations (callable, str, list): The transformations to apply
                to the Terms. Can be a str (the name of a predefined method
                found in the transformations module, e.g., 'scale'), a callable
                that accepts an array as its first argument and returns another
                array, or a list of strings or callables.
            groupby (list): A list of variables to group the transformation(s)
                by. For example, passing transformations='scale',
                groupby=['subject'] would apply the scaling transformation
                separately to each subject.
            args, kwargs: Optional positional and keyword arguments passed
                to the transformation method.
        '''
        for term in listify(terms):
            for trans in listify(transformations):
                self.terms[term].transform(trans, groupby, *args, **kwargs)


class Term(object):

    def __init__(self, variable, data, label=None,
                 categorical=False, random=False, split_by=None,
                 transformations=None, drop_first=False, **kwargs):
        '''
        Args:
            variable (str): The name of the DataFrame column that contains the
                data to use for the Term.
            data (DataFrame): The pandas DataFrame from which to draw data.
            label (str): Optional name of the Term. If None, the variable name
                is reused.
            categorical (bool): If True, the source variable is interpreted as
                nominal/categorical. If False, the source variable is treated
                as continuous.
            random (bool): If True, the Term is added to the model as a random
                effect; if False, treated as a fixed effect.
            split_by (Term): a Term instance to split on.
                split the named variable on. Use to specify nesting/crossing
                structures.
            transformations (list): List of transformations to apply to the
                data as soon as the Term is initialized.
            drop_first (bool): If True, uses n - 1 coding for categorical
                variables (i.e., a variable with 8 levels is coded using 7
                dummy variables, where each one is implicitly contrasted
                against the omitted first level). If False, uses n dummies to
                code n levels. Ignored if categorical = False.
            kwargs: Optional keyword arguments passed to the model-building
                back-end.
        '''
        self.variable = listify(variable)
        self.label = label or '_'.join(self.variable)
        self.transformations = []
        self.categorical = categorical
        self.random = random
        self.split_by = split_by
        self.data_source = data
        self.drop_first = drop_first
        self.levels = None
        self.hash = hash((tuple(self.variable), categorical))
        self.kwargs = kwargs

        # Load data
        self._setup()

        if transformations is not None:
            for t in listify(transformations):
                self.transform(t)

        self.values = self.data

        if split_by is not None:
            self.values = np.einsum('ab,ac->abc', self.values, self.split_by.values)

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
            self.levels = mapping
            recoded = data.loc[:, self.variable].replace(mapping).values
            data = pd.get_dummies(recoded, drop_first=self.drop_first)

        else:
            if  len(self.variable) > 1:
                raise ValueError("Adding a list of terms is only "
                        "supported for categorical variables "
                        "(e.g., random factors).")
            data = data.convert_objects(convert_numeric=True)

        self.data = data

    def transform(self, transformation, groupby=None, *args, **kwargs):
        ''' Apply an arbitrary transformation to the Term's data.
        Args:
            transformation (str, callable): The transformation to apply. Either
                the name of a predefined method in the transformations module
                (e.g., 'scale'), or a callable.
            groupby (str, list): Optional list of variables to group the
                transformation by.
            args, kwargs: Optional positional and keyword arguments to pass
                onto the transformation callable.
        '''
        if not callable(transformation):
            transformation = getattr(tr, transformation)
        if groupby is not None:
            groups = self.data_source[groupby]
            self.data = self.data.groupby(groups).apply(transformation, *args, **kwargs)
        else:
            self.data = transformation(self.data, *args, **kwargs)
        self.transformations.append(transformation.__name__)


# class ModelResults(object):
#     pass
