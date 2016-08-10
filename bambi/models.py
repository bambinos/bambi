import pandas as pd
import numpy as np
from six import string_types
from collections import OrderedDict
from bambi.utils import listify
from patsy import dmatrices
import warnings


class Model(object):

    def __init__(self, data=None, intercept=True, backend='pymc3'):
        '''
        Args:
            dataset (DataFrame): the pandas DF containing the data to use.
        '''
        self.data = data
        # self.intercept = intercept
        self.reset()

        # if intercept:
        #     if 'intercept' not in self.data.columns:
        #         self.data['intercept'] = 1
        #     elif self.data['Intercept'].nunique() > 1:
        #         warnings.warn("The input dataset contains an existing column named"
        #                       " 'intercept' that has more than one unique value. "
        #                       "Note that this may cause unexpected behavior if "
        #                       "intercepts are added to the model via add_term() "
        #                       "calls.")

        #     self.add_term('intercept', self.data)

        if backend.lower() == 'pymc3':
            from bambi.backends import PyMC3BackEnd
            self.backend = PyMC3BackEnd()
        else:
            raise ValueError("At the moment, only the PyMC3 backend is supported.")

    def reset(self):
        # self.cache = OrderedDict()
        self.contrasts = OrderedDict()
        self.terms = OrderedDict()
        self.y = None

    def build(self, y=None):
        ''' Build the PyMC3 model. '''
        if y is not None:
            self.set_y(y)
        elif self.y is None:
            raise ValueError("No outcome (y) variable is set! Please call "
                             "set_y() before build() or fit().")
        self.backend.build(self)
        self.built = True

    def fit(self, formula=None, y=None, random=None, **kwargs):
        ''' Run the BackEnd to fit the model. '''
        if not self.built:
            warnings.warn("Current Bayesian model has not been built yet; "
              "building it first before sampling begins.")
            self.build(y)
        self.backend.run(**kwargs)

    def set_y(self, label):
        ''' Set the outcome variable. '''
        if self.y is not None:
            self.terms[self.y.label] = self.y
        self.y = self.terms.pop(label)
        self.built = False

    def add_term(self, variable, data=None, label=None,
                 categorical=False, random=False, split_by=None,
                 drop_first=False, prior=None):
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
                    drop_first)
        # self.cache[term.hash] = term
        self.terms[term.label] = term
        self.built = False


class Term(object):

    def __init__(self, variable, data, label=None, categorical=False,
                 random=False, split_by=None, drop_first=False, prior=None,
                 **kwargs):
        '''
        Args:
            variable (str): The name of the DataFrame column that contains the
                data to use for the Term.
            data (DataFrame, ndarray): The pandas DataFrame or numpy array from
                containing the data. If a DF is passed, the variable names
                are used to extract the target columns. If a numpy array is
                passed, all columns of the array are used, without selection.
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
            drop_first (bool): If True, uses n - 1 coding for categorical
                variables (i.e., a variable with 8 levels is coded using 7
                dummy variables, where each one is implicitly contrasted
                against the omitted first level). If False, uses n dummies to
                code n levels. Ignored if categorical = False.
            prior (dict): A specification of the prior(s) to use.
                Must have keys for 'name' and 'args'; optionally, can also
                pass 'sigma', which is another dict with name/arg keys.
            kwargs: Optional keyword arguments passed to the model-building
                back-end.
        '''
        self.variable = listify(variable)
        self.label = label or '_'.join(self.variable)
        self.categorical = categorical
        self.random = random
        self.split_by = split_by
        self.data_source = data
        self.drop_first = drop_first
        self.prior = prior
        self.levels = None
        # self.hash = hash((tuple(self.variable), categorical))
        self.kwargs = kwargs

        # Load data
        self._setup()

    def _setup(self):

        data = self.data_source

        # for DFs, we do additional processing. if we get anything else, we
        # assume the user wants the values modeled as-is.
        if isinstance(data, pd.DataFrame):
            data = data[self.variable].copy()

            if self.categorical:
                # Handle multiple variables; will fail gracefully if only 1 exists
                try:
                    data = data.stack()
                except: pass
                n_cols = data.nunique()
                levels = data.unique()
                mapping = OrderedDict(zip(levels, list(range(n_cols))))
                self.levels = levels
                recoded = data.loc[:, self.variable].replace(mapping).values
                data = pd.get_dummies(recoded, drop_first=self.drop_first)

            else:
                if  len(self.variable) > 1:
                    raise ValueError("Adding a list of terms is only "
                            "supported for categorical variables "
                            "(e.g., random factors).")
                data = data.convert_objects(convert_numeric=True)

        self.data = data

        self.values = self.data.values

        if self.split_by is not None:
            self.values = np.einsum('ab,ac->abc', self.values, self.split_by.values)

        # TODO: come up with a more sensible way of getting/setting default priors
        if self.prior is None:
            from bambi.priors import default_priors
            if self.label == 'Intercept':
                self.prior = default_priors['Intercept']
            elif self.random:
                self.prior = default_priors['random']
            else:
                self.prior = default_priors['fixed']
