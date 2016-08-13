import pandas as pd
import numpy as np
from six import string_types
from collections import OrderedDict, defaultdict
from bambi.utils import listify
from patsy import dmatrices, dmatrix
import warnings
from bambi.priors import default_priors
import re


def listify(obj):
    return obj if isinstance(obj, (list, tuple)) else [obj]


class Model(object):

    def __init__(self, data=None, intercept=True, backend='pymc3'):
        '''
        Args:
            dataset (DataFrame): the pandas DF containing the data to use.
        '''

        obj_cols = data.select_dtypes(['object']).columns
        data[obj_cols] = data[obj_cols].apply(lambda x: x.astype('category'))
        self.data = data
        # Some random effects stuff later requires us to make guesses about
        # column groupings into terms based on patsy's naming scheme.
        if re.search("[\[\]]+", ''.join(data.columns)):
            warnings.warn("At least one of the column names in the specified "
                          "dataset contain square brackets ('[' or ']')."
                          "This may cause unexpected behavior if you specify "
                          "models with random effects. You are encouraged to "
                          "rename your columns to avoid square brackets.")

        self.reset()

        if backend.lower() == 'pymc3':
            from bambi.backends import PyMC3BackEnd
            self.backend = PyMC3BackEnd()
        else:
            raise ValueError("At the moment, only the PyMC3 backend is supported.")

        if intercept:
            self.add_intercept()

    def reset(self):
        # self.cache = OrderedDict()
        self.contrasts = OrderedDict()
        self.terms = OrderedDict()
        self.y = None

    def build(self):
        ''' Build the PyMC3 model. '''
        if self.y is None:
            raise ValueError("No outcome (y) variable is set! Please call "
                             "set_y() before build() or fit().")
        self.backend.build(self)
        self.built = True

    def fit(self, fixed=None, random=None, **kwargs):
        if fixed is not None:
            self.add_formula(fixed, random=random, append=False)
        ''' Run the BackEnd to fit the model. '''
        if not self.built:
            warnings.warn("Current Bayesian model has not been built yet; "
              "building it first before sampling begins.")
            self.build()
        return self.backend.run(self, **kwargs)

    def add_intercept(self):
        n = len(self.data)
        df = pd.DataFrame(np.ones((n, 1)), columns=['Intercept'])
        self.add_term('Intercept', df)

    def add_formula(self, f, random=None, append=False, priors=None,
                    categorical=None):

        data = self.data

        if not append:
            self.reset()

        # Explicitly convert columns to category if desired--though this can
        # also be done within the formula using C().
        if categorical is not None:
            data = data.copy()
            categorical = listify(categorical)
            data[categorical] = data[categorical].apply(lambda x: x.astype('category'))

        if '~' in f:
            y, X = dmatrices(f, data=data)
            y_label = y.design_info.term_names[0]
            self.add_y(y_label)
        else:
            X = dmatrix(f, data=data)

        # Loop over predictor terms
        for _name, _slice in X.design_info.term_name_slices.items():
            cols = X.design_info.column_names[_slice]
            term_data = pd.DataFrame(X[:, _slice], columns=cols)
            self.add_term(_name, data=term_data)

        # Random effects
        if random is not None:
            random = listify(random)
            for f in random:
                kwargs = {'random': True}
                # '1|factor' is considered identical to 'factor'
                f = re.sub(r'^1\s+\|(.*)', r'\1', f).strip()
                if '|' not in f:
                    kwargs['categorical'] = True
                    variable = f
                else:
                    variable, split_by = re.split('\s*\|\s*', f)
                    kwargs['split_by'] = split_by
                self.add_term(variable=variable, label=f, **kwargs)

    def add_y(self, *args, **kwargs):
        self.add_term(*args, **kwargs)
        name = list(self.terms.values())[-1].name # last-added term
        self.set_y(name)

    def add_term(self, variable, data=None, label=None, categorical=False,
                 random=False, split_by=None, prior=None):
        ''' Create a new Term and add it to the current Model. All positional
        and keyword arguments are passed directly to the Term initializer. '''

        if data is None:
            data = self.data.copy()

        if categorical:
            data[variable] = data[variable].astype('category')

        # Extract splitting variable
        if split_by is not None:
            data[split_by] = data[split_by].astype('category')
            split_by = listify(split_by)
            group_term = ':'.join(split_by)
            f = '0 + %s : (%s)' % (variable, group_term)
            data = dmatrix(f, data=data)
            cols = data.design_info.column_names
            data = pd.DataFrame(data, columns=cols)

            # For categorical random effects, one variance term per split_by level
            if random and categorical:
                # if group_term not in self.terms:
                #     raise ValueError("The variable '%s' cannot be nested in or"
                #                  " crossed with '%s', because the latter does "
                #                  "not exist yet. Please make sure that you "
                #                  "explicitly add all terms to the model before"
                #                  " crossing or nesting with other terms." %
                #                  (variable, split_by))
                split_data = {}
                groups = list(set([re.sub(r'^.*?\:', '', c) for c in cols]))
                for g in groups:
                    patt = re.escape(r':%s' % g) + '$'
                    level_data = data.filter(regex=patt)
                    level_data.columns = [c.split(':')[0] for c in level_data.columns]
                    level_data = level_data.loc[:, (level_data!=0).any(axis=0)]
                    split_data[g] = level_data.values
                data = split_data

        elif categorical or (variable in data.columns and \
                             data[variable].dtype.name in ['object', 'category']):
            data = pd.get_dummies(data[variable])
        else:
            # If all columns have identical names except for levels in [],
            # assume they've already been contrast-coded, and pass data as-is
            cols = [re.sub('\[.*?\]', '', c) for c in data.columns]
            if len(set(cols)) > 1:
                data = data[[variable]]

        if label is None:
            label = variable

        if random:
            term = RandomTerm(label, data, categorical=categorical)
        else:
            term = Term(label, data, categorical=categorical)
        self.terms[term.name] = term
        self.built = False

    def set_y(self, label):
        ''' Set the outcome variable. '''
        if self.y is not None:
            self.terms[self.y.label] = self.y
        self.y = self.terms.pop(label)
        self.built = False


class Term(object):

    type_ = 'fixed'

    def __init__(self, name, data, categorical=False, prior=None, **kwargs):
        '''
        Args:
            name (str): Name of the term.
            data (DataFrame, ndarray): The pandas DataFrame or numpy array from
                containing the data. If a DF is passed, the variable names
                are used to extract the target columns. If a numpy array is
                passed, all columns of the array are used, without selection.
            categorical (bool): If True, the source variable is interpreted as
                nominal/categorical. If False, the source variable is treated
                as continuous.
            prior (dict): A specification of the prior(s) to use.
                Must have keys for 'name' and 'args'; optionally, can also
                pass 'sigma', which is another dict with name/arg keys.
            kwargs: Optional keyword arguments passed to the model-building
                back-end.
        '''
        self.name = name
        self.categorical = categorical
        self.prior = prior
        if isinstance(data, pd.DataFrame):
            self.levels = list(data.columns)
            data = data.values
        elif isinstance(data, dict):
            pass
        else:
            data = np.atleast_2d(data)
            self.levels = list(range(data.shape[1]))
        self.data = data
        self.kwargs = kwargs

        # TODO: come up with a more sensible way of getting/setting default priors
        if self.prior is None:
            if self.name == 'Intercept':
                self.prior = default_priors['intercept']
            else:
                self.prior = default_priors['fixed']


class RandomTerm(Term):

    type_ = 'random'

    def __init__(self, data, name, yoke=False, prior=None, **kwargs):
        self.yoke = yoke
        if prior is None:
            prior = default_priors['random']
        super(RandomTerm, self).__init__(data, name, prior=prior, **kwargs)

