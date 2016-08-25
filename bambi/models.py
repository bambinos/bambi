import pandas as pd
import numpy as np
from bambi.external.six import string_types
from collections import OrderedDict, defaultdict
from bambi.utils import listify
from patsy import dmatrices, dmatrix
import statsmodels.api as sm
import warnings
from bambi.priors import PriorFactory
from copy import deepcopy
import re


def listify(obj):
    return obj if isinstance(obj, (list, tuple)) else [obj]


class Model(object):

    def __init__(self, data=None, intercept=False, backend='pymc3',
                 default_priors=None):
        '''
        Args:
            dataset (DataFrame): the pandas DF containing the data to use.
        '''

        self.default_priors = PriorFactory(default_priors)

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

        # compute and store information used to set the default priors
        # X = fixed effects design matrix
        # R2X = 1 - 1/VIF for each x, i.e., R2 for predicting each x from all other x's
        # R2Y = R2 for predicting y from all x's *other than* the current x
        X = pd.concat([pd.DataFrame(x.data, columns=x.levels) for x in self.terms.values()
            if x.type_=='fixed' and x.name != 'Intercept'], axis=1)
        self.R2X = pd.Series({x:sm.OLS(X[x], sm.add_constant(X.drop(x, axis=1))).fit().rsquared
            for x in list(X.columns)})
        self.R2Y = pd.Series({x:sm.OLS(self.y.data, sm.add_constant(X.drop(x, axis=1))).fit().rsquared
            for x in list(X.columns)})
        self.SDX = X.std()
        self.SDY = self.y.data.std()
        self.meanX = X.mean(0)

        for t in self.terms.values():
            t._setup() 
        self.backend.build(self)
        self.built = True

    def fit(self, fixed=None, random=None, family='gaussian', link=None,
            run=True, categorical=None, **kwargs):
        if fixed is not None or random is not None:
            self.add_formula(fixed=fixed, random=random, append=False,
                             family=family, link=link, categorical=categorical)
        ''' Run the BackEnd to fit the model. '''
        if not self.built:
            warnings.warn("Current Bayesian model has not been built yet; "
              "building it first before sampling begins.")
            self.build()
        if run:
            return self.backend.run(self, **kwargs)

    def add_intercept(self):
        n = len(self.data)
        df = pd.DataFrame(np.ones((n, 1)), columns=['Intercept'])
        self.add_term('Intercept', df)

    def add_formula(self, fixed=None, random=None, append=False, priors=None,
                    categorical=None, family='gaussian', link=None):

        data = self.data

        if not append:
            self.reset()

        if fixed is not None:
            # Explicitly convert columns to category if desired--though this can
            # also be done within the formula using C().
            if categorical is not None:
                data = data.copy()
                categorical = listify(categorical)
                data[categorical] = data[categorical].apply(lambda x: x.astype('category'))

            if '~' in fixed:
                y, X = dmatrices(fixed, data=data)
                y_label = y.design_info.term_names[0]
                self.add_y(y_label, family=family, link=link)
            else:
                X = dmatrix(fixed, data=data)

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
                if re.search('[\*\(\)\+\-]+', f):
                    raise ValueError("Random term '%s' contains an invalid "
                        "character. Note that formula-style operators other "
                        "than | are not currently supported in random effects "
                        "specifications.")
                # '1|factor' is considered identical to 'factor'
                f = re.sub(r'^1\s*\|(.*)', r'\1', f).strip()
                if '|' not in f:
                    kwargs['categorical'] = True
                    kwargs['drop_first'] = False
                    variable = f
                else:
                    variable, split_by = re.split('\s*\|\s*', f)
                    kwargs['split_by'] = split_by
                self.add_term(variable=variable, label=f, **kwargs)

    def add_y(self, variable, family='gaussian', link=None, *args,
              **kwargs):

        if isinstance(family, string_types):
            family = self.default_priors.get(family=family)
        self.family = family

        # Override family's link if another is explicitly passed
        if link is not None:
            self.family.link = link

        prior = self.family.prior

        # implement default HalfCauchy prior for normal sigma (beta = sd(Y))
        if self.family.name=='gaussian':
            prior.args['sd'].update(beta=self.data[variable].std())

        self.add_term(variable, prior=prior, *args, **kwargs)
        # use last-added term name b/c it could have been changed in add_term
        name = list(self.terms.values())[-1].name
        self.set_y(name)

    def add_term(self, variable, data=None, label=None, categorical=False,
                 random=False, split_by=None, prior=None, drop_first=True):
        ''' Create a new Term and add it to the current Model. All positional
        and keyword arguments are passed directly to the Term initializer. '''

        if data is None:
            data = self.data.copy()

        # Make sure user didn't forget to set categorical=True
        elif variable in data.columns and \
             data.loc[:, variable].dtype.name in ['object', 'category']:
             categorical = True

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
            data = pd.get_dummies(data[variable], drop_first=drop_first)
        else:
            # If all columns have identical names except for levels in [],
            # assume they've already been contrast-coded, and pass data as-is
            cols = [re.sub('\[.*?\]', '', c) for c in data.columns]
            if len(set(cols)) > 1:
                data = data[[variable]]

        if label is None:
            label = variable

        if random:
            term = RandomTerm(self, label, data, categorical=categorical,
                              prior=prior)
        else:
            term = Term(self, label, data, categorical=categorical,
                        prior=prior)
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

    def __init__(self, model, name, data, categorical=False, prior=None,
                 **kwargs):
        '''
        Args:
            model (Model): The associated Model instance.
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
        self.model = model
        self.name = name
        self.categorical = categorical
        self.prior = prior

        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            self.levels = list(data.columns)
            data = data.values
        elif isinstance(data, dict):
            pass   # Random effects pass through here
        else:
            data = np.atleast_2d(data)
            self.levels = list(range(data.shape[1]))

        self.data = data
        self.kwargs = kwargs

    def _setup(self):
        # set up default priors if no prior has been explicitly set
        if self.prior is None:
            term_type = 'intercept' if self.name == 'Intercept' else 'fixed'
            self.prior = self.model.default_priors.get(term=term_type)
            # these default priors are only defined for Normal priors, although we
            # could probably easily handle Cauchy by just substituting 'sd' -> 'beta'
            if self.prior.name=='Normal':
                # the default 'wide' prior SD is sqrt(1/3) = .577 on the partial corr scale,
                # which is the SD of a flat prior over [-1,1]. Wider than that would be weird
                # TODO: support other defaults such as superwide = .8, medium = .4, narrow = .2
                wide = 3**-.5
                # handle slopes
                if term_type=='fixed':
                    slope_constant = self.model.SDY * (1 - self.model.R2Y[self.levels]) / \
                                     self.model.SDX[self.levels] / (1 - self.model.R2X[self.levels])
                    self.prior.update(sd = wide * slope_constant.values)
                # handle the intercept
                else:
                    index = list(self.model.R2Y.index)
                    intercept_SD = self.model.SDY * (1 - self.model.R2Y[index]) / \
                        self.model.SDX[index] / (1 - self.model.R2X[index])
                    intercept_SD *= wide
                    intercept_SD = np.dot(intercept_SD**2, self.model.meanX[index]**2)**.5
                    self.prior.update(sd = intercept_SD)


class RandomTerm(Term):

    type_ = 'random'

    def __init__(self, model, name, data, yoke=False, prior=None, **kwargs):

        self.yoke = yoke
        super(RandomTerm, self).__init__(model, name, data, prior=prior, **kwargs)

    def _setup(self):
        # set up default priors if no prior has been explicitly set
        if self.prior is None:
            term_type = 'intercept' if '|' not in self.name else 'slope'
            self.prior = self.model.default_priors.get(term='random')
            # these default priors are only defined for HalfCauchy priors,
            if self.prior.args['sd'].name=='HalfCauchy':
                # as above, 'wide' prior SD is sqrt(1/3) = .577 on the partial corr scale,
                # which is the SD of a flat prior over [-1,1]. Wider than that would be weird
                # TODO: support other defaults such as superwide = .8, medium = .4, narrow = .2
                wide = 3**-.5
                # handle random slopes
                if term_type=='slope':
                    # get name of corresponding fixed effect
                    fix = re.sub(r'\|.*', r'', self.name).strip()
                    # only proceed if there does exist a corresponding fixed effect.
                    # note that without this, it would break on random slopes for
                    # categorical predictors! Here we simply skip that case, but we
                    # should make it correctly handle default priors for that case
                    if fix in list(self.model.R2Y.index):
                        slope_constant = self.model.SDY * (1 - self.model.R2Y[fix]) / \
                                         self.model.SDX[fix] / (1 - self.model.R2X[fix])
                        self.prior.args['sd'].update(beta = wide * slope_constant)
                # handle random intercepts
                else:
                    index = list(self.model.R2Y.index)
                    intercept_beta = self.model.SDY * (1 - self.model.R2Y[index]) / \
                        self.model.SDX[index] / (1 - self.model.R2X[index])
                    intercept_beta *= wide
                    intercept_beta = np.dot(intercept_beta**2, self.model.meanX[index]**2)**.5
                    self.prior.args['sd'].update(beta = intercept_beta)

