import pandas as pd
import numpy as np
from bambi.external.six import string_types
from collections import OrderedDict, defaultdict
from bambi.utils import listify
from patsy import dmatrices, dmatrix
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

        # compute information used to set the default priors
        # X = fixed effects design matrix (excluding intercept/constant term)
        # r2_x = 1 - 1/VIF for each x, i.e., R2 for predicting each x from all other x's
        # r2_y = R2 for predicting y from all x's *other than* the current x
        X = pd.concat([pd.DataFrame(x.data, columns=x.levels) for x in self.terms.values()
            if x.type_=='fixed' and x.name != 'Intercept'], axis=1)
        default_prior_info = {
            'r2_x':pd.Series({x:pd.stats.api.ols(y=X[x], x=X.drop(x, axis=1)).r2
                for x in list(X.columns)}),
            'r2_y':pd.Series({x:pd.stats.api.ols(y=self.y.data.squeeze(), x=X.drop(x, axis=1)).r2
                for x in list(X.columns)}),
            'sd_x':X.std(),
            'sd_y':self.y.data.std(),
            'mean_x':X.mean(axis=0),
            'mean_y':self.y.data.mean()
        }

        # save some info possibly useful for diagnostics, and send to ModelResults
        # mat = correlation matrix of X, w/ diagonal replaced by X means
        mat = X.corr()
        for x in list(mat.columns): mat.loc[x,x] = default_prior_info['mean_x'][x]
        self._diagnostics = {
            # the Variance Inflation Factors (VIF), which is possibly useful for diagnostics
            'VIF':1/(1 - default_prior_info['r2_x']),
            'corr_mean_X':mat
        }

        for t in self.terms.values():
            t._setup(model=self, default_prior_info=default_prior_info) 
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
                f = f.strip()
                kwargs = {'random': True}
                if re.search('[\*\(\)]+', f):
                    raise ValueError("Random term '%s' contains an invalid "
                        "character. Note that only the | and + operators are "
                        "currently supported in random effects specifications.")

                # Split specification into intercept, predictor, and grouper
                patt = r'^([01]+)*[\s\+]*([^\|]+)\|*(.*)'
                intcpt, pred, grpr = re.search(patt, f).groups()
                label = '{}|{}'.format(pred, grpr) if grpr else pred

                # Default to including random intercepts
                if intcpt is None:
                    intcpt = 1
                intcpt = int(intcpt)

                # If there's no grouper, we must be adding random intercepts
                if not grpr:
                    kwargs.update(dict(categorical=True, drop_first=False))

                else:
                    # Add random slopes unless they were explicitly excluded
                    if intcpt and grpr not in self.terms:
                        self.add_term(variable=grpr, categorical=True,
                                      random=True, drop_first=False)
                    # For categoricals, flip the predictor and grouper before
                    # passing to add_term(). This allows us to take advantage
                    # of the convenient split_by semantics.
                    if self.data[pred].dtype.name in ['object', 'category']:
                        variable, kwargs['split_by'] = grpr, pred
                        kwargs['categorical'] = True
                        if not intcpt:
                            kwargs['drop_first'] = False
                    else:
                        variable, kwargs['split_by'] = pred, grpr

                self.add_term(variable=variable, label=label, **kwargs)

    def add_y(self, variable, family='gaussian', link=None, prior=None, *args,
              **kwargs):

        if isinstance(family, string_types):
            family = self.default_priors.get(family=family)
        self.family = family

        # Override family's link if another is explicitly passed
        if link is not None:
            self.family.link = link

        if prior is None:
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
            term = RandomTerm(name=label, data=data, categorical=categorical,
                              prior=prior)
        else:
            term = Term(name=label, data=data, categorical=categorical,
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

    def __init__(self, name, data, categorical=False, prior=None,
                 **kwargs):
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

    def _setup(self, model, default_prior_info):
        # set up default priors if no prior has been explicitly set
        if self.prior is None:
            term_type = 'intercept' if self.name == 'Intercept' else 'fixed'
            self.prior = model.default_priors.get(term=term_type)
            # these default priors are only defined for Normal priors, although we
            # could probably easily handle Cauchy by just substituting 'sd' -> 'beta'
            if self.prior.name=='Normal':
                # the default 'wide' prior SD is sqrt(1/3) = .577 on the partial corr scale,
                # which is the SD of a flat prior over [-1,1]. Wider than that would be weird
                # TODO: support other defaults such as superwide = .8, medium = .4, narrow = .2
                wide = 3**-.5
                # handle slopes
                if term_type=='fixed':
                    slope_constant = default_prior_info['sd_y'] * (1 - default_prior_info['r2_y'][self.levels]) / \
                                     default_prior_info['sd_x'][self.levels] / (1 - default_prior_info['r2_x'][self.levels])
                    self.prior.update(sd = wide * slope_constant.values)
                # handle the intercept
                else:
                    index = list(default_prior_info['r2_y'].index)
                    intercept_SD = default_prior_info['sd_y'] * (1 - default_prior_info['r2_y'][index]) / \
                        default_prior_info['sd_x'][index] / (1 - default_prior_info['r2_x'][index])
                    intercept_SD *= wide
                    intercept_SD = np.dot(intercept_SD**2, default_prior_info['mean_x'][index]**2)**.5
                    self.prior.update(mu=default_prior_info['mean_y'], sd=intercept_SD)


class RandomTerm(Term):

    type_ = 'random'

    def __init__(self, name, data, yoke=False, prior=None, **kwargs):

        self.yoke = yoke
        super(RandomTerm, self).__init__(name, data, prior=prior, **kwargs)

    def _setup(self, model, default_prior_info):
        # set up default priors if no prior has been explicitly set
        if self.prior is None:
            term_type = 'intercept' if '|' not in self.name else 'slope'
            self.prior = model.default_priors.get(term='random')
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
                    if fix in list(default_prior_info['r2_y'].index):
                        slope_constant = default_prior_info['sd_y'] * (1 - default_prior_info['r2_y'][fix]) / \
                                         default_prior_info['sd_x'][fix] / (1 - default_prior_info['r2_x'][fix])
                        self.prior.args['sd'].update(beta = wide * slope_constant)
                # handle random intercepts
                else:
                    index = list(default_prior_info['r2_y'].index)
                    intercept_beta = default_prior_info['sd_y'] * (1 - default_prior_info['r2_y'][index]) / \
                        default_prior_info['sd_x'][index] / (1 - default_prior_info['r2_x'][index])
                    intercept_beta *= wide
                    intercept_beta = np.dot(intercept_beta**2, default_prior_info['mean_x'][index]**2)**.5
                    self.prior.args['sd'].update(beta = intercept_beta)

