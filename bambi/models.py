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


class Model(object):

    def __init__(self, data=None, intercept=False, backend='pymc3',
                 default_priors=None):
        '''
        Args:
            data (DataFrame, str): the dataset to use. Either a pandas
                DataFrame, or the name of the file containing the data, which
                will be passed to pd.read_table().
            intercept (bool): If True, an intercept term is added to the model
                at initialization. Defaults to False, as both fixed and random
                effect specifications will add an intercept by default.
            backend (str): The name of the BackEnd to use. Currently only
                'pymc3' is supported.
            default_priors (dict, str): An optional specification of the
                default priors to use for all model terms. Either a dict
                containing named distributions, families, and terms (see the
                documentation in priors.PriorFactory for details), or the name
                of a JSON file containing the same information.
        '''

        if isinstance(data, string_types):
            data = pd.read_table(data, sep=None)

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
        '''
        Reset list of terms and y-variable.
        '''
        self.terms = OrderedDict()
        self.y = None

    def build(self):
        ''' Set up the model for sampling/fitting. Performs any steps that
        require access to all model terms (e.g., scaling priors on each term),
        then calls the BackEnd's build() method.
        '''
        if self.y is None:
            raise ValueError("No outcome (y) variable is set! Please call "
                             "add_y() or specify an outcome variable using the"
                             " formula interface before build() or fit().")

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

    def fit(self, fixed=None, random=None, priors=None, family='gaussian',
            link=None, run=True, categorical=None, **kwargs):
        '''
        Fit the model using the current BackEnd.
        Args:
            fixed (str): Optional formula specification of fixed effects.
            random (list): Optional list-based specification of random effects.
            priors (dict): Optional specification of priors for one or more
                terms. A dict where the keys are the names of terms in the
                model, and the values are either instances of class Prior or
                ints or floats that specify the width of the priors on a
                standardized scale.
            family (str, Family): A specification of the model family
                (analogous to the family object in R). Either a string, or an
                instance of class priors.Family. If a string is passed, a
                family with the corresponding name must be defined in the
                defaults loaded at Model initialization. Valid pre-defined
                families are 'gaussian', 'binomial', 'poisson', and 't'.
            link (str): The model link function to use. Can be either a string
                (must be one of the options defined in the current backend;
                typically this will include at least 'identity', 'logit',
                'inverse', and 'exp'), or a callable that takes a 1D ndarray
                or theano tensor as the sole argument and returns one with
                the same shape.
            run (bool): Whether or not to immediately begin fitting the model
                once any set up of passed arguments is complete.
            categorical (str, list): The names of any variables to treat as
                categorical. Can be either a single variable name, or a list
                of names. If categorical is None, the data type of the columns
                in the DataFrame will be used to infer handling. In cases where
                numeric columns are to be treated as categoricals (e.g., random
                factors coded as numerical IDs), explicitly passing variable
                names via this argument is recommended.
        '''
        if fixed is not None or random is not None:
            self.add_formula(fixed=fixed, random=random, priors=priors,
                             family=family, link=link, categorical=categorical,
                             append=False)
        ''' Run the BackEnd to fit the model. '''
        if not self.built:
            warnings.warn("Current Bayesian model has not been built yet; "
              "building it first before sampling begins.")
            self.build()
        if run:
            return self.backend.run(**kwargs)

    def add_intercept(self):
        '''
        Adds a constant term to the model. Generally unnecessary when using the
        formula interface, but useful when specifying the model via add_term().
        '''
        n = len(self.data)
        df = pd.DataFrame(np.ones((n, 1)), columns=['Intercept'])
        self.add_term('Intercept', df)

    def add_formula(self, fixed=None, random=None, priors=None,
                    family='gaussian', link=None, categorical=None,
                    append=False):
        '''
        Adds one or more terms to the model via an R-like formula syntax.
        Args:
            fixed (str): Optional formula specification of fixed effects.
            random (list): Optional list-based specification of random effects.
            priors (dict): Optional specification of priors for one or more
                terms. A dict where the keys are the names of terms in the
                model, and the values are either instances of class Prior or
                ints or floats that specify the width of the priors on a
                standardized scale.
            family (str, Family): A specification of the model family
                (analogous to the family object in R). Either a string, or an
                instance of class priors.Family. If a string is passed, a
                family with the corresponding name must be defined in the
                defaults loaded at Model initialization. Valid pre-defined
                families are 'gaussian', 'binomial', 'poisson', and 't'.
            link (str): The model link function to use. Can be either a string
                (must be one of the options defined in the current backend;
                typically this will include at least 'identity', 'logit',
                'inverse', and 'exp'), or a callable that takes a 1D ndarray
                or theano tensor as the sole argument and returns one with
                the same shape.
            categorical (str, list): The names of any variables to treat as
                categorical. Can be either a single variable name, or a list
                of names. If categorical is None, the data type of the columns
                in the DataFrame will be used to infer handling. In cases where
                numeric columns are to be treated as categoricals (e.g., random
                factors coded as numerical IDs), explicitly passing variable
                names via this argument is recommended.
            append (bool): if True, terms are appended to the existing model
                rather than replacing any existing terms. This allows
                formula-based specification of the model in stages.
        '''
        data = self.data

        if priors is None:
            priors = {}

        if not append:
            self.reset()

        if fixed is not None:
            # Explicitly convert columns to category if desired--though this
            # can also be done within the formula using C().
            if categorical is not None:
                data = data.copy()
                cats = listify(categorical)
                data[cats] = data[cats].apply(lambda x: x.astype('category'))

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
                prior = priors.pop(_name, priors.pop('fixed', None))
                self.add_term(_name, data=term_data, prior=prior)

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
                    variable = pred

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

                prior = priors.pop(label, priors.pop('random', None))
                self.add_term(variable=variable, label=label, **kwargs)

    def add_y(self, variable, prior=None, family='gaussian', link=None, *args,
              **kwargs):
        '''
        Add a dependent (or outcome) variable to the model.
        Args:
            variable (str): the name of the dataset column containing the
                y values.
            prior (Prior, int, float): Optional specification of the prior.
                Can be either an instance of priors.Prior, or a numeric value.
                In the latter case, the distribution specified in the defaults
                will be used, and the passed value will be used to scale the
                appropriate variance parameter.
            family (str, Family): A specification of the model family
                (analogous to the family object in R). Either a string, or an
                instance of class priors.Family. If a string is passed, a
                family with the corresponding name must be defined in the
                defaults loaded at Model initialization. Valid pre-defined
                families are 'gaussian', 'binomial', 'poisson', and 't'.
            link (str): The model link function to use. Can be either a string
                (must be one of the options defined in the current backend;
                typically this will include at least 'identity', 'logit',
                'inverse', and 'exp'), or a callable that takes a 1D ndarray
                or theano tensor as the sole argument and returns one with
                the same shape.
            args, kwargs: Optional positional and keyword arguments to pass
                onto add_term().
        '''
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
        # use last-added term name b/c it could have been changed by add_term
        name = list(self.terms.values())[-1].name
        self.y = self.terms.pop(name)
        self.built = False

    def add_term(self, variable, data=None, label=None, categorical=False,
                 random=False, split_by=None, prior=None, drop_first=True):
        '''
        Add a term to the model.
        Args:
            variable (str): The name of the dataset column to use; also used
                as the Term instance label if not otherwise specified using
                the label argument.
            data (DataFrame): Optional pandas DataFrame containing the term
                values to use. If None (default), the correct column will be
                extracted from the dataset currently loaded into the model
                (based on the name passed in the variable argument).
            label (str): Optional label/name to use for the term. If None, the
                label will be automatically generated based on the variable
                name and additional arguments.
            categorical (bool): Whether or not the input variable should be
                treated as categorical (defaults to False).
            random (bool): If True, the predictor variable is modeled as a
                random effect; if False, the predictor is modeled as a fixed
                effect.
            split_by (str): An optional name of another dataset column to
                "split" the target variable on. In practice, this is primarily
                used to specify the grouping variable when adding random
                slopes or intercepts. For example, if variable='subject',
                categorical=True, random=True, and split_by='condition',
                a separate set of random subject slopes will be added for each
                level of the condition variable. In this case, this would be
                roughly analogous to an lme4-style specification like
                'condition|subject'.
            prior (Prior, int, float): Optional specification of the prior.
                Can be either an instance of priors.Prior, or a numeric value.
                In the latter case, the distribution specified in the defaults
                will be used, and the passed value will be used to scale the
                appropriate variance parameter.
            drop_first (bool): indicates whether to use full rank or N-1 coding
                when the predictor is categorical. If True, the N levels of the
                categorical variable will be represented using N dummy columns.
                If False, the predictor will be represented using N-1 binary
                indicators, where each indicator codes the contrast between
                the N_j and N_0 columns, for j = {1..N-1}.

        Notes: One can think of bambi's split_by operation as a sequence of two
            steps. First, the target variable is multiplied by the splitting
            variable. This is equivalent to a formula call like 'A:B'. Second,
            the columns of the resulting matrix are "grouped" by the levels
            of the split_by variable.
        '''

        if data is None:
            data = self.data.copy()

        # Make sure user didn't forget to set categorical=True
        elif variable in data.columns and \
             data.loc[:, variable].dtype.name in ['object', 'category']:
             categorical = True

        if categorical:
            data[variable] = data[variable].astype('category')

        if split_by is not None:
            # Extract splitting variable. We do the dummy-coding of the
            # grouping variable in pandas rather than patsy because there's
            # no easy way to get the desired coding (reduced-rank for the
            # grouping variable, but full-rank for the predictor) in patsy
            # without using custom contrast schemes and totally screwing up
            # the variable naming scheme.
            grps = pd.get_dummies(data[split_by], drop_first=drop_first)
            data = {split_by: grps.values, variable: data[variable].values}
            f = '0 + %s:%s' % (variable, split_by)
            data = dmatrix(f, data=data)
            cols = data.design_info.column_names
            data = pd.DataFrame(data, columns=cols)

            # For categorical effects, one variance term per split_by level
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

    @property
    def term_names(self):
        return list(self.terms.keys())


class Term(object):

    type_ = 'fixed'

    def __init__(self, name, data, categorical=False, prior=None):
        '''
        Args:
            name (str): Name of the term.
            data (DataFrame, Series, ndarray): The term values.
            categorical (bool): If True, the source variable is interpreted as
                nominal/categorical. If False, the source variable is treated
                as continuous.
            prior (Prior): A specification of the prior(s) to use. An instance
                of class priors.Prior.
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

