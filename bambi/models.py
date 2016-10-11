import pandas as pd
import numpy as np
import pymc3 as pm
from pymc3.model import FreeRV
import matplotlib.pyplot as plt
from bambi.external.six import string_types
from collections import OrderedDict, defaultdict
from bambi.utils import listify
from patsy import dmatrices, dmatrix
import re, warnings
from bambi.priors import PriorFactory, PriorScaler, Prior
from copy import deepcopy


class Model(object):

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
        auto_scale (bool): If True (default), priors are automatically rescaled
            to the data (to be weakly informative) any time default priors are
            used. Note that any priors explicitly set by the user will always
            take precedence over default priors.
        dropna (bool): When True, rows with any missing values in either the
            predictors or outcome are automatically dropped from the dataset in
            a listwise manner.
    '''

    def __init__(self, data=None, intercept=False, backend='pymc3',
                 default_priors=None, auto_scale=True, dropna=False):

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
            raise ValueError(
                "At the moment, only the PyMC3 backend is supported.")

        if intercept:
            self.add_intercept()

        self.auto_scale = auto_scale
        self.dropna = dropna

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

        # Check for NaNs and halt if dropna is False--otherwise issue warning.
        arrs = []
        for t in self.terms.values():
            if isinstance(t.data, dict):
                arrs.extend(list(t.data.values()))
            else:
                arrs.append(t.data)
        X = np.concatenate(arrs + [self.y.data], axis=1)
        num_na = np.isnan(X).any(1).sum()
        if num_na:
            msg = "%d rows were found contain at least one missing value." % num_na
            if not self.dropna:
                msg += "Please make sure the dataset contains no missing " \
                       "values. Alternatively, if you want rows with missing " \
                       "values to be automatically deleted in a list-wise " \
                       "manner (not recommended), please set dropna=True at " \
                       "model initialization."
                raise ValueError(msg)
            msg += " Automatically removing %d rows from the dataset." % num_na
            warnings.warn(msg)

        # compute information used to set the default priors
        # X = fixed effects design matrix (excluding intercept/constant term)
        # r2_x = 1 - 1/VIF for each x, i.e., R2 for predicting each x from all
        # other x's r2_y = R2 for predicting y from all x's *other than* the
        # current x.
        # only compute these stats if there are multiple terms in the model
        terms = [t for t in self.fixed_terms.values() if t.name != 'Intercept']

        if len(self.fixed_terms) > 1:

            X = [pd.DataFrame(x.data, columns=x.levels) for x in terms]
            X = pd.concat(X, axis=1)

            # interim solution for handling non-normal models
            sd_y_defaults = {
                'gaussian': {
                    'identity': self.y.data.std(),
                    'logit': self.y.data.std(),
                    'probit': self.y.data.std(),
                    'inverse': self.y.data.std(),
                    'log': self.y.data.std()
                },
                'binomial': {
                    'identity': self.y.data.std(),
                    'logit': np.pi / 3**.5,
                    'probit': 1,
                    'inverse': self.y.data.std(),
                    'log': self.y.data.std()
                },
                'poisson': {
                    'identity': self.y.data.std(),
                    'logit': self.y.data.std(),
                    'probit': self.y.data.std(),
                    'inverse': self.y.data.std(),
                    'log': self.y.data.std()
                },
                't': {
                    'identity': self.y.data.std(),
                    'logit': self.y.data.std(),
                    'probit': self.y.data.std(),
                    'inverse': self.y.data.std(),
                    'log': self.y.data.std()
                }
            }

            self.dm_statistics = {
                'r2_x': pd.Series({
                    x: pd.stats.api.ols(
                        y=X[x], x=X.drop(x, axis=1),
                        intercept=True if 'Intercept' in self.term_names else False).r2
                    for x in list(X.columns)}),
                'r2_y': pd.Series({
                    x: pd.stats.api.ols(
                        y=self.y.data.squeeze(), x=X.drop(x, axis=1),
                        intercept=True if 'Intercept' in self.term_names else False).r2
                    for x in list(X.columns)}),
                'sd_x': X.std(),
                'sd_y': sd_y_defaults[self.family.name][self.family.link],
                'mean_x': X.mean(axis=0)
            }

            # save potentially useful info for diagnostics and send to ModelResults
            # mat = correlation matrix of X, w/ diagonal replaced by X means
            mat = X.corr()
            for x in list(mat.columns):
                mat.loc[x, x] = self.dm_statistics['mean_x'][x]
            self._diagnostics = {
                # the Variance Inflation Factors (VIF), which is possibly useful
                # for diagnostics
                'VIF': 1/(1 - self.dm_statistics['r2_x']),
                'corr_mean_X': mat
            }

            # throw informative error if there is perfect collinearity among the fixed effects
            if any(self.dm_statistics['r2_x'] > .999):
                raise ValueError("There is perfect collinearity among the fixed effects!\n" + \
                    "Printing some design matrix statistics:\n" + \
                    str(self.dm_statistics) + '\n' + \
                    str(self._diagnostics))

        # only set priors if there is at least one term in the model
        if len(self.terms) > 0:
            # Get and scale default priors if none are defined yet
            scaler = PriorScaler(self)
            for t in self.terms.values():
                if not isinstance(t.prior, Prior):
                    scaler.scale(t)

        # For binomial models with n_trials = 1 (most common use case),
        # tell user which event is being modeled
        if self.family.name=='binomial' and np.max(self.y.data) < 1.01:
            event = next(i for i,x in enumerate(self.y.data.flatten()) if x>.99)
            warnings.warn('Modeling the probability that {}==\'{}\''.format(
                self.y.name, str(self.data[self.y.name][event])))

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
                ints, floats, or strings that specify the width of the priors
                on a standardized scale.
            family (str, Family): A specification of the model family
                (analogous to the family object in R). Either a string, or an
                instance of class priors.Family. If a string is passed, a
                family with the corresponding name must be defined in the
                defaults loaded at Model initialization. Valid pre-defined
                families are 'gaussian', 'binomial', 'poisson', and 't'.
            link (str): The model link function to use. Can be either a string
                (must be one of the options defined in the current backend;
                typically this will include at least 'identity', 'logit',
                'inverse', and 'log'), or a callable that takes a 1D ndarray
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
        if run:
            if not self.built:
                warnings.warn("Current Bayesian model has not been built yet; "
                              "building it first before sampling begins.")
                self.build()
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
                    append=True):
        '''
        Adds one or more terms to the model via an R-like formula syntax.
        Args:
            fixed (str): Optional formula specification of fixed effects.
            random (list): Optional list-based specification of random effects.
            priors (dict): Optional specification of priors for one or more
                terms. A dict where the keys are the names of terms in the
                model, and the values are either instances of class Prior or
                ints, floats, or strings that specify the width of the priors
                on a standardized scale.
            family (str, Family): A specification of the model family
                (analogous to the family object in R). Either a string, or an
                instance of class priors.Family. If a string is passed, a
                family with the corresponding name must be defined in the
                defaults loaded at Model initialization. Valid pre-defined
                families are 'gaussian', 'binomial', 'poisson', and 't'.
            link (str): The model link function to use. Can be either a string
                (must be one of the options defined in the current backend;
                typically this will include at least 'identity', 'logit',
                'inverse', and 'log'), or a callable that takes a 1D ndarray
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
                # check to see if formula is using the 'y[event] ~ x' syntax
                # (for binomial models). If so, chop it into groups:
                # 1 = 'y[event]', 2 = 'y', 3 = 'event', 4 = 'x'
                # If this syntax is not being used, event = None
                event = re.match(r'^((\S+)\[(\S+)\])\s*~(.*)$', fixed)
                if event is not None:
                    fixed = '{}~{}'.format(event.group(2),event.group(4))
                y, X = dmatrices(fixed, data=data)
                y_label = y.design_info.term_names[0]
                if event is not None:
                    # pass in new Y data that has 1 if y=event and 0 otherwise
                    y_data = y[:,y.design_info.column_names.index(event.group(1))]
                    y_data = pd.DataFrame({event.group(3): y_data})
                    self.add_y(y_label, family=family, link=link, data=y_data)
                else:
                    # use Y as-is
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
                                     "character. Note that only the | and + "
                                     "operators are currently supported in "
                                     "random effects specifications.")

                # replace explicit intercept terms like '1|subj' with just 'subj'
                f = re.sub(r'^1\s*\|(.*)', r'\1', f).strip()

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
                    # If we're adding slopes, add random intercepts as well,
                    # unless they were explicitly excluded
                    if intcpt and grpr not in self.terms:
                        self.add_term(variable=grpr, categorical=True,
                                      random=True, drop_first=False)
                    if self.data[pred].dtype.name in ['object', 'category']:
                        kwargs['categorical'] = True
                        if not intcpt:
                            kwargs['drop_first'] = False
                    variable, kwargs['over'] = pred, grpr

                prior = priors.pop(label, priors.pop('random', None))
                self.add_term(variable=variable, label=label, **kwargs)

    def add_y(self, variable, prior=None, family='gaussian', link=None, *args,
              **kwargs):
        '''
        Add a dependent (or outcome) variable to the model.
        Args:
            variable (str): the name of the dataset column containing the
                y values.
            prior (Prior, int, float, str): Optional specification of prior.
                Can be an instance of class Prior, a numeric value, or a string
                describing the width. In the numeric case, the distribution
                specified in the defaults will be used, and the passed value
                will be used to scale the appropriate variance parameter. For
                strings (e.g., 'wide', 'narrow', 'medium', or 'superwide'),
                predefined values will be used.
            family (str, Family): A specification of the model family
                (analogous to the family object in R). Either a string, or an
                instance of class priors.Family. If a string is passed, a
                family with the corresponding name must be defined in the
                defaults loaded at Model initialization. Valid pre-defined
                families are 'gaussian', 'binomial', 'poisson', and 't'.
            link (str): The model link function to use. Can be either a string
                (must be one of the options defined in the current backend;
                typically this will include at least 'identity', 'logit',
                'inverse', and 'log'), or a callable that takes a 1D ndarray
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

        # implement default Uniform [0, sd(Y)] prior for residual SD
        if self.family.name == 'gaussian':
            prior.update(sd=Prior('Uniform', lower=0, upper=self.data[variable].std()))

        self.add_term(variable, prior=prior, *args, **kwargs)
        # use last-added term name b/c it could have been changed by add_term
        name = list(self.terms.values())[-1].name
        self.y = self.terms.pop(name)
        self.built = False

    def add_term(self, variable, data=None, label=None, categorical=False,
                 random=False, over=None, prior=None, drop_first=True):
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
            over (str): When adding random slopes, the name of the variable the
                slopes are randomly distributed over. For example, if
                variable='condition', categorical=True, random=True, and
                over='subject', a separate set of random subject slopes will be
                added for each level of the condition variable. This is
                analogous to the lme4 specification of 'condition|subject'.
            prior (Prior, int, float, str): Optional specification of prior.
                Can be an instance of class Prior, a numeric value, or a string
                describing the width. In the numeric case, the distribution
                specified in the defaults will be used, and the passed value
                will be used to scale the appropriate variance parameter. For
                strings (e.g., 'wide', 'narrow', 'medium', or 'superwide'),
                predefined values will be used.
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
        if variable in data.columns and \
                data.loc[:, variable].dtype.name in ['object', 'category']:
            categorical = True

        else:
            # If all columns have identical names except for levels in [],
            # assume they've already been contrast-coded, and pass data as-is
            cols = [re.sub('\[.*?\]', '', c) for c in data.columns]
            if len(set(cols)) > 1:
                X = data[[variable]]

        if categorical:
            X = pd.get_dummies(data[variable], drop_first=drop_first)
        elif variable in data.columns:
            X = data[[variable]]
        else:
            X = data

        if random and over is not None:
            id_var = pd.get_dummies(data[over], drop_first=False)
            data = {over: id_var.values, variable: X.values}
            f = '0 + %s:%s' % (over, variable)
            data = dmatrix(f, data=data)
            cols = data.design_info.column_names
            data = pd.DataFrame(data, columns=cols)

            # For categorical effects, one variance term per predictor level
            if categorical:
                split_data = {}
                groups = list(set([re.sub(r'^.*?\:', '', c) for c in cols]))
                for g in groups:
                    patt = re.escape(r':%s' % g) + '$'
                    level_data = data.filter(regex=patt)
                    level_data.columns = [
                        c.split(':')[0] for c in level_data.columns]
                    level_data = level_data.loc[
                        :, (level_data != 0).any(axis=0)]
                    split_data[g] = level_data.values
                data = split_data
        else:
            data = X

        if label is None:
            label = variable
            if over is not None:
                label += '|%s' % over

        term = Term(name=label, data=data, categorical=categorical,
                    random=random, prior=prior)
        self.terms[term.name] = term
        self.built = False

    def set_priors(self, priors=None, fixed=None, random=None):
        '''
        Set priors for one or more existing terms.
        Args:
            priors (dict): Dict of priors to update. Keys are names of terms
                to update; values are the new priors (either a Prior instance,
                or an int or float that scales the default priors). Note that
                a tuple can be passed as the key, in which case the same prior
                will be applied to all terms named in the tuple.
            fixed (Prior, int, float, str): a prior specification to apply to
                all fixed terms currently included in the model.
            random (Prior, int, float, str): a prior specification to apply to
                all random terms currently included in the model.
        '''

        targets = {}

        if fixed is not None:
            targets.update({name: fixed for name in self.fixed_terms.keys()})

        if random is not None:
            targets.update({name: random for name in self.random_terms.keys()})

        if priors is not None:
            for k, prior in priors.items():
                for name in listify(k):
                    if name not in self.terms:
                        raise ValueError("The model contains no term with "
                                         "the name '%s'." % name)
                    targets[name] = prior

        for name, prior in targets.items():
            self.terms[name].prior = prior

    def plot(self, kind='priors'):
        # Currently this only supports plotting priors for fixed effects
        if not self.built:
            raise ValueError("Cannot plot priors until model is built!")

        with pm.Model():
            # get priors for fixed fx, separately for each level of each predictor
            dists = []
            for t in self.fixed_terms.values():
                for i,l in enumerate(t.levels):
                    params = {k: v[i % len(v)] if isinstance(v, np.ndarray) else v
                        for k,v in t.prior.args.items()}
                    dists += [getattr(pm, t.prior.name)(l, **params)]

            # get priors for random effect SDs
            for t in self.random_terms.values():
                prior = t.prior.args['sd'].name
                params = t.prior.args['sd'].args
                dists += [getattr(pm, prior)(t.name+'_sd', **params)]

            # add priors on Y params if applicable
            y_prior = [(k,v) for k,v in self.y.prior.args.items()
                if isinstance(v, Prior)]
            if len(y_prior):
                for p in y_prior:
                    dists += [getattr(pm, p[1].name)('_'.join([self.y.name,
                        p[0]]), **p[1].args)]
            
            # make the plot!
            p = float(len(dists))
            fig, axes = plt.subplots(int(np.ceil(p/2)), 2,
                figsize=(12,np.ceil(p/2)*2))
            # in case there is only 1 row
            if int(np.ceil(p/2))<2: axes = axes[None,:]
            for i,d in enumerate(dists):
                dist = d.distribution if isinstance(d, FreeRV) else d
                samp = pd.Series(dist.random(size=1000).flatten())
                samp.plot(kind='hist', ax=axes[divmod(i,2)[0], divmod(i,2)[1]],
                    normed=True)
                samp.plot(kind='kde', ax=axes[divmod(i,2)[0], divmod(i,2)[1]],
                    color='b')
                axes[divmod(i,2)[0], divmod(i,2)[1]].set_title(d.name)
            fig.tight_layout()
        
        return axes

    @property
    def term_names(self):
        ''' Return names of all terms in order of addition to model. '''
        return list(self.terms.keys())

    @property
    def fixed_terms(self):
        ''' Return dict of all and only fixed effects in model. '''
        return {k: v for (k, v) in self.terms.items() if not v.random}

    @property
    def random_terms(self):
        ''' Return dict of all and only random effects in model. '''
        return {k: v for (k, v) in self.terms.items() if v.random}


class Term(object):

    '''
    Representation of a single model term.
    Args:
        name (str): Name of the term.
        data (DataFrame, Series, ndarray): The term values.
        categorical (bool): If True, the source variable is interpreted as
            nominal/categorical. If False, the source variable is treated
            as continuous.
        prior (Prior): A specification of the prior(s) to use. An instance
            of class priors.Prior.
    '''
    def __init__(self, name, data, categorical=False, random=False, prior=None):

        self.name = name
        self.categorical = categorical
        self.random = random
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
