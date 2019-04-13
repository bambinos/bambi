import pandas as pd
import numpy as np
from collections import OrderedDict
from patsy import dmatrices, dmatrix
import re
import warnings
import statsmodels.api as sm
import matplotlib.pyplot as plt
from copy import deepcopy
from bambi.external.six import string_types
from bambi.external.patsy import Custom_NA, rename_columns
from bambi.priors import PriorFactory, PriorScaler, Prior
from bambi.utils import listify
import pymc3 as pm


class Model(object):

    '''
    Attributes:
        data (DataFrame, str): the dataset to use. Either a pandas
            DataFrame, or the name of the file containing the data, which
            will be passed to pd.read_csv().
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
        taylor (int): Order of Taylor expansion to use in approximate variance
            when constructing the default priors. Should be between 1 and 13.
            Lower values are less accurate, tending to undershoot the correct
            prior width, but are faster to compute and more stable. Odd-
            numbered values tend to work better. Defaults to 5 for Normal
            models and 1 for non-Normal models. Values higher than the defaults
            are generally not recommended as they can be unstable.
        noncentered (bool): If True (default), uses a non-centered
            parameterization for normal hyperpriors on grouped parameters.
            If False, naive (centered) parameterization is used.
    '''

    def __init__(self, data=None, default_priors=None, auto_scale=True,
                 dropna=False, taylor=None, noncentered=True):

        if isinstance(data, string_types):
            data = pd.read_csv(data, sep=None, engine="python")

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

        self.auto_scale = auto_scale
        self.dropna = dropna
        self.taylor = taylor
        self.noncentered = noncentered
        self._backend_name = None

        # build() will loop over these, calling _add() and _set_priors()
        self.added_terms = []
        self._added_priors = {}

        # if dropna=True, completes gets updated by add() to track complete cases
        self.completes = []
        self.clean_data = None

    def reset(self):
        '''
        Reset list of terms and y-variable.
        '''
        self.terms = OrderedDict()
        self.y = None
        self.backend = None
        self.added_terms = []
        self._added_priors = {}
        self.completes = []
        self.clean_data = None

    def _set_backend(self, backend):

        backend = backend.lower()

        if backend.startswith('pymc'):
            from bambi.backends import PyMC3BackEnd
            self.backend = PyMC3BackEnd()
        elif backend == 'stan':
            from bambi.backends import StanBackEnd
            self.backend = StanBackEnd()
        else:
            raise ValueError(
                "At the moment, only the PyMC3 and Stan backends are "
                "supported.")

        self._backend_name = backend

    def build(self, backend=None):
        '''Set up the model for sampling/fitting.

        Performs any steps that require access to all model terms (e.g., scaling priors
        on each term), then calls the BackEnd's build() method.

        Args:
            backend (str): The name of the backend to use for model fitting.
                Currently, 'pymc' and 'stan' are supported. If None, assume
                that fit() has already been called (possibly without building),
                and look in self._backend_name.
        '''

        # retain only the complete cases
        n_total = len(self.data.index)
        if len(self.completes):
            completes = [set(x) for x in sum(self.completes, [])]
            completes = set.intersection(*completes)
        else:
            completes = [x for x in range(len(self.data.index))]
        self.clean_data = self.data.iloc[list(completes), :]
        # warn the user about any dropped rows
        if len(completes) < n_total:
            msg = "Automatically removing {}/{} rows from the dataset."
            msg = msg.format(n_total - len(completes), n_total)
            warnings.warn(msg)

        # loop over the added terms and actually _add() them
        for term_args in self.added_terms:
            self._add(**term_args)

        # set custom priors
        self._set_priors(**self._added_priors)

        # prepare all priors
        for name, term in self.terms.items():
            type_ = 'intercept' if name == 'Intercept' else \
                'random' if self.terms[name].random else 'fixed'
            term.prior = self._prepare_prior(term.prior, type_)

        # check for backend
        if backend is None:
            if self._backend_name is None:
                raise ValueError("Error: no backend was passed or set in the "
                                 "Model; did you forget to call fit()?")
            backend = self._backend_name

        # check for outcome
        if self.y is None:
            raise ValueError("No outcome (y) variable is set! Please specify "
                             "an outcome variable using the formula interface "
                             "before build() or fit().")

        # X = fixed effects design matrix (excluding intercept/constant term)
        # r2_x = 1 - 1/VIF, i.e., R2 for predicting each x from all other x's.
        # only compute these stats if there are multiple terms in the model
        terms = [t for t in self.fixed_terms.values() if t.name != 'Intercept']

        if len(self.fixed_terms) > 1:

            X = [pd.DataFrame(x.data, columns=x.levels) for x in terms]
            X = pd.concat(X, axis=1)

            self.dm_statistics = {
                'r2_x': pd.Series({
                    x: sm.OLS(endog=X[x],
                              exog=sm.add_constant(X.drop(x, axis=1))
                              if 'Intercept' in self.term_names
                              else X.drop(x, axis=1)).fit().rsquared
                    for x in list(X.columns)}),
                'sd_x': X.std(),
                'mean_x': X.mean(axis=0)
            }

            # save potentially useful info for diagnostics, send to
            # ModelResults.
            # mat = correlation matrix of X, w/ diagonal replaced by X means
            mat = X.corr()
            for x in list(mat.columns):
                mat.loc[x, x] = self.dm_statistics['mean_x'][x]
            self._diagnostics = {
                # the Variance Inflation Factors (VIF), which is possibly
                # useful for diagnostics
                'VIF': 1/(1 - self.dm_statistics['r2_x']),
                'corr_mean_X': mat
            }

            # throw informative error if perfect collinearity among fixed fx
            if any(self.dm_statistics['r2_x'] > .999):
                raise ValueError(
                    "There is perfect collinearity among the fixed effects!\n"
                    "Printing some design matrix statistics:\n" +
                    str(self.dm_statistics) + '\n' +
                    str(self._diagnostics))

        # throw informative error message if any categorical predictors have 1
        # category
        num_cats = [x.data.size for x in self.fixed_terms.values()]
        if any(np.array(num_cats) == 0):
            raise ValueError(
                "At least one categorical predictor contains only 1 category!")

        # only set priors if there is at least one term in the model
        if len(self.terms) > 0:
            # Get and scale default priors if none are defined yet
            if self.taylor is not None:
                taylor = self.taylor
            else:
                taylor = 5 if self.family.name == 'gaussian' else 1
            scaler = PriorScaler(self, taylor=taylor)
            scaler.scale()

        # For bernoulli models with n_trials = 1 (most common use case),
        # tell user which event is being modeled
        if self.family.name == 'bernoulli' and np.max(self.y.data) < 1.01:
            event = next(
                i for i, x in enumerate(self.y.data.flatten()) if x > .99)
            warnings.warn('Modeling the probability that {}==\'{}\''.format(
                self.y.name, str(self.clean_data[self.y.name].iloc[event])))

        self._set_backend(backend)
        self.backend.build(self)
        self.built = True

    def fit(self, fixed=None, random=None, priors=None, family='gaussian',
            link=None, run=True, categorical=None, backend=None, **kwargs):
        '''Fit the model using the specified BackEnd.

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
                families are 'gaussian', 'bernoulli', 'poisson', and 't'.
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
            backend (str): The name of the BackEnd to use. Currently only
                'pymc' and 'stan' backends are supported. Defaults to PyMC3.
        '''

        if fixed is not None or random is not None:
            self.add(fixed=fixed, random=random, priors=priors, family=family,
                     link=link, categorical=categorical, append=False)

        ''' Run the BackEnd to fit the model. '''
        if backend is None:
            backend = 'pymc' if self._backend_name is None else self._backend_name

        if run:
            if not self.built or backend != self._backend_name:
                self.build(backend)
            return self.backend.run(**kwargs)

        self._backend_name = backend

    def add(self, fixed=None, random=None, priors=None, family='gaussian',
            link=None, categorical=None, append=True):
        '''Adds one or more terms to the model via an R-like formula syntax.

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
                families are 'gaussian', 'bernoulli', 'poisson', and 't'.
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
            append (bool): If True, terms are appended to the existing model
                rather than replacing any existing terms. This allows
                formula-based specification of the model in stages.
        '''

        data = self.data

        # Primitive values (floats, strs) can be overwritten with Prior objects
        # so we need to make sure to copy first to avoid bad things happening
        # if user is re-using same prior dict in multiple models.
        if priors is None:
            priors = {}
        else:
            priors = deepcopy(priors)

        if not append:
            self.reset()

        # Explicitly convert columns to category if desired--though this
        # can also be done within the formula using C().
        if categorical is not None:
            data = data.copy()
            cats = listify(categorical)
            data[cats] = data[cats].apply(lambda x: x.astype('category'))

        # Custom patsy.missing.NAAction class. Similar to patsy drop/raise
        # defaults, but changes the raised message and logs any dropped rows
        NA_handler = Custom_NA(dropna=self.dropna)

        # screen fixed terms
        if fixed is not None:
            if '~' in fixed:
                clean_fix = re.sub(r'\[.+\]', '', fixed)
                dmatrices(clean_fix, data=data, NA_action=NA_handler)
            else:
                dmatrix(fixed, data=data, NA_action=NA_handler)

        # screen random terms
        if random is not None:
            for term in listify(random):
                for side in term.split('|'):
                    dmatrix(side, data=data, NA_action=NA_handler)

        # update the running list of complete cases
        if len(NA_handler.completes):
            self.completes.append(NA_handler.completes)

        # save arguments to pass to _add()
        args = dict(zip(
            ['fixed', 'random', 'priors', 'family', 'link', 'categorical'],
            [fixed, random, priors, family, link, categorical]))
        self.added_terms.append(args)

        self.built = False

    def _add(self, fixed=None, random=None, priors=None, family='gaussian',
             link=None, categorical=None, append=True):
        '''Internal version of add(), with the same arguments.

        Runs during Model.build()
        '''

        # use cleaned data with NAs removed (if user requested)
        data = self.clean_data
        # alter this pandas flag to avoid false positive SettingWithCopyWarnings
        data._is_copy = False

        # Explicitly convert columns to category if desired--though this
        # can also be done within the formula using C().
        if categorical is not None:
            data = data.copy()
            cats = listify(categorical)
            data[cats] = data[cats].apply(lambda x: x.astype('category'))

        if fixed is not None:
            if '~' in fixed:
                # check to see if formula is using the 'y[event] ~ x' syntax
                # (for bernoulli models). If so, chop it into groups:
                # 1 = 'y[event]', 2 = 'y', 3 = 'event', 4 = 'x'
                # If this syntax is not being used, event = None
                event = re.match(r'^((\S+)\[(\S+)\])\s*~(.*)$', fixed)
                if event is not None:
                    fixed = '{}~{}'.format(event.group(2), event.group(4))
                y, X = dmatrices(fixed, data=data, NA_action='raise')
                y_label = y.design_info.term_names[0]
                if event is not None:
                    # pass in new Y data that has 1 if y=event and 0 otherwise
                    y_data = y[:, y.design_info.column_names.index(event.group(1))]
                    y_data = pd.DataFrame({event.group(3): y_data})
                    self._add_y(y_label, family=family, link=link, data=y_data)
                else:
                    # use Y as-is
                    self._add_y(y_label, family=family, link=link)
            else:
                X = dmatrix(fixed, data=data, NA_action='raise')

            # Loop over predictor terms
            for _name, _slice in X.design_info.term_name_slices.items():
                cols = X.design_info.column_names[_slice]
                term_data = pd.DataFrame(X[:, _slice], columns=cols)
                prior = priors.pop(_name, priors.get('fixed', None))
                self.terms[_name] = Term(_name, term_data, prior=prior)

        # Random effects
        if random is not None:

            random = listify(random)

            for f in random:

                f = f.strip()

                # Split specification into intercept, predictor, and grouper
                patt = r'^([01]+)*[\s\+]*([^\|]+)*\|(.*)'

                intcpt, pred, grpr = re.search(patt, f).groups()
                label = '{}|{}'.format(pred, grpr) if pred else grpr
                prior = priors.pop(label, priors.get('random', None))

                # Treat all grouping variables as categoricals, regardless of
                # their dtype and what the user may have specified in the
                # 'categorical' argument.
                var_names = re.findall('(\w+)', grpr)
                for v in var_names:
                    if v in data.columns:
                        data.loc[:, v] = data.loc[:, v].astype('category')
                        self.clean_data.loc[:, v] = data.loc[:, v]

                # Default to including random intercepts
                intcpt = 1 if intcpt is None else int(intcpt)

                grpr_df = dmatrix('0+%s' % grpr, data, return_type='dataframe',
                                  NA_action='raise')

                # If there's no predictor, we must be adding random intercepts
                if not pred and grpr not in self.terms:
                    name = '1|' + grpr
                    pred = np.ones((len(grpr_df), 1))
                    term = RandomTerm(name, grpr_df, pred, grpr_df.values,
                                      categorical=True, prior=prior)
                    self.terms[name] = term
                else:
                    pred_df = dmatrix('%s+%s' % (intcpt, pred), data,
                                      return_type='dataframe',
                                      NA_action='raise')
                    # determine value of the 'constant' attribute
                    const = np.atleast_2d(pred_df.T).T.sum(1).var() == 0

                    for col, i in pred_df.design_info.column_name_indexes.items():
                        pred_data = pred_df.iloc[:, i]
                        lev_data = grpr_df.multiply(pred_data, axis=0)

                        # Also rename intercepts and skip if already added.
                        # This can happen if user specifies something like
                        # random=['1|school', 'student|school'].
                        if col == 'Intercept':
                            if grpr in self.terms:
                                continue
                            label = '1|%s' % grpr
                        else:
                            label = col + '|' + grpr

                        prior = priors.pop(label, priors.get('random', None))

                        # Categorical or continuous is determined from data
                        ld = lev_data.values
                        if ((ld == 0) | (ld == 1)).all():
                            lev_data = lev_data.astype(int)
                            cat = True
                        else:
                            cat = False

                        pred_data = pred_data[:, None]  # Must be 2D later
                        term = RandomTerm(label, lev_data, pred_data,
                                          grpr_df.values, categorical=cat,
                                          constant=const if const else None,
                                          prior=prior)
                        self.terms[label] = term

    def _add_y(self, variable, prior=None, family='gaussian', link=None, *args,
               **kwargs):
        '''Add a dependent (or outcome) variable to the model.

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
                families are 'gaussian', 'bernoulli', 'poisson', and 't'.
            link (str): The model link function to use. Can be either a string
                (must be one of the options defined in the current backend;
                typically this will include at least 'identity', 'logit',
                'inverse', and 'log'), or a callable that takes a 1D ndarray
                or theano tensor as the sole argument and returns one with
                the same shape.
            args, kwargs: Optional positional and keyword arguments to pass
                onto Term initializer.
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
            prior.update(sd=Prior('Uniform', lower=0,
                                  upper=self.clean_data[variable].std()))

        data = kwargs.pop('data', self.clean_data[variable])
        term = Term(variable, data, prior=prior, *args, **kwargs)
        self.y = term
        self.built = False

    def _match_derived_terms(self, name):
        '''
        Returns all (random) terms whose named are derived from the
        specified string. For example, 'condition|subject' should match the
        terms with names '1|subject', 'condition[T.1]|subject', and so on.
        Only works for strings with grouping operator ('|').
        '''

        if '|' not in name:
            return None

        patt = r'^([01]+)*[\s\+]*([^\|]+)*\|(.*)'
        intcpt, pred, grpr = re.search(patt, name).groups()

        intcpt = '1|%s' % grpr
        if not pred:
            return [self.terms[intcpt]] if intcpt in self.terms else None

        source = '%s|%s' % (pred, grpr)
        found = [t for (n, t) in self.terms.items() if n == intcpt or
                 re.sub('(\[.*?\])', '', n) == source]
        # If only the intercept matches, return None, because we want to err
        # on the side of caution and not consider '1|subject' to be a match for
        # 'condition|subject' if no slopes are found (e.g., the intercept could
        # have been set by some other specification like 'gender|subject').
        return found if found and (len(found) > 1 or found[0].name != intcpt) \
            else None

    def set_priors(self, priors=None, fixed=None, random=None,
                   match_derived_names=True):
        '''Set priors for one or more existing terms.

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
            match_derived_names (bool): if True, the specified prior(s) will be
                applied not only to terms that match the keyword exactly,
                but to the levels of random effects that were derived from
                the original specification with the passed name. For example,
                `priors={'condition|subject':0.5}` would apply the prior
                to the terms with names '1|subject', 'condition[T.1]|subject',
                and so on. If False, an exact match is required for the
                prior to be applied.
        '''
        # save arguments to pass to _set_priors() at build time
        kwargs = dict(zip(
            ['priors', 'fixed', 'random', 'match_derived_names'],
            [priors, fixed, random, match_derived_names]))
        self._added_priors.update(kwargs)

        self.built = False

    def _set_priors(self, priors=None, fixed=None, random=None,
                    match_derived_names=True):
        '''Internal version of set_priors(), with same arguments.

        Runs during Model.build().
        '''

        targets = {}

        if fixed is not None:
            targets.update({name: fixed for name in self.fixed_terms.keys()})

        if random is not None:
            targets.update({name: random for name in self.random_terms.keys()})

        if priors is not None:
            for k, prior in priors.items():
                for name in listify(k):
                    term_names = list(self.terms.keys())
                    msg = "No terms in model match '%s'." % name
                    if name not in term_names:
                        terms = self._match_derived_terms(name)
                        if not match_derived_names or terms is None:
                            raise ValueError(msg)
                        for t in terms:
                            targets[t.name] = prior
                    else:
                        targets[name] = prior

        for name, prior in targets.items():
            self.terms[name].prior = prior

    # helper function to correctly set default priors, auto_scaling, etc.
    def _prepare_prior(self, prior, _type):
        '''
        Args:
            prior: Prior object, or float, or None.
            type (string): 'intercept, 'fixed', or 'random'.
        '''

        if prior is None and not self.auto_scale:
            prior = self.default_priors.get(term=_type + '_flat')

        if isinstance(prior, Prior):
            prior._auto_scale = False
        else:
            _scale = prior
            prior = self.default_priors.get(term=_type)
            prior.scale = _scale
            if prior.scale is not None:
                prior._auto_scale = False
        return prior

    def plot(self, varnames=None):
        self.plot_priors(varnames)

    def plot_priors(self, varnames=None):
        if not self.built:
            raise ValueError("Cannot plot priors until model is built!")

        with pm.Model():
            # get priors for fixed fx, separately for each level of each
            # predictor
            dists = []
            for t in self.fixed_terms.values():
                if varnames is not None and t.name not in varnames:
                    continue
                for i, l in enumerate(t.levels):
                    params = {k: v[i % len(v)]
                              if isinstance(v, np.ndarray) else v
                              for k, v in t.prior.args.items()}
                    dists += [getattr(pm, t.prior.name)(l, **params)]

            # get priors for random effect SDs
            for t in self.random_terms.values():
                if varnames is not None and t.name not in varnames:
                    continue
                prior = t.prior.args['sd'].name
                params = t.prior.args['sd'].args
                dists += [getattr(pm, prior)(t.name + '_sd', **params)]

            # add priors on Y params if applicable
            y_prior = [(k, v) for k, v in self.y.prior.args.items()
                       if isinstance(v, Prior)]
            if len(y_prior):
                for p in y_prior:
                    pm_attr = getattr(pm, p[1].name)
                    p = pm_attr('_'.join([self.y.name, p[0]]), **p[1].args)
                    dists.extend([p])

            # make the plot!
            p = float(len(dists))
            fig, axes = plt.subplots(int(np.ceil(p / 2)), 2,
                                     figsize=(12, np.ceil(p / 2) * 2))
            # in case there is only 1 row
            if int(np.ceil(p / 2)) < 2:
                axes = axes[None, :]
            for i, d in enumerate(dists):
                dist = d.distribution if isinstance(d, pm.model.FreeRV) else d
                samp = pd.Series(dist.random(size=1000).flatten())
                samp.plot(kind='hist', ax=axes[divmod(i, 2)[0], divmod(i, 2)[1]],
                          density=True)
                samp.plot(kind='kde', ax=axes[divmod(i, 2)[0], divmod(i, 2)[1]],
                          color='b')
                axes[divmod(i, 2)[0], divmod(i, 2)[1]].set_title(d.name)
            fig.tight_layout()

        return axes

    @property
    def term_names(self):
        '''Return names of all terms in order of addition to model.'''
        return list(self.terms.keys())

    @property
    def fixed_terms(self):
        '''Return dict of all and only fixed effects in model.'''
        return {k: v for (k, v) in self.terms.items() if not v.random}

    @property
    def random_terms(self):
        '''Return dict of all and only random effects in model.'''
        return {k: v for (k, v) in self.terms.items() if v.random}


class Term(object):

    '''Representation of a single (fixed) model term.

    Attributes:
        name (str): Name of the term.
        data (DataFrame, Series, ndarray): The term values.
        categorical (bool): If True, the source variable is interpreted as
            nominal/categorical. If False, the source variable is treated
            as continuous.
        prior (Prior): A specification of the prior(s) to use. An instance
            of class priors.Prior.
        constant (bool): indicates whether the term levels collectively
            act as a constant, in which case the term is treated as an
            intercept for prior distribution purposes.
    '''

    random = False

    def __init__(self, name, data, categorical=False, prior=None, constant=None):

        self.name = name
        self.categorical = categorical
        self._reduced_data = None

        if isinstance(data, pd.Series):
            data = data.to_frame()
        if isinstance(data, pd.DataFrame):
            self.levels = list(data.columns)
            data = data.values

        # Random effects pass through here
        else:
            data = np.atleast_2d(data)
            self.levels = list(range(data.shape[1]))

        self.data = data

        # identify and flag intercept and cell-means terms (i.e., full-rank
        # dummy codes), which receive special priors
        if constant is None:
            self.constant = np.atleast_2d(data.T).T.sum(1).var() == 0
        else:
            self.constant = constant

        self.prior = prior


class RandomTerm(Term):

    random = True

    def __init__(self, name, data, predictor, grouper, categorical=False,
        prior=None, constant=None):

        super(RandomTerm, self).__init__(name, data, categorical, prior,
              constant)
        self.grouper = grouper
        self.predictor = predictor
        self.group_index = self._invert_dummies(grouper)

    def _invert_dummies(self, dummies):
        '''
        For the sake of computational efficiency (i.e., to avoid lots of
        large matrix multiplications in the backends), invert the dummy-coding
        process and represent full-rank dummies as a vector of indices into the
        coefficients.
        '''
        vec = np.zeros(len(dummies), dtype=int)
        for i in range(1, dummies.shape[1]):
            vec[dummies[:, i] == 1] = i
        return vec
