import json
import re
from copy import deepcopy
from os.path import dirname, join

import numpy as np
import pandas as pd
from scipy.special import hyp2f1
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM

from bambi.external.six import string_types


class Family(object):

    '''A specification of model family.

    Attributes:
        name (str): Family name.
        prior (Prior): A Prior instance specifying the model likelihood prior.
        link (str): The name of the link function transforming the linear
            model prediction to a parameter of the likelihood.
        parent (str): The name of the prior parameter to set to the link-
            transformed predicted outcome (e.g., mu, p, etc.).
    '''

    def __init__(self, name, prior, link, parent):
        self.name = name
        self.prior = prior
        self.link = link
        self.parent = parent
        fams = {
            'gaussian': families.Gaussian,
            'bernoulli': families.Binomial,
            'poisson': families.Poisson,
            't': None  # not implemented in statsmodels
        }
        self.smfamily = fams[name] if name in fams.keys() else None


class Prior(object):

    '''Abstract specification of a term prior.

    Attributes:
        name (str): Name of prior distribution (e.g., Normal, Bernoulli, etc.)
        kwargs (dict): Optional keywords specifying the parameters of the
            named distribution.
    '''

    def __init__(self, name, scale=None, **kwargs):
        self.name = name
        self._auto_scale = True
        self.scale = scale
        self.args = {}
        self.update(**kwargs)

    def update(self, **kwargs):
        '''Update the model arguments with additional arguments.

        Args:
            kwargs (dict): Optional keyword arguments to add to prior args.
        '''

        # Backends expect numpy arrays, so make sure all numeric values are
        # represented as such.
        kwargs = {k: (np.array(v) if isinstance(v, (int, float)) else v)
                  for k, v in kwargs.items()}
        self.args.update(kwargs)


class PriorFactory(object):

    '''An object that supports specification and easy retrieval of default priors.

    Attributes:
        defaults (str, dict): Optional base configuration containing default
            priors for distribution, families, and term types. If a string,
            the name of a JSON file containing the config. If a dict, must
            contain keys for 'dists', 'terms', and 'families'; see the built-in
            JSON configuration for an example. If None, a built-in set of
            priors will be used as defaults.
        dists (dict): Optional specification of named distributions to use
            as priors. Each key gives the name of a newly defined distribution;
            values are two-element lists, where the first element is the name
            of the built-in distribution to use ('Normal', 'Cauchy', etc.),
            and the second element is a dictionary of parameters on that
            distribution (e.g., {'mu': 0, 'sd': 10}). Priors can be nested
            to arbitrary depths by replacing any parameter with another prior
            specification.
        terms (dict): Optional specification of default priors for different
            model term types. Valid keys are 'intercept', 'fixed', or 'random'.
            Values are either strings preprended by a #, in which case they
            are interpreted as pointers to distributions named in the dists
            dictionary, or key -> value specifications in the same format as
            elements in the dists dictionary.
        families (dict): Optional specification of default priors for named
            family objects. Keys are family names, and values are dicts
            containing mandatory keys for 'dist', 'link', and 'parent'.

    Examples:
        >>> dists = { 'my_dist': ['Normal', {'mu': 10, 'sd': 1000}]}
        >>> pf = PriorFactory(dists=dists)

        >>> families = { 'normalish': { 'dist': ['normal', {sd: '#my_dist'}],
        >>>                             link:'identity', parent: 'mu'}}
        >>> pf = PriorFactory(dists=dists, families=families)
    '''

    def __init__(self, defaults=None, dists=None, terms=None, families=None):

        if defaults is None:
            defaults = join(dirname(__file__), 'config', 'priors.json')

        if isinstance(defaults, string_types):
            defaults = json.load(open(defaults, 'r'))

        # Just in case the user plans to use the same defaults elsewhere
        defaults = deepcopy(defaults)

        if isinstance(dists, dict):
            defaults['dists'].update(dists)

        if isinstance(terms, dict):
            defaults['terms'].update(terms)

        if isinstance(families, dict):
            defaults['families'].update(families)

        self.dists = defaults['dists']
        self.terms = defaults['terms']
        self.families = defaults['families']

    def _get_prior(self, spec, **kwargs):

        if isinstance(spec, string_types):
            spec = re.sub(r'^\#', '', spec)
            return self._get_prior(self.dists[spec])
        elif isinstance(spec, (list, tuple)):
            name, args = spec
            if name.startswith('#'):
                name = re.sub(r'^\#', '', name)
                prior = self._get_prior(self.dists[name])
            else:
                prior = Prior(name, **kwargs)
            args = {k: self._get_prior(v) for (k, v) in args.items()}
            prior.update(**args)
            return prior
        else:
            return spec

    def get(self, dist=None, term=None, family=None):
        '''Retrieve default prior for a named distribution, term type, or family.

        Args:
            dist (str): Name of desired distribution. Note that the name is
                the key in the defaults dictionary, not the name of the
                Distribution object used to construct the prior.
            term (str): The type of term family to retrieve defaults for.
                Must be one of 'intercept', 'fixed', or 'random'.
            family (str): The name of the Family to retrieve. Must be a value
                defined internally. In the default config, this is one of
                'gaussian', 'bernoulli', 'poisson', or 't'.
        '''

        if dist is not None:
            if dist not in self.dists:
                raise ValueError(
                    "'%s' is not a valid distribution name." % dist)
            return self._get_prior(self.dists[dist])
        elif term is not None:
            if term not in self.terms:
                raise ValueError("'%s' is not a valid term type." % term)
            return self._get_prior(self.terms[term])
        elif family is not None:
            if family not in self.families:
                raise ValueError("'%s' is not a valid family name." % family)
            _f = self.families[family]
            prior = self._get_prior(_f['dist'])
            return Family(family, prior, _f['link'], _f['parent'])


class PriorScaler(object):

    # Default is 'wide'. The wide prior SD is sqrt(1/3) = .577 on the partial
    # corr scale, which is the SD of a flat prior over [-1,1].
    names = {
        'narrow': 0.2,
        'medium': 0.4,
        'wide': 3 ** -0.5,
        'superwide': 0.8
    }

    def __init__(self, model, taylor):
        self.model = model
        self.stats = model.dm_statistics if hasattr(model, 'dm_statistics') \
            else None
        self.dm = pd.DataFrame({lev: t.data[:, i]
                                for t in model.fixed_terms.values()
                                for i, lev in enumerate(t.levels)})
        self.priors = {}
        missing = 'drop' if self.model.dropna else 'none'
        self.mle = GLM(endog=self.model.y.data, exog=self.dm,
                          family=self.model.family.smfamily(),
                          missing=missing).fit()
        self.taylor = taylor
        with open(join(dirname(__file__), 'config', 'derivs.txt'), 'r') as file:
            self.deriv = [next(file).strip('\n') for x in range(taylor+1)]

    def _get_slope_stats(self, exog, predictor, sd_corr, full_mod=None,
                         points=4):
        '''
        Args:
            full_mod: statsmodels GLM to replace MLE model. For when 'predictor'
                is not in the fixed part of the model.
            points: number of points to use for LL approximation.
        '''

        if full_mod is None:
            full_mod = self.mle

        # figure out which column of exog to drop for the null model
        keeps = [i for i, x in enumerate(list(exog.columns))
                 if not np.array_equal(predictor, exog[x].values.flatten())]
        i = [x for x in range(exog.shape[1]) if x not in keeps][0]

        # get log-likelihood values from beta=0 to beta=MLE
        values = np.linspace(0., full_mod.params[i], points)
        # if there are multiple predictors, use statsmodels to optimize the LL
        if keeps:
            null = [GLM(endog=self.model.y.data, exog=exog,
                           family=self.model.family.smfamily()).fit_constrained(
                                str(exog.columns[i])+'='+str(val),
                                start_params=full_mod.params.values)
                    for val in values[:-1]]
            null = np.append(null, full_mod)
            ll = np.array([x.llf for x in null])
        # if just a single predictor, use statsmodels to evaluate the LL
        else:
            null = [self.model.family.smfamily().loglike(
                np.squeeze(self.model.y.data), val*predictor)
                for val in values[:-1]]
            ll = np.append(null, full_mod.llf)

        # compute params of quartic approximatino to log-likelihood
        # c: intercept, d: shift parameter
        # a: quartic coefficient, b: quadratic coefficient

        c, d = ll[-1], -(full_mod.params[i].item())
        X = np.array([(values+d)**4,
                       (values+d)**2]).T
        a, b = np.squeeze(
            np.linalg.multi_dot([
                np.linalg.inv(np.dot(X.T, X)),
                X.T,
                (ll[:, None] - c)
            ])
        )

        # m, v: mean and variance of beta distribution of correlations
        # p, q: corresponding shape parameters of beta distribution
        m = .5
        v = sd_corr**2/4
        p = m*(m*(1-m)/v - 1)
        q = (1-m)*(m*(1-m)/v - 1)

        # function to return central moments of rescaled beta distribution
        def moment(k): return (2*p/(p+q))**k * hyp2f1(p, -k, p+q, (p+q)/p)

        # evaluate the derivatives of beta = f(correlation).
        # dict 'point' gives points about which to Taylor expand. We want to
        # expand about the mean (generally 0), but some of the derivatives
        # do not exist at 0. Evaluating at a point very close to 0 (e.g., .001)
        # generally gives good results, but the higher order the expansion, the
        # further from 0 we need to evaluate the derivatives, or they blow up.
        point = dict(zip(range(1, 14), 2**np.linspace(-1, 5, 13)/100))
        vals = dict(a=a, b=b, n=len(self.model.y.data), r=point[self.taylor])
        _deriv = [eval(x, globals(), vals) for x in self.deriv]

        # compute and return the approximate SD
        def term(i, j):
            return 1/np.math.factorial(i) * 1/np.math.factorial(j) \
                * _deriv[i] * _deriv[j] \
                * (moment(i+j) - moment(i)*moment(j))
        terms = [term(i, j) for i in range(1, self.taylor+1)
                 for j in range(1, self.taylor+1)]
        return np.array(terms).sum()**.5

    def _get_intercept_stats(self, add_slopes=True):
        # start with mean and variance of Y on the link scale
        mod = GLM(endog=self.model.y.data,
                     exog=np.repeat(1, len(self.model.y.data)),
                     family=self.model.family.smfamily(),
                     missing='drop' if self.model.dropna else 'none').fit()
        mu = mod.params
        # multiply SE by sqrt(N) to turn it into (approx.) SD(Y) on link scale
        sd = (mod.cov_params()[0] * len(mod.mu))**.5

        # modify mu and sd based on means and SDs of slope priors.
        if len(self.model.fixed_terms) > 1 and add_slopes:
            means = np.array([x['mu'] for x in self.priors.values()])
            sds = np.array([x['sd'] for x in self.priors.values()])
            # add to intercept prior
            index = list(self.priors.keys())
            mu -= np.dot(means, self.stats['mean_x'][index])
            sd = (sd**2 + np.dot(sds**2, self.stats['mean_x'][index]**2))**.5

        return mu, sd

    def _scale_fixed(self, term):

        # these defaults are only defined for Normal priors
        if term.prior.name != 'Normal':
            return

        mu = []
        sd = []
        sd_corr = term.prior.scale
        for pred in term.data.T:
            mu += [0]
            sd += [self._get_slope_stats(exog=self.dm, predictor=pred,
                                         sd_corr=sd_corr)]

        # save and set prior
        for i, lev in enumerate(term.levels):
            self.priors.update({lev: {'mu': mu[i], 'sd': sd[i]}})
        term.prior.update(mu=np.array(mu), sd=np.array(sd))

    def _scale_intercept(self, term):

        # default priors are only defined for Normal priors
        if term.prior.name != 'Normal':
            return

        # get prior mean and SD for fixed intercept
        mu, sd = self._get_intercept_stats()

        # save and set prior
        term.prior.update(mu=mu, sd=sd)

    def _scale_random(self, term):

        # these default priors are only defined for HalfNormal priors
        if term.prior.args['sd'].name != 'HalfNormal':
            return

        sd_corr = term.prior.scale

        # recreate the corresponding fixed effect data
        fix_data = term.data.sum(axis=1)

        # handle intercepts and cell means
        if term.constant:
            mu, sd = self._get_intercept_stats()
            sd *= sd_corr
        # handle slopes
        else:
            exists = [x for x in self.dm.columns
                      if np.array_equal(fix_data, self.dm[x].values)]
            # handle case where there IS a corresponding fixed effect
            if exists and exists[0] in self.priors.keys():
                sd = self.priors[exists[0]]['sd']
            # handle case where there IS NOT a corresponding fixed effect
            else:
                # the usual case: add the random effect data as a fixed effect
                # in the design matrix
                if not exists:
                    fix_dataframe = pd.DataFrame(fix_data)
                    # things break if column names are integers (the default)
                    fix_dataframe.rename(
                        columns={c: '_'+str(c) for c in fix_dataframe.columns},
                        inplace=True)
                    exog = self.dm.join(fix_dataframe)
                # this handles the corner case where there technically is the
                # corresponding fixed effect, but the parameterization differs
                # between the fixed- and random-effect specification. usually
                # this means the fixed effects use cell-means coding but the
                # random effects use k-1 coding
                else:
                    group = term.name.split('|')[1]
                    exog = self.model.random_terms.values()
                    exog = [v.data.sum(1) for v in exog
                            if v.name.split('|')[-1] == group]
                    index = ['_'+str(i) for i in range(len(exog))]
                    exog = pd.DataFrame(exog, index=index).T
                # this will replace self.mle (which is missing predictors)
                missing = 'drop' if self.model.dropna else 'none'
                full_mod = GLM(endog=self.model.y.data, exog=exog,
                                  family=self.model.family.smfamily(),
                                  missing=missing).fit()
                sd = self._get_slope_stats(exog=exog, predictor=fix_data,
                                           full_mod=full_mod, sd_corr=sd_corr)

        # set the prior SD.
        term.prior.args['sd'].update(sd=np.squeeze(np.atleast_1d(sd)))

    def scale(self):
        # classify all terms
        fixed_intercepts = [t for t in self.model.terms.values()
                            if not t.random and t.data.sum(1).var() == 0]
        fixed_slopes = [t for t in self.model.terms.values()
                        if not t.random and not t.data.sum(1).var() == 0]
        random_terms = [t for t in self.model.terms.values() if t.random]

        # arrange them in the order in which they should be initialized
        term_list = fixed_slopes + fixed_intercepts + random_terms
        term_types = ['fixed']*len(fixed_slopes) + \
            ['intercept']*len(fixed_intercepts) + \
            ['random']*len(random_terms)

        # initialize them in order
        for t, term_type in zip(term_list, term_types):
            if t.prior.scale is None:
                if not t.prior._auto_scale or not self.model.auto_scale:
                    continue
                t.prior.scale = 'wide'

            # Convert scale names to float
            if isinstance(t.prior.scale, string_types):
                t.prior.scale = PriorScaler.names[t.prior.scale]

            # scale it!
            getattr(self, '_scale_%s' % term_type)(t)
