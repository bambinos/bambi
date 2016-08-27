import numpy as np
from os.path import dirname, join
from bambi.external.six import string_types
from copy import deepcopy
import json
import re


class Family(object):

    '''
    A specification of model family.
    Args:
        name (str): Family name
        prior (Prior): A Prior instance specifying the model likelihood prior
        link (str): The name of the link function transforming the linear
            model prediction to a parameter of the likelihood
        parent (str): The name of the prior parameter to set to the link-
            transformed predicted outcome (e.g., mu, p, etc.).
    '''

    def __init__(self, name, prior, link, parent):
        self.name = name
        self.prior = prior
        self.link = link
        self.parent = parent


class Prior(object):

    '''
    Abstract specification of a term prior.
    Args:
        name (str): Name of prior distribution (e.g., Normal, Binomial, etc.)
        kwargs (dict): Optional keywords specifying the parameters of the
            named distribution.
    '''

    def __init__(self, name, **kwargs):
        self.name = name
        self.args = {}
        self.update(**kwargs)

    def update(self, **kwargs):
        '''
        Update the model arguments with additional arguments.
        Args:
            kwargs (dict): Optional keyword arguments to add to prior args.
        '''
        self.args.update(kwargs)


class PriorFactory(object):

    '''
    An object that supports specification and easy retrieval of default priors.
    Args:
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

    def _get_prior(self, spec):

        if isinstance(spec, string_types):
            spec = re.sub('^\#', '', spec)
            return self._get_prior(self.dists[spec])
        elif isinstance(spec, (list, tuple)):
            name, args = spec
            if name.startswith('#'):
                name = re.sub('^\#', '', name)
                prior = self._get_prior(self.dists[name])
            else:
                prior = Prior(name)
            args = {k: self._get_prior(v) for (k, v) in args.items()}
            prior.update(**args)
            return prior
        else:
            return spec

    def get(self, dist=None, term=None, family=None, **kwargs):
        '''
        Retrieve default prior for a named distribution, term type, or family.
        Args:
            dist (str): Name of desired distribution. Note that the name is
                the key in the defaults dictionary, not the name of the
                Distribution object used to construct the prior.
            term (str): The type of term family to retrieve defaults for.
                Must be one of 'intercept', 'fixed', or 'random'.
            family (str): The name of the Family to retrieve. Must be a value
                defined internally. In the default config, this is one of
                'gaussian', 'binomial', 'poisson', or 't'.
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

    # Default is 'value'. The value prior SD is sqrt(1/3) = .577 on the partial
    # corr scale, which is the SD of a flat prior over [-1,1].
    names = {
        'narrow': 0.2,
        'medium': 0.4,
        'value': 3 ** -0.5,
        'supervalue': 0.8
    }

    def __init__(self, model):
        self.model = model
        self.stats = model.dm_statistics # purely for brevity

    def _scale_intercept(self, term, value):

        # default priors are only defined for Normal priors, although
        # we could probably easily handle Cauchy by just substituting
        # 'sd' -> 'beta'
        if term.prior.name != 'Normal':
            return

        index = list(self.stats['r2_y'].index)
        sd = self.stats['sd_y'] * (1 - self.stats['r2_y'][index]) / \
            self.stats['sd_x'][index] / (1 - self.stats['r2_x'][index])
        sd *= value
        sd = np.dot(sd**2, self.stats['mean_x'][index]**2)**.5
        term.prior.update(mu=self.stats['mean_y'], sd=sd)

    def _scale_fixed(self, term, value):

        if term.prior.name != 'Normal':
            return

        slope_constant = self.stats['sd_y'] * \
            (1 - self.stats['r2_y'][term.levels]) / \
            self.stats['sd_x'][term.levels] / \
            (1 - self.stats['r2_x'][term.levels])
        term.prior.update(sd=value * slope_constant.values)

    def _scale_random(self, term, value):

        term_type = 'intercept' if '|' not in term.name else 'slope'

        # these default priors are only defined for HalfCauchy priors
        if term.prior.args['sd'].name != 'HalfCauchy':
            return

        # handle random slopes
        if term_type == 'slope':
            # get name of corresponding fixed effect
            fix = re.sub(r'\|.*', r'', term.name).strip()
            # only proceed if there does exist a corresponding fixed
            # effect. note that without this, it would break on random
            # slopes for categorical predictors! Here we simply skip
            # that case, but we should make it correctly handle default
            # priors for that case
            if fix not in list(self.stats['r2_y'].index):
                return
            slope_constant = self.stats['sd_y'] * \
                (1 - self.stats['r2_y'][fix]) / self.stats['sd_x'][fix] / \
                (1 - self.stats['r2_x'][fix])
            term.prior.args['sd'].update(beta=value * slope_constant)
        # handle random intercepts
        else:
            index = list(self.stats['r2_y'].index)
            beta = self.stats['sd_y'] * (1 - self.stats['r2_y'][index]) / \
                self.stats['sd_x'][index] / (1 - self.stats['r2_x'][index])
            beta *= value
            beta = np.dot(beta**2, self.stats['mean_x'][index]**2)**.5
            term.prior.args['sd'].update(beta=beta)

    def scale(self, term):

        if term.name == 'Intercept':
            term_type = 'intercept'
        else:
            term_type = term.type_.lower()

        if term.prior is None:
            value = 'value'
        else:
            value = term.prior

        if isinstance(value, string_types):
            value = PriorScaler.names[value]

        term.prior = self.model.default_priors.get(term=term_type)
        getattr(self, '_scale_%s' % term_type)(term, value)
