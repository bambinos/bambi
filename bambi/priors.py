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
