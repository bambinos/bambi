import numpy as np
from os.path import dirname, join
from six import string_types
from copy import deepcopy
import json
import re


class Family(object):

    def __init__(self, name, prior, link, parent):
        self.name = name
        self.prior = prior
        self.link = link
        self.parent = parent


class Prior(object):

    def __init__(self, name, **kwargs):
        self.name = name
        self.args = {}
        self.update(**kwargs)

    def update(self, **kwargs):
        self.args.update(kwargs)


class PriorFactory(object):

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

        # def _expand(v):
        #     if isinstance(v, dict):
        #         return self._get_prior(v['name'], v['args'])
        #     if isinstance(v, string_types):
        #         return self._get_prior(**self.dists[v.lower()])
        #     return v

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

        # args = {k: expand(v) for (k, v) in args.items()}
        # return dict(name=name, args=args)

    def get(self, dist=None, term=None, family=None, **kwargs):
        if dist is not None:
            if dist not in self.dists:
                raise ValueError(
                    "'%s' is not a valid distribution name." % dist)
            return self._get_prior(**self.dists[dist])
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
