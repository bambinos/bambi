from abc import ABCMeta, abstractmethod
from bambi.external.six import string_types
import numpy as np
import warnings
from bambi.results import ModelResults
from bambi.priors import Prior
import theano
try:
    import pymc3 as pm
except:
    warnings.warn("PyMC3 could not be imported. You will not be able to use "
                  "PyMC3 as the back-end for your models.")


class BackEnd(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def run(self):
        pass


class PyMC3BackEnd(BackEnd):

    # Available link functions
    links = {
        'identity': lambda x: x,
        'logit': theano.tensor.nnet.sigmoid,
        'inverse': theano.tensor.inv,
        'exp': theano.tensor.exp
    }

    def __init__(self):
        self.reset()

    def reset(self):
        self.model = pm.Model()
        self.mu = None
        self.dists = {}
        self.shared_params = {}

    def _build_dist(self, label, dist, **kwargs):
        ''' Build and return a PyMC3 Distribution. '''
        if isinstance(dist, string_types):
            if not hasattr(pm, dist):
                raise ValueError("The Distribution class '%s' was not "
                                 "found in PyMC3." % dist)
            dist = getattr(pm, dist)
        # Inspect all args in case we have hyperparameters
        def _expand_args(k, v, label):
            if isinstance(v, Prior):
                label = '%s_%s' % (label, k)
                return self._build_dist(label, v.name, **v.args)
            return v

        kwargs = {k: _expand_args(k, v, label) for (k, v) in kwargs.items()}
        return dist(label, **kwargs)

    def build(self, model, reset=True):

        if reset:
            self.reset()

        with self.model:

            self.mu = theano.shared(np.zeros((len(model.y.data), 1)))

            for t in model.terms.values():

                data = t.data
                label = t.name
                dist_name = t.prior.name
                dist_args = t.prior.args

                # Effects w/ hyperparameters (i.e., random effects)
                if isinstance(data, dict):
                    for level, level_data in data.items():
                        n_cols = level_data.shape[1]
                        mu_label = 'u_%s_%s' % (label, level)
                        u = self._build_dist(mu_label, dist_name,
                                             shape=n_cols, **dist_args)
                        self.mu += pm.dot(level_data, u)[:, None]
                else:
                    prefix = 'b_' if t.type_ == 'fixed' else 'u_'
                    n_cols = data.shape[1]
                    coef = self._build_dist(prefix + label, dist_name,
                                         shape=n_cols, **dist_args)
                    self.mu += pm.dot(data, coef)[:, None]

            y = model.y.data
            y_prior = model.family.prior
            link_f = self.links[model.family.link]
            y_prior.args[model.family.parent] = link_f(self.mu)
            y_prior.args['observed'] = y
            y_like = self._build_dist('likelihood', y_prior.name, **y_prior.args)

    def run(self, model_spec, start=None, find_map=False, **kwargs):
        samples = kwargs.pop('samples', 1000)
        with self.model:
            if start is None and find_map:
                start = pm.find_MAP()
            self.trace = pm.sample(samples, start=start, **kwargs)
        return ModelResults(model_spec, self.trace)
