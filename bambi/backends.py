from abc import ABCMeta, abstractmethod
from six import string_types
import numpy as np
import warnings
from bambi.priors import default_priors
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

    def __init__(self):
        self.reset()

    def reset(self):
        self.model = pm.Model()
        self.mu = 0.
        self.dists = {}
        self.shared_params = {}

    def _build_dist(self, label, dist, **kwargs):
        ''' Build and return a PyMC3 Distribution. '''
        if isinstance(dist, string_types):
            if not hasattr(pm, dist):
                raise ValueError("The Distribution class '%s' was not "
                                 "found in PyMC3." % dist)
            dist = getattr(pm, dist)
        return dist(label, **kwargs)

    def build(self, model, reset=True):

        if reset:
            self.reset()

        with self.model:

            for t in model.terms.values():

                data = t.data
                label = t.name
                dist_name = t.prior['name']
                dist_args = t.prior['args']

                # Random effects
                if t.type_ == 'random':

                    # User can pass sigma specification in sigma_kws.
                    # If not provided, default to HalfCauchy with beta = 10.
                    try:
                        sigma_dist_name = t.prior['sigma']['name']
                        sigma_dist_args = t.prior['sigma']['args']
                    except:
                        sigma_dist_name = 'HalfCauchy'
                        sigma_dist_args = {'beta': 10}

                    if isinstance(data, dict):
                        for level, level_data in data.items():
                            n_cols = level_data.shape[1]
                            sigma_label = 'sigma_%s_%s' % (label, level)
                            sigma = self._build_dist(sigma_label, sigma_dist_name,
                                                     **sigma_dist_args)
                            mu_label = 'u_%s_%s' % (label, level)
                            dist_args['sd'] = sigma
                            u = self._build_dist(mu_label, dist_name,
                                                 shape=n_cols, **dist_args)
                            self.mu += pm.dot(level_data, u)[:, None]
                    else:
                        n_cols = data.shape[1]
                        sigma = self._build_dist('sigma_' + label, sigma_dist_name,
                                                 **sigma_dist_args)
                        dist_args['sd'] = sigma
                        u = self._build_dist('u_' + label, dist_name,
                                             shape=n_cols, **dist_args)
                        self.mu += pm.dot(data, u)[:, None]

                # Fixed effects
                else:
                    n_cols = data.shape[1]
                    b = self._build_dist('b_' + label, dist_name,
                                         shape=n_cols, **dist_args)
                    self.mu += pm.dot(data, b)[:, None]

            # TODO: accept sigma params as an argument
            sigma_params = default_priors['sigma']
            sigma = self._build_dist('sigma', sigma_params['name'],
                                     **sigma_params['args'])
            y = model.y.data
            y_obs = pm.Normal('y_pred', mu=self.mu, sd=sigma, observed=y)

    def run(self, start=None, find_map=False, **kwargs):
        samples = kwargs.pop('samples', 1000)
        with self.model:
            if start is None and find_map:
                start = pm.find_MAP()
            self.trace = pm.sample(samples, start=start, **kwargs)
