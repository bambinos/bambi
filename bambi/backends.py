from abc import ABCMeta, abstractmethod
from bambi.external.six import string_types
import numpy as np
import warnings
from bambi.results import PyMC3Results, PyMC3ADVIResults
from bambi.priors import Prior
import theano
try:
    import pymc3 as pm
except:
    warnings.warn("PyMC3 could not be imported. You will not be able to use "
                  "PyMC3 as the back-end for your models.")


class BackEnd(object):

    '''
    Base class for BackEnd hierarchy.
    '''
    __metaclass__ = ABCMeta

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def run(self):
        pass


class PyMC3BackEnd(BackEnd):

    '''
    PyMC3 model-fitting back-end.
    '''

    # Available link functions
    links = {
        'identity': lambda x: x,
        'logit': theano.tensor.nnet.sigmoid,
        'inverse': theano.tensor.inv,
        'log': theano.tensor.log
    }

    def __init__(self):
        self.reset()

    def reset(self):
        '''
        Reset PyMC3 model and all tracked distributions and parameters.
        '''
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

    def build(self, spec, reset=True):
        '''
        Compile the PyMC3 model from an abstract model specification.
        Args:
            spec (Model): A bambi Model instance containing the abstract
                specification of the model to compile.
            reset (bool): if True (default), resets the PyMC3BackEnd instance
                before compiling.
        '''
        if reset:
            self.reset()

        with self.model:

            self.mu = 0.

            for t in spec.terms.values():

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
                        self.mu += pm.math.dot(level_data, u)[:, None]
                else:
                    prefix = 'u_' if t.random else 'b_'
                    n_cols = data.shape[1]
                    coef = self._build_dist(prefix + label, dist_name,
                                            shape=n_cols, **dist_args)
                    self.mu += pm.math.dot(data, coef)[:, None]

            y = spec.y.data
            y_prior = spec.family.prior
            link_f = spec.family.link
            if not callable(link_f):
                link_f = self.links[link_f]
            y_prior.args[spec.family.parent] = link_f(self.mu)
            y_prior.args['observed'] = y
            y_like = self._build_dist(
                spec.y.name, y_prior.name, **y_prior.args)

            self.spec = spec

    def run(self, start=None, method='mcmc', find_map=False, **kwargs):
        '''
        Run the PyMC3 MCMC sampler.
        Args:
            start: Starting parameter values to pass to sampler; see
                pm.sample() documentation for details.
            method: The method to use for fitting the model. By default,
                'mcmc', in which case the PyMC3 sampler will be used.
                Alternatively, 'advi', in which case the model will be fitted
                using  automatic differentiation variational inference as
                implemented in PyMC3.
            find_map (bool): whether or not to use the maximum a posteriori
                estimate as a starting point; passed directly to PyMC3.
            kwargs (dict): Optional keyword arguments passed onto the sampler.
        Returns: A PyMC3ModelResults instance.
        '''
        if method == 'mcmc':
            samples = kwargs.pop('samples', 1000)
            with self.model:
                if start is None and find_map:
                    start = pm.find_MAP()
                self.trace = pm.sample(samples, start=start, **kwargs)
            return PyMC3Results(self.spec, self.trace)

        elif method == 'advi':
            with self.model:
                self.advi_params = pm.variational.advi(start, **kwargs)
            return PyMC3ADVIResults(self.spec, self.advi_params)
