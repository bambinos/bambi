from __future__ import absolute_import
from .base import BackEnd
from bambi.priors import Prior
from bambi.results import MCMCResults, PyMC3ADVIResults
from bambi.external.six import string_types
import theano
import pymc3 as pm
import numpy as np
import re
from pymc3.model import TransformedRV


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

    dists = {
        'HalfFlat': pm.Bound(pm.Flat, lower=0)
    }

    def __init__(self):

        self.reset()

    def reset(self):
        '''
        Reset PyMC3 model and all tracked distributions and parameters.
        '''
        self.model = pm.Model()
        self.mu = None
        self.par_groups = {}

    def _build_dist(self, spec, label, dist, **kwargs):
        ''' Build and return a PyMC3 Distribution. '''
        if isinstance(dist, string_types):
            if hasattr(pm, dist):
                dist = getattr(pm, dist)
            elif dist in self.dists:
                dist = self.dists[dist]
            else:
                raise ValueError("The Distribution class '%s' was not "
                                 "found in PyMC3 or the PyMC3BackEnd." % dist)

        # Inspect all args in case we have hyperparameters
        def _expand_args(k, v, label):
            if isinstance(v, Prior):
                label = '%s_%s' % (label, k)
                return self._build_dist(spec, label, v.name, **v.args)
            return v

        kwargs = {k: _expand_args(k, v, label) for (k, v) in kwargs.items()}

        # Non-centered parameterization for hyperpriors
        if spec.noncentered and 'sd' in kwargs and 'observed' not in kwargs \
                and isinstance(kwargs['sd'], pm.model.TransformedRV):
            old_sd = kwargs['sd']
            _offset = pm.Normal(label + '_offset', mu=0, sd=1, shape=kwargs['shape'])
            return pm.Deterministic(label, _offset * old_sd)

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

                if t._reduced_data is not None:
                    n_cols = t._reduced_data.max() + 1
                else:
                    n_cols = data.shape[1]

                coef = self._build_dist(spec, label, dist_name,
                                        shape=n_cols, **dist_args)

                if t._reduced_data is not None:
                    self.mu += coef[t._reduced_data][:, None]
                else:
                    self.mu += pm.math.dot(data, coef)[:, None]

            y = spec.y.data
            y_prior = spec.family.prior
            link_f = spec.family.link
            if not callable(link_f):
                link_f = self.links[link_f]
            y_prior.args[spec.family.parent] = link_f(self.mu)
            y_prior.args['observed'] = y
            y_like = self._build_dist(spec, spec.y.name, y_prior.name,
                                      **y_prior.args)

            self.spec = spec

    def _get_transformed_vars(self):
        # here we determine which variables have been internally transformed
        # (e.g., sd_log). transformed vars are actually 'untransformed' from
        # the PyMC3 model's perspective, and it's the backtransformed (e.g.,
        # sd) variable that is 'transformed'. So the logic of this is to find
        # and remove 'untransformed' variables that have a 'transformed'
        # counterpart, in that 'untransformed' varname is the 'transformed'
        # varname plus some suffix (such as '_log' or '_interval')
        rvs = self.model.unobserved_RVs
        trans = set(var.name for var in rvs if isinstance(var, TransformedRV))
        untrans = set(var.name for var in rvs) - trans
        untrans = set(x for x in untrans if not any([t in x for t in trans]))
        return [x for x in self.trace.varnames if x not in (trans | untrans)]

    def run(self, start=None, method='mcmc', init=None, n_init=10000,
            find_map=False, **kwargs):
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
            init: Initialization method (see PyMC3 sampler documentation).
                In PyMC3, this defaults to 'advi', but we set it to None.
            n_init: Number of initialization iterations if init = 'advi' or
                'nuts'. Default is kind of in PyMC3 for the kinds of models
                we expect to see run with bambi, so we lower it considerably.
            find_map (bool): whether or not to use the maximum a posteriori
                estimate as a starting point; passed directly to PyMC3.
            kwargs (dict): Optional keyword arguments passed onto the sampler.
        Returns: A PyMC3ModelResults instance.
        '''

        if method == 'mcmc':
            samples = kwargs.pop('samples', 1000)
            njobs = kwargs.pop('chains', 1)
            with self.model:
                if start is None and find_map:
                    start = pm.find_MAP()
                self.trace = pm.sample(samples, start=start, init=init,
                                       n_init=n_init, njobs=njobs, **kwargs)
            return self._convert_to_results()

        elif method == 'advi':
            with self.model:
                self.advi_params = pm.variational.advi(start, **kwargs)
            return PyMC3ADVIResults(self.spec, self.advi_params,
                                    transformed_vars=self._get_transformed_vars())

    def _convert_to_results(self):
        # grab samples as big, unlabelled array
        # dimensions 0, 1, 2 = samples, chains, variables
        data = np.array([np.array([np.atleast_2d(x.T).T[:,i] \
                          for x in self.trace._straces[j].samples.values() \
                          for i in range(np.atleast_2d(x.T).T.shape[1])])
                for j in range(len(self.trace._straces))])
        data = np.swapaxes(np.swapaxes(data, 0, 1), 0, 2)

        # grab info necessary for making samplearray pretty
        names = list(self.trace._straces[0].var_shapes.keys())
        dims = list(self.trace._straces[0].var_shapes.values())
        def get_levels(key, value):
            if len(value):
                if not self.spec.terms[re.sub(r'_offset$', '', key)].random:
                    return self.spec.terms[key].levels
                else:
                    return [key.split('|')[0]+'|'+x \
                        for x in self.spec.terms[re.sub(r'_offset$', '', key)].levels]
            else:
                return [key]
        levels = sum([get_levels(k, v) \
            for k, v in self.trace._straces[0].var_shapes.items()], [])

        # instantiate
        return MCMCResults(model=self.spec, data=data, names=names, dims=dims,
            levels=levels, transformed_vars=self._get_transformed_vars())
