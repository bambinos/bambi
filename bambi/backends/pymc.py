from __future__ import absolute_import
from .base import BackEnd
from bambi.priors import Prior
from bambi.results import MCMCResults, PyMC3ADVIResults
from bambi.external.six import string_types
from collections import OrderedDict
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
        'inverse_squared': lambda x: theano.tensor.inv(theano.tensor.sqrt(x)),
        'log': theano.tensor.exp
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
            _offset = pm.Normal(label + '_offset', mu=0, sd=1,
                                shape=kwargs['shape'])
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

                n_cols = t.data.shape[1]

                coef = self._build_dist(spec, label, dist_name,
                                        shape=n_cols, **dist_args)

                if t.random:
                    self.mu += coef[t.group_index][:, None] * t.predictor
                else:
                    self.mu += pm.math.dot(data, coef)[:, None]

            y = spec.y.data
            y_prior = spec.family.prior
            link_f = spec.family.link
            if isinstance(link_f, string_types):
                link_f = self.links[link_f]
            else:
                link_f = link_f
            y_prior.args[spec.family.parent] = link_f(self.mu)
            y_prior.args['observed'] = y
            y_like = self._build_dist(spec, spec.y.name, y_prior.name,
                                      **y_prior.args)

            self.spec = spec

    def _get_transformed_vars(self):
        rvs = self.model.unobserved_RVs
        # identify the variables that pymc3 back-transformed to original scale
        trans = [var.name for var in rvs if isinstance(var, TransformedRV)]
        # find the corresponding transformed variables
        trans = set(x.name for x in rvs if any([t in x.name for t in trans])) \
            - set(trans)
        # add any "centered" random effects to the list
        for varname in [var.name for var in rvs]:
            if re.search(r'_offset$', varname) is not None:
                trans.add(varname)

        return trans

    def run(self, start=None, method='mcmc', init='auto', n_init=50000, **kwargs):
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
            init: Initialization method (see PyMC3 sampler documentation). Currently, this is
                 `'jitter+adapt_diag'`, but this can change in the future.
            n_init: Number of initialization iterations if init = 'advi' or
                'nuts'. Default is kind of in PyMC3 for the kinds of models
                we expect to see run with bambi, so we lower it considerably.
            kwargs (dict): Optional keyword arguments passed onto the sampler.
        Returns: A PyMC3ModelResults instance.
        '''

        if method == 'mcmc':
            samples = kwargs.pop('samples', 1000)
            cores = kwargs.pop('chains', 1)
            with self.model:
                self.trace = pm.sample(samples, start=start, init=init,
                                       n_init=n_init, cores=cores, **kwargs)
            return self._convert_to_results()

        elif method == 'advi':
            with self.model:
                self.advi_params = pm.variational.advi(start, **kwargs)
            return PyMC3ADVIResults(self.spec, self.advi_params,
                                    transformed_vars=self._get_transformed_vars())

    def _convert_to_results(self):
        # grab samples as big, unlabelled array
        # dimensions 0, 1, 2 = samples, chains, variables
        data = np.array([np.array([np.atleast_2d(x.T).T[:, i]
                         for x in self.trace._straces[j].samples.values()
                         for i in range(np.atleast_2d(x.T).T.shape[1])])
                         for j in range(len(self.trace._straces))])
        data = np.swapaxes(np.swapaxes(data, 0, 1), 0, 2)

        # arrange var_shapes dictionary in same order as samples dictionary
        shapes = OrderedDict()
        for key in self.trace._straces[0].samples:
            shapes[key] = self.trace._straces[0].var_shapes[key]

        # grab info necessary for making samplearray pretty
        names = list(shapes.keys())
        dims = list(shapes.values())
        def get_levels(key, value):
            if len(value):
                # fixed effects
                if not self.spec.terms[re.sub(r'_offset$', '', key)].random:
                    return self.spec.terms[key].levels
                # random effects
                else:
                    re1 = re.match(r'(.+)(?=_offset)(_offset)', key)
                    # handle "centered" terms
                    if re1 is None:
                        return [key.split('|')[0]+'|'+x
                                for x in self.spec.terms[key].levels]
                    # handle "non-centered" terms
                    else:
                        return ['{}|{}_offset{}'.format(key.split('|')[0],
                                *re.match(r'^(.+)(\[.+\])$', x).groups())
                                for x in self.spec.terms[re1.group(1)].levels]
            else:
                return [key]
        levels = sum([get_levels(k, v) for k, v in shapes.items()], [])

        # instantiate
        return MCMCResults(model=self.spec, data=data, names=names, dims=dims,
                           levels=levels,
                           transformed_vars=self._get_transformed_vars())
