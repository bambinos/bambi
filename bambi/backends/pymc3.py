from .base import BackEnd
from bambi.external.six import string_types
import numpy as np
import pandas as pd
from bambi.results import MCMCResults, PyMC3ADVIResults
from bambi.priors import Prior
import matplotlib.pyplot as plt
import theano
try:
    import pymc3 as pm
    from pymc3.model import FreeRV, TransformedRV
except:
    pm = None


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
        if pm is None:
            raise ImportError("Could not import PyMC3; please make sure it's "
                              "installed.")
        self.reset()

    def reset(self):
        '''
        Reset PyMC3 model and all tracked distributions and parameters.
        '''
        self.model = pm.Model()
        self.mu = None

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
            with self.model:
                if start is None and find_map:
                    start = pm.find_MAP()
                self.trace = pm.sample(samples, start=start, init=init,
                                       n_init=n_init, **kwargs)
            return MCMCResults(self.spec, self.trace,
                               transformed_vars=self._get_transformed_vars())

        elif method == 'advi':
            with self.model:
                self.advi_params = pm.variational.advi(start, **kwargs)
            return PyMC3ADVIResults(self.spec, self.advi_params,
                                    transformed_vars=self._get_transformed_vars())

    def plot_priors(self, model):
        # Currently this only supports plotting priors for fixed effects
        if not model.built:
            raise ValueError("Cannot plot priors until model is built!")

        with pm.Model():
            # get priors for fixed fx, separately for each level of each predictor
            dists = []
            for t in model.fixed_terms.values():
                for i,l in enumerate(t.levels):
                    params = {k: v[i % len(v)] \
                        if isinstance(v, np.ndarray) else v
                        for k,v in t.prior.args.items()}
                    dists += [getattr(pm, t.prior.name)(l, **params)]

            # get priors for random effect SDs
            for t in model.random_terms.values():
                prior = t.prior.args['sd'].name
                params = t.prior.args['sd'].args
                dists += [getattr(pm, prior)(t.name+'_sd', **params)]

            # add priors on Y params if applicable
            y_prior = [(k,v) for k,v in model.y.prior.args.items()
                if isinstance(v, Prior)]
            if len(y_prior):
                for p in y_prior:
                    dists += [getattr(pm, p[1].name)('_'.join([model.y.name,
                        p[0]]), **p[1].args)]

            # make the plot!
            p = float(len(dists))
            fig, axes = plt.subplots(int(np.ceil(p/2)), 2,
                figsize=(12,np.ceil(p/2)*2))
            # in case there is only 1 row
            if int(np.ceil(p/2))<2: axes = axes[None,:]
            for i,d in enumerate(dists):
                dist = d.distribution if isinstance(d, FreeRV) else d
                samp = pd.Series(dist.random(size=1000).flatten())
                samp.plot(kind='hist', ax=axes[divmod(i,2)[0], divmod(i,2)[1]],
                    normed=True)
                samp.plot(kind='kde', ax=axes[divmod(i,2)[0], divmod(i,2)[1]],
                    color='b')
                axes[divmod(i,2)[0], divmod(i,2)[1]].set_title(d.name)
            fig.tight_layout()

        return axes
