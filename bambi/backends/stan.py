from .base import BackEnd
from bambi.external.six import string_types
import numpy as np
from bambi.results import MCMCResults
from bambi.priors import Prior
import pymc3 as pm
try:
    import pystan as ps
except:
    ps = None


class StanBackEnd(BackEnd):
    '''
    Stan/PyStan model-fitting back-end.
    '''

    dists = {
        'Normal': ('normal', ['mu', 'sd']),
        'Cauchy': ('cauchy', ['alpha', 'beta'])
    }

    def __init__(self):
        if ps is None:
            raise ImportError("Could not import PyStan; please make sure it's "
                              "installed.")
        self.reset()

    def reset(self):
        '''
        Reset Stan model and all tracked distributions and parameters.
        '''
        self.parameters = []
        self.transformed_parameters = []
        self.data = []
        self.transformed_data = []
        self.X = {}
        self.model = []
        self.mu = []
        self._suppress_vars = []  # variables to suppress in output

    # def _build_dist(self, label, dist, **kwargs):
    #     ''' Build and return a Stan Distribution. '''
    #     if dist not in self.dists:
    #         raise ValueError("There is no distribution named '%s' in Stan."
    #                          % dist)

    #     def _expand_args(k, v, label):
    #         if isinstance(v, Prior):
    #             label = '%s_%s' % (label, k)
    #             return self._build_dist(label, v.name, **v.args)
    #         return v

    #     kwargs = {k: _expand_args(k, v, label) for (k, v) in kwargs.items()}
    #     return dist(label, **kwargs)

    def _map_dist(self, dist, **kwargs):
        if dist not in self.dists:
            raise ValueError("There is no distribution named '%s' in Stan."
                             % dist)
        dist_name, dist_params = self.dists[dist]
        missing = set(dist_params) - set(list(kwargs.keys()))
        if missing:
            raise ValueError("The following mandatory parameters of the %s "
                             "distribution are missing: %s." % (dist, missing))
        dp = [kwargs[p] for p in dist_params]
        dp = [float(p[0]) if hasattr(p, 'shape') else p for p in dp]
        return '%s(%s);' % (dist_name, ', '.join([str(p) for p in dp]))

    def build(self, spec, reset=True):
        '''
        Compile the Stan model from an abstract model specification.
        Args:
            spec (Model): A bambi Model instance containing the abstract
                specification of the model to compile.
            reset (bool): if True (default), resets the StanBackEnd instance
                before compiling.
        '''
        if reset:
            self.reset()

        n_cases = len(spec.y.data)

        for t in spec.terms.values():

            data = t.data
            label = t.name
            dist_name = t.prior.name
            dist_args = t.prior.args

            # Effects with hyperpriors (i.e., random effects)
            if isinstance(data, dict):
                pass
                # for level, level_data in data.items():
                #     n_cols = level_data.shape[1]
                #     mu_label = 'u_%s_%s' % (label, level)
                #     u = self._build_dist(mu_label, dist_name,
                #                          shape=n_cols, **dist_args)
                #     self.mu += pm.math.dot(level_data, u)[:, None]
            else:
                prefix = 'u_' if t.random else 'b_'
                n_cols = data.shape[1]
                name = prefix + label

                if n_cols == 1:
                    stan_par = 'real'
                    stan_data = 'vector[%d]' % (n_cases)
                else:
                    stan_par = 'vector[%d]' % n_cols
                    stan_data = 'matrix[%d, %d]' % (n_cases, n_cols)

                # Add to Stan model
                self.data.append('%s %s_data;' % (stan_data, name))
                self.parameters.append('%s %s;' % (stan_par, name))
                _dist = self._map_dist(dist_name, **dist_args)
                self.model.append('%s ~ %s;' % (name, _dist))
                self.mu.append('%s_data * %s' % (name, name))

                # Add to PyStan input data
                self.X[name + '_data'] = data.squeeze()

        # yhat
        yhat = 'yhat = ' + ' + '.join(self.mu) + ';'
        self.transformed_parameters.append('vector[%d] yhat;' % n_cases)
        self.transformed_parameters.append(yhat)
        self._suppress_vars.append('yhat')

        # y
        self.data.append('vector[%d] y;' % n_cases)
        self.parameters.append('real<lower=0> sigma;')
        self.model.append('y ~ normal(yhat, sigma);')
        self.X['y'] = spec.y.data.squeeze()

        # Construct the stan script
        def format_block(name):
            key = name.replace(' ', '_')
            els = ''.join(['\t%s\n' % e for e in getattr(self, key)])
            return '%s {\n%s}\n' % (name, els)

        blocks = ['data', 'transformed data', 'parameters',
                  'transformed parameters', 'model']
        self.model_code = ''.join([format_block(bl) for bl in blocks])
        self.spec = spec
        self.stan_model = ps.StanModel(model_code=self.model_code)

    def run(self, samples=1000, chains=1, **kwargs):
        '''
        Run the Stan sampler.
        Args:
            samples (int): Number of samples to obtain (in each chain).
            chains (int): Number of chains to use.
            kwargs (dict): Optional keyword arguments passed onto the PyStan
                StanModel.sampling() call.
        Returns: A PyMC3ModelResults instance.
        '''
        self.fit = self.stan_model.sampling(data=self.X, iter=samples,
                                            chains=chains, **kwargs)
        return self._convert_to_multitrace()

    def _convert_to_multitrace(self):
        ''' Convert a PyStan stanfit4model object to a PyMC3 MultiTrace. '''

        result = self.fit.extract(permuted=True, inc_warmup=True)
        varnames = list(result.keys())
        straces = []
        n_chains = self.fit.sim['chains']
        n_samples = self.fit.sim['iter'] - self.fit.sim['warmup']
        dummy = pm.Model()  # Need to fool the NDArray

        for i in range(n_chains):
            _strace = pm.backends.NDArray(model=dummy)
            _strace.setup(n_samples, i)
            _strace.varnames = varnames
            for k, par in result.items():
                start = i * n_samples
                _strace.samples[k] = par[start:(start+n_samples)]
            straces.append(_strace)

        self.trace = pm.backends.base.MultiTrace(straces)
        return MCMCResults(self.spec, self.trace, self._suppress_vars)

    def plot_priors(self):
        raise ValueError("Prior plotting has not been implemented yet for "
                         "the Stan back-end; sorry!")
