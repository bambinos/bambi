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
        'Normal': {'name': 'normal', 'args': ['#mu', '#sd']},
        'Cauchy': {'name': 'cauchy', 'args': ['#alpha', '#beta']},
        'HalfNormal': {'name': 'normal', 'args': ['0', '#sd'],
                       'bounds':'<lower=0>'},
        'HalfCauchy': {'name': 'cauchy', 'args': ['0', '#beta'],
                       'bounds': '<lower=0>'}
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

        def _map_dist(dist, **kwargs):
            ''' Maps PyMC3 distribution names and attrs in the Prior object
            to the corresponding Stan names and argument order. '''
            if dist not in self.dists:
                raise ValueError("There is no distribution named '%s' "
                                 "in Stan." % dist)

            stan_dist = self.dists[dist]
            dist_name = stan_dist['name']
            dist_args = stan_dist['args']
            dist_bounds = stan_dist.get('bounds', '')

            lookup_args = [a[1:] for a in dist_args if a.startswith('#')]
            missing = set(lookup_args) - set(list(kwargs.keys()))
            if missing:
                raise ValueError("The following mandatory parameters of "
                                 "the %s distribution are missing: %s." 
                                 % (dist, missing))

            # Named arguments to take from the Prior object are denoted with
            # a '#'; otherwise we take the value in the self.dists dict as-is.
            dp = [kwargs[p[1:]] if p.startswith('#') else p for p in dist_args]

            # Sometimes we get numpy arrays at this stage, so convert to float
            dp = [float(p[0]) if isinstance(p, np.ndarray) else p for p in dp]

            dist_term = '%s(%s);' % (dist_name, ', '.join([str(p) for p in dp]))

            return dist_term, dist_bounds

        def _add_data(name, data):
            ''' Add all model components that directly touch or relate to data.
            '''
            if n_cols == 1:
                stan_data = 'vector[%d]' % (n_cases)
            else:
                stan_data = 'matrix[%d, %d]' % (n_cases, n_cols)
            self.data.append('%s %s_data;' % (stan_data, name))
            self.X[name + '_data'] = data.squeeze().astype(float)
            self.mu.append('%s_data * %s' % (name, name))

        def _add_parameters(name, dist_name, n_cols, **dist_args):
            ''' Add all model components related to latent parameters. We
            handle these separately from the data components, as the parameters
            can have nested specifications (in the case of random effects). '''

            def _expand_args(k, v, name):
                if isinstance(v, Prior):
                    name = '%s_%s' % (name, k)
                    return _add_parameters(name, v.name, 1, **v.args)
                return v

            kwargs = {k: _expand_args(k, v, name) for (k, v) in dist_args.items()}
            _dist, _bounds = _map_dist(dist_name, **kwargs)

            if n_cols == 1:
                stan_par = 'real'
            else:
                stan_par = 'vector[%d]' % n_cols

            self.parameters.append('%s%s %s;' % (stan_par, _bounds, name))
            self.model.append('%s ~ %s;' % (name, _dist))
            return name

        for t in spec.terms.values():

            data = t.data
            label = t.name
            dist_name = t.prior.name
            dist_args = t.prior.args

            # Effects with hyperpriors (i.e., random effects)
            if isinstance(data, dict):
                for level, level_data in data.items():
                    n_cols = level_data.shape[1]
                    name = 'u_%s_%s' % (label, level)
                    _add_data(name, level_data)
                    _add_parameters(name, dist_name, n_cols, **dist_args)

            else:
                prefix = 'u_' if t.random else 'b_'
                n_cols = data.shape[1]
                name = prefix + label

                # Add to Stan model
                _add_data(name, data)
                _add_parameters(name, dist_name, n_cols, **dist_args)

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
