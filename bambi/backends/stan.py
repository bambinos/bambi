from __future__ import absolute_import
from .base import BackEnd
from bambi.priors import Prior
from bambi.results import MCMCResults, SampleArray
import numpy as np
import pymc3 as pm
import re
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
                       'bounds': '<lower=0>'},
        'HalfCauchy': {'name': 'cauchy', 'args': ['0', '#beta'],
                       'bounds': '<lower=0>'},
        'Uniform': {'name': 'uniform', 'args': ['#lower', '#upper']},
        'Flat': {'name': None, 'args': []},
        'HalfFlat':  {'name': None, 'args': [], 'bounds': '<lower=0>'}
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
        self.mu_cont = []
        self.mu_cat = []
        self._suppress_vars = []  # variables to suppress in output
        # Stan uses limited set for variable names, so track variable names
        # we may need to simplify for the model code and then sub back later.
        self._original_names = {}

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
        self.data.append('int<lower=1> N;')
        self.X['N'] = n_cases

        def _sanitize_name(name):
            ''' Stan only allows alphanumeric chars and underscore, and
            variable names must begin with a letter. So replace all invalid
            chars with '_', prepend with 'b_', and track for later
            re-substitution. '''
            if name in self._original_names:
                return name
            clean = 'b_' + re.sub('[^a-zA-Z0-9\_]+', '_', name)
            self._original_names[clean] = name
            return clean

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

            # Flat/HalfFlat/undefined priors are handled separately
            if dist_name is None:
                return None, dist_bounds

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
            dp = [float(p.ravel()[0]) if isinstance(p, np.ndarray) else p
                  for p in dp]

            dist_term = '%s(%s)' % (
                dist_name, ', '.join([str(p) for p in dp]))

            return dist_term, dist_bounds

        def _add_data(name, data):
            ''' Add all model components that directly touch or relate to data.
            '''
            if data.shape[1] == 1:
                if n_cols > 1:
                    stan_data = 'int %s[N];'
                else:
                    stan_data = 'vector[N] %s;'
            else:
                stan_data = ('matrix[N, %d]' % (n_cols)) + ' %s;'
            data_name = _sanitize_name('%s_data' % name)
            var_name = _sanitize_name(name)
            self.data.append(stan_data % data_name)
            self.X[data_name] = data.squeeze()

            if data.shape[1] == 1 and n_cols > 1:
                code = '%s[%s[n]]' % (var_name, data_name)
                self.mu_cat.append(code)
            else:
                self.mu_cont.append('%s * %s' % (data_name, var_name))

        def _add_parameters(name, dist_name, n_cols, **dist_args):
            ''' Add all model components related to latent parameters. We
            handle these separately from the data components, as the parameters
            can have nested specifications (in the case of random effects). '''

            def _expand_args(k, v, name):
                if isinstance(v, Prior):
                    name = _sanitize_name('%s_%s' % (name, k))
                    return _add_parameters(name, v.name, 1, **v.args)
                return v

            kwargs = {k: _expand_args(k, v, name)
                      for (k, v) in dist_args.items()}
            _dist, _bounds = _map_dist(dist_name, **kwargs)

            if n_cols == 1:
                stan_par = 'real'
            else:
                stan_par = 'vector[%d]' % n_cols

            var_name = _sanitize_name(name)
            self.parameters.append('%s%s %s;' % (stan_par, _bounds, var_name))
            if _dist is not None:
                self.model.append('%s ~ %s;' % (var_name, _dist))
            return name

        for t in spec.terms.values():

            data = t.data
            label = t.name
            dist_name = t.prior.name
            dist_args = t.prior.args

            if t._reduced_data is not None:
                data = t._reduced_data
                if data.max() > 1:
                    data += 1  # Stan indexes from 1, not 0
                n_cols = data.max()
            else:
                n_cols = data.shape[1]

            # Add to Stan model
            _add_data(label, data)
            _add_parameters(label, dist_name, n_cols, **dist_args)

        # yhat
        self.transformed_parameters.append('vector[N] yhat;')
        if self.mu_cont:
            yhat_cont = 'yhat = %s;' % ' + '.join(self.mu_cont)
            self.transformed_parameters.append(yhat_cont)
        else:
            self.mu_cat.insert(0, '0')

        if self.mu_cat:
            loops = ('for (n in 1:N)\n'
                    '\t\tyhat[n] = %s' % ' + '.join(self.mu_cat) + ';\n\t')
            self.transformed_parameters.append(loops)
        self._suppress_vars.append('yhat')

        # y
        self.data.append('vector[N] y;')
        self.parameters.append('real<lower=0> {}_sd;'.format(spec.y.name))
        self.model.append('y ~ normal(yhat, {}_sd);'.format(spec.y.name))
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
        _res = self.fit.extract(permuted=True, inc_warmup=True)

        # Substitute original var names where we had to strip chars for Stan
        result = {}
        for k, v in _res.items():
            if k in self._original_names:
                k = self._original_names[k]
            result[k] = v

        varnames = list(result.keys())
        straces = []
        n_chains = self.fit.sim['chains']
        n_samples = self.fit.sim['iter'] - self.fit.sim['warmup']
        dummy = pm.Model()  # Need to fool the NDArray

        for i in range(n_chains):
            _strace = pm.backends.NDArray(model=dummy)
            _strace.setup(n_samples, i)
            _strace.draw_idx = _strace.draws
            _strace.varnames = varnames
            for k, par in result.items():
                start = i * n_samples
                _strace.samples[k] = par[start:(start+n_samples)]
            straces.append(_strace)

        self.trace = pm.backends.base.MultiTrace(straces)
        return MCMCResults(self.spec, self.trace, self._suppress_vars)

    def _convert_to_samplearray(self):
        # grab samples as big, unlabelled array
        # dimensions 0, 1, 2 = samples, chains, variables
        data = self.fit.extract(permuted=False, inc_warmup=True)     

        # grab info necessary for making samplearray pretty
        names = [self._original_names[x] \
                 if x in self._original_names else x \
                 for x in self.fit.sim['pars_oi']]
        dims = [tuple(x) for x in self.fit.sim['dims_oi']]
        def replace_name(name):
            tname = re.sub(r'\[.+\]$', '', name)
            if tname in self._original_names:
                lev = re.search(r'\[(.+)\]$', name)
                if lev is not None:
                    tname = self._original_names[tname]
                    lev = self.model.terms[tname].levels[int(lev.group(1))]
                    return '{}|{}'.format(tname.split('|')[0], lev)
                else:
                    return self._original_names[name]
            else:
                return name
        levels = [replace_name(x) for x in self.fit.sim['fnames_oi']]

        return SampleArray(data=data, names=names, dims=dims, levels=levels)
