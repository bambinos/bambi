from __future__ import absolute_import
from .base import BackEnd
from bambi.priors import Prior
from bambi.results import MCMCResults
import numpy as np
import pymc3 as pm
import re
try:
    import pystan as ps
except:
    ps = None
from six import string_types


class StanBackEnd(BackEnd):

    '''
    Stan/PyStan model-fitting back-end.
    '''

    # distribution names and arg names should match those in priors.json,
    dists = {
        'Normal': {'name': 'normal', 'args': ['#mu', '#sd']},
        'Bernoulli': {'name': 'bernoulli', 'args': ['#p']},
        'Poisson': {'name': 'poisson', 'args': ['#mu']},
        'Cauchy': {'name': 'cauchy', 'args': ['#alpha', '#beta']},
        'HalfNormal': {'name': 'normal', 'args': ['0', '#sd'],
                       'bounds': '<lower=0>'},
        'HalfCauchy': {'name': 'cauchy', 'args': ['0', '#beta'],
                       'bounds': '<lower=0>'},
        # for Uniform, the bounds are the parameters. _map_dist fills these in
        'Uniform': {'name': 'uniform', 'args': ['#lower', '#upper'],
                    'bounds': '<lower={}, upper={}>'},
        'Flat': {'name': None, 'args': []},
        'HalfFlat':  {'name': None, 'args': [], 'bounds': '<lower=0>'}
    }

    # maps from spec.family.name to 'dists' dictionary
    # and gives the Stan variable type for the response
    families = {
        'gaussian': {'name': 'Normal', 'type': float,
            'format': ['vector[N]', '']},
        'bernoulli': {'name': 'Bernoulli', 'type': int,
            'format': ['int', '[N]']},
        'poisson': {'name': 'Poisson', 'type': int,
            'format': ['int', '[N]']}
    }

    # maps from spec.family.link to Stan inverse link function name
    links = {
        'identity': '',
        'logit': 'inv_logit',
        'log': 'exp',
        'inverse_squared': 'inv_sqrt'
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
        self.expressions = []
        self.data = []
        self.transformed_data = []
        self.X = {}
        self.model = []
        self.mu_cont = []
        self.mu_cat = []
        self._original_names = {}
        # variables to suppress in output. Stan uses limited set for variable
        # names, so track variable names we may need to simplify for the model
        # code and then sub back later.
        self._suppress_vars = ['yhat', 'lp__']

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
            variable names must begin with a letter. Additionally, Stan
            reserves a few hundred strings that can't be used as variable
            names. So to play it safe, we replace all invalid chars with '_',
            and prepend all variables with 'b_'. We substitute the original
            names back in later. '''
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

            dist_term = '%s(%s)' % (dist_name, ', '.join([str(p) for p in dp]))

            # handle Uniform variables, for which the bounds are the parameters
            if dist_name=='uniform':
                dist_bounds = dist_bounds.format(*dp)

            return dist_term, dist_bounds

        def _add_data(name, data, term):
            ''' Add all model components that directly touch or relate to data.
            '''
            if data.shape[1] == 1:
                # For random effects, index into grouping variable
                if n_cols > 1:
                    index_name = _sanitize_name('%s_grp_ind' % name)
                    self.data.append('int %s[N];' % index_name)
                    self.X[index_name] = t.group_index + 1  # 1-based indexing
                predictor = 'vector[N] %s;'
            else:
                predictor = ('matrix[N, %d]' % (n_cols)) + ' %s;'

            data_name = _sanitize_name('%s_data' % name)
            var_name = _sanitize_name(name)
            self.data.append(predictor % data_name)
            self.X[data_name] = data.squeeze()

            if data.shape[1] == 1 and n_cols > 1:
                code = '%s[%s[n]] * %s[n]' % (var_name, index_name, data_name)
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

            # non-centered parameterization
            if spec.noncentered and 'sd' in kwargs and \
                    isinstance(kwargs['sd'], string_types):
                offset_name = _sanitize_name(name + '_offset')
                offset = 'vector[%d] %s;' % (n_cols, offset_name)
                self.parameters.append(offset)
                self.model.append('%s ~ normal(0, 1);' % offset_name)
                self.transformed_parameters.append('%s%s %s;' % (stan_par,
                                                                 _bounds,
                                                                 var_name))
                trans = '%s = multiply(%s, %s);' % (var_name, offset_name,
                                                    kwargs['sd'])
                self.expressions.append(trans)

            else:
                self.parameters.append('%s%s %s;' % (stan_par, _bounds,
                                                     var_name))
                if _dist is not None:
                    self.model.append('%s ~ %s;' % (var_name, _dist))

            return name

        for t in spec.terms.values():

            data = t.data
            label = t.name
            dist_name = t.prior.name
            dist_args = t.prior.args

            n_cols = data.shape[1]

            if t.random:
                data = t.predictor

            # Add to Stan model
            _add_data(label, data, t)
            _add_parameters(label, dist_name, n_cols, **dist_args)

        # yhat
        self.transformed_parameters.append('vector[N] yhat;')
        if self.mu_cont:
            yhat_cont = 'yhat = %s;' % ' + '.join(self.mu_cont)
            self.expressions.append(yhat_cont)
        else:
            self.mu_cat.insert(0, '0')

        if self.mu_cat:
            loops = ('for (n in 1:N)\n\t\tyhat[n] = yhat[n] + %s'
                     % ' + '.join(self.mu_cat) + ';\n\t')
            self.expressions.append(loops)

        # Add expressions that go in transformed parameter block (they have
        # to come after variable definitions)
        self.transformed_parameters += self.expressions

        # add response variable (y)
        _response_format = self.families[spec.family.name]['format']
        self.data.append('{} y{};'.format(*_response_format))

        # add response distribution parameters other than the location
        # parameter
        for k, v in spec.family.prior.args.items():
            if k != spec.family.parent and isinstance(v, Prior):
                _bounds = _map_dist(v.name, **v.args)[1]
                _param = 'real{} {}_{};'.format(_bounds, spec.y.name, k)
                self.parameters.append(_param)

        # specify the response distribution
        _response_dist = self.families[spec.family.name]['name']
        _response_args = '{}(yhat)'.format(self.links[spec.family.link])
        _response_args = {spec.family.parent: _response_args}
        for k, v in spec.family.prior.args.items():
            if k != spec.family.parent:
                _response_args[k] = '{}_{}'.format(spec.y.name, k) \
                    if isinstance(v, Prior) else str(v)
        _dist = _map_dist(_response_dist, **_response_args)[0]
        self.model.append('y ~ {};'.format(_dist))

        # add the data
        _response_type = self.families[spec.family.name]['type']
        self.X['y'] = spec.y.data.astype(_response_type).squeeze()

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
        return self._convert_to_results()

    def _convert_to_results(self):
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
                    # detect and handle offset terms
                    rgx = re.search(r'_offset$', tname)
                    if rgx is not None:
                        tname = tname[:rgx.start()]
                    # Note that PyStan uses 1-based indexing
                    lev = self.spec.terms[tname].levels[int(lev.group(1)) - 1]
                    split = tname.split('|')
                    # random effects
                    if len(split) == 2:
                        if rgx is None:
                            return '{}|{}'.format(split[0], lev)
                        else:
                            lev = re.search(r'(\[.+\])', lev).group(0)
                            return '{}|{}_offset{}'.format(*(split + [lev]))
                    # fixed effects
                    elif len(split) == 1:
                        return lev
                else:
                    return self._original_names[name]
            else:
                return name
        levels = [replace_name(x) for x in self.fit.sim['fnames_oi']]
        # Suppress non-centered parameterization parameters
        self._suppress_vars.extend([n for n in names if '_offset' in n])

        # instantiate
        return MCMCResults(model=self.spec, data=data, names=names, dims=dims,
                           levels=levels, transformed_vars=self._suppress_vars)
