import numpy as np

default_priors = {
    'intercept': {
        'name': 'Cauchy',
        'args': {
            'alpha': 0.,
            'beta': 10.
        }
    },
    'fixed': {
        'name': 'Normal',
        'args': {
            'mu': 0.,
            'sd': 10
        }
    },
    'random': {
        'name': 'Normal',
        'args': {
            'mu': 0.,
            'sd': 10
        },
        'sigma': {
            'name': 'Uniform',
            'args': {
                'lower': 0,
                'upper': 1000
            }
        }
    },
    'sigma': {
        'name': 'HalfCauchy',
        'args': {
            'beta': 10
        }
    }
}
