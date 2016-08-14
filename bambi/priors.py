import numpy as np

default_priors = {
    'intercept': {
        'name': 'Cauchy',
        'args': {
            'alpha': 0.,
            'beta': 1.
        }
    },
    'fixed': {
        'name': 'Normal',
        'args': {
            'mu': 0.,
            'sd': 1.
        }
    },
    'random': {
        'name': 'Normal',
        'args': {
            'mu': 0.
        },
        'sigma': {
            'name': 'HalfCauchy',
            'args': {
                'beta': 1.
            }
        }
    },
    'sigma': {
        'name': 'HalfCauchy',
        'args': {
            'beta': 1.
        }
    }
}
