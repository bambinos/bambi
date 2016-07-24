import numpy as np

default_priors = {
    'intercept': {
        'name': 'Uniform',
        'args': {
            'lower': -1000,
            'upper': 1000
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
                'lower': -1000,
                'upper': 1000
            }
        }
    }
}
