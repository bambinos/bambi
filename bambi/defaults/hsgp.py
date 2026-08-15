"""Default priors for parameters of covariance kernels in HSGP terms."""

HSGP_COV_PARAMS_DEFAULT_PRIORS = {
    "ExpQuad": {
        "sigma": {"name": "Exponential", "lam": 1},
        "ell": {"name": "InverseGamma", "alpha": 3, "beta": 2},
    },
    "Matern32": {
        "sigma": {"name": "Exponential", "lam": 1},
        "ell": {"name": "InverseGamma", "alpha": 3, "beta": 2},
    },
    "Matern52": {
        "sigma": {"name": "Exponential", "lam": 1},
        "ell": {"name": "InverseGamma", "alpha": 3, "beta": 2},
    },
}
