"""This module contains the default priors for the parameters of different covariance kernels
that can be used in HSGP terms."""

# fmt: off
HSGP_COV_PARAMS_DEFAULT_PRIORS = {
    "ExpQuad": {
        "sigma": "Exponential",
        "ell": "InverseGamma"
    },
    "Matern32": {
        "sigma": "Exponential",
        "ell": "InverseGamma"
    },
    "Matern52": {
        "sigma": "Exponential",
        "ell": "InverseGamma"
    },
}
# fmt: on
