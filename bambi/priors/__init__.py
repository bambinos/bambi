"""Classes to represent prior distributions and methods to set automatic priors"""

from bambi.priors.prior import Prior
from bambi.priors.scaler import scale_priors

__all__ = [
    "Prior",
    "scale_priors",
]
