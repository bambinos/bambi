"""Classes to represent prior distributions and methods to set automatic priors"""
from .prior import Prior
from .scaler_mle import PriorScalerMLE
from .scaler_default import PriorScaler

__all__ = [
    "Prior",
    "PriorScaler",
    "PriorScalerMLE",
]
