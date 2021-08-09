import logging

from pymc3 import math

from .data import clear_data_home, load_data
from .models import Model
from .priors import Prior, Family, Likelihood, Link
from .backends import PyMC3BackEnd
from .version import __version__


__all__ = ["Model", "Prior", "Family", "Likelihood", "Link", "PyMC3BackEnd"]

_log = logging.getLogger("bambi")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
