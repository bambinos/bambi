import logging

from pymc import math

from .data import clear_data_home, load_data
from .families import Family, Likelihood, Link
from .models import Model
from .priors import Prior
from .backend import PyMCModel
from .version import __version__


__all__ = ["Model", "Prior", "Family", "Likelihood", "Link", "PyMCModel"]

_log = logging.getLogger("bambi")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
