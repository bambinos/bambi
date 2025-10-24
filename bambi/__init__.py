import logging

from importlib.metadata import version

from pymc import math

from bambi.backend import PyMCModel
from bambi.config import config
from bambi.data import clear_data_home, load_data
from bambi.families import Family, Likelihood, Link
from bambi.formula import Formula
from bambi.models import Model
from bambi.priors import Prior
from bambi import interpret

__version__ = version("bambi")

__all__ = [
    "Model",
    "Prior",
    "Family",
    "Likelihood",
    "Link",
    "PyMCModel",
    "Formula",
    "clear_data_home",
    "config",
    "load_data",
    "math",
]

_log = logging.getLogger("bambi")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
