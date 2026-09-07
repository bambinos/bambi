import logging
from importlib.metadata import version

from pymc import math

from bambi import interpret
from bambi.config import config
from bambi.data import clear_data_home, load_data
from bambi.families import Family, Likelihood, Link
from bambi.formula import Formula
from bambi.models import Model
from bambi.priors import Prior

__version__ = version("bambi")

__all__ = [
    "Family",
    "Formula",
    "Likelihood",
    "Link",
    "Model",
    "Prior",
    "clear_data_home",
    "config",
    "interpret",
    "load_data",
    "math",
]

_log = logging.getLogger("bambi")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
