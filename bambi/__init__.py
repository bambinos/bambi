import logging

from .models import Model
from .priors import Prior, Family
from .backends import PyMC3BackEnd
from .version import __version__


__all__ = ["Model", "Prior", "Family", "PyMC3BackEnd"]

_log = logging.getLogger("bambi")

if not logging.root.handlers:
    _log.setLevel(logging.INFO)
    if len(_log.handlers) == 0:
        handler = logging.StreamHandler()
        _log.addHandler(handler)
