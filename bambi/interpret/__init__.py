import logging

from bambi.interpret.effects import comparisons, predictions, slopes
from bambi.interpret.helpers import data_grid, select_draws
from bambi.interpret.plotting import plot_comparisons, plot_predictions, plot_slopes

__all__ = [
    "comparisons",
    "data_grid",
    "logger",
    "select_draws",
    "slopes",
    "predictions",
    "plot_comparisons",
    "plot_predictions",
    "plot_slopes",
]

logger = logging.getLogger("__bambi_interpret__")

if not logging.root.handlers:
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler())
