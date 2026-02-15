import logging

from bambi.interpret.effects import (
    comparisons,
    plot_comparisons,
    plot_predictions,
    plot_slopes,
    predictions,
    slopes,
)

__all__ = [
    "comparisons",
    "predictions",
    "slopes",
    "plot_comparisons",
    "plot_predictions",
    "plot_slopes",
]

logger = logging.getLogger("__bambi_interpret__")

if not logging.root.handlers:
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler())
