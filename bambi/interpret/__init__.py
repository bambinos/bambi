import logging

from bambi.interpret.effects import (
    comparisons,
    plot_comparisons,
    plot_predictions,
    predictions,
)

__all__ = [
    "comparisons",
    "predictions",
    "plot_comparisons",
    "plot_predictions",
]

logger = logging.getLogger("__bambi_interpret__")

if not logging.root.handlers:
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        logger.addHandler(logging.StreamHandler())
