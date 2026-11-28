"""Posterior g-computation with explicit causal inputs and reference populations."""

from bambi.causal.effects import g_computation
from bambi.causal.plots import plot_effects
from bambi.causal.types import CausalResult

__all__ = ["CausalResult", "g_computation", "plot_effects"]
