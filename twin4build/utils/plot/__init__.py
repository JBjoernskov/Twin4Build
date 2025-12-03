# Local folder imports
from .plot import Colors, Entry, plot, plot_component

__all__ = ["Entry", "plot", "plot_component", "Colors"]

# Make Entry, Option, and Colors available directly in the plot namespace
# This allows tb.plot.Entry, tb.plot.Option, and tb.plot.Colors usage
