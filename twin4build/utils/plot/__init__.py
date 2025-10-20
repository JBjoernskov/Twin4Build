from .plot import Entry, plot, plot_component, Colors

__all__ = ['Entry', 'plot', 'plot_component', 'Colors']

# Make Entry, Option, and Colors available directly in the plot namespace
# This allows tb.plot.Entry, tb.plot.Option, and tb.plot.Colors usage