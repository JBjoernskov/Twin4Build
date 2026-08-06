r"""
Plotting functions and formatting utilities for data visualization.

Mathematical Formulation:

1. Time Series Plotting:
   For a time series :math:`y(t)`:

   .. math::

      y_{formatted}(t) = \begin{cases}
      y(t) & \text{if } y_{min} \leq y(t) \leq y_{max} \\
      y_{min} & \text{if } y(t) < y_{min} \\
      y_{max} & \text{if } y(t) > y_{max}
      \end{cases}

   where:
   - :math:`y_{min}, y_{max}` are the y-axis limits
   - :math:`t` is the time index

2. Multi-Axis Alignment:
   For multiple y-axes with values :math:`y_1, y_2, ..., y_n`:

   .. math::

      y_i' = \frac{y_i - y_{i,min}}{y_{i,max} - y_{i,min}} \cdot (y_{ref,max} - y_{ref,min}) + y_{ref,min}

   where:
   - :math:`y_i` is the original value on axis i
   - :math:`y_{i,min}, y_{i,max}` are the min/max values on axis i
   - :math:`y_{ref,min}, y_{ref,max}` are the reference axis limits

3. Time Label Formatting:
   For time :math:`t` with evaluation metric :math:`m`:

   .. math::

      label(t) = \begin{cases}
      h(t) & \text{if } m = \text{"H"} \\
      d(t) & \text{if } m = \text{"D"} \\
      w(t) & \text{if } m = \text{"W"} \\
      M(t) & \text{if } m = \text{"M"} \\
      Y(t) & \text{if } m = \text{"A"}
      \end{cases}

   where:
   - :math:`h(t)` is the hour format
   - :math:`d(t)` is the day format
   - :math:`w(t)` is the week format
   - :math:`M(t)` is the month format
   - :math:`Y(t)` is the year format
"""

# Standard library imports
import itertools
import math
import shutil
from collections import namedtuple
from itertools import cycle

# Third party imports
import matplotlib.dates as mdates
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import cm
from matplotlib import colors as mplcolor

# import corner
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

# Local application imports
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.plot.align_y_axes import alignYaxes


class Entry:
    """
    A simple class for specifying plot data with named parameters (matplotlib-like API).

    Args:
        data: Data array/tensor to plot (required).
        fmt: Plot format string combining linestyle and marker (None for automatic style selection).
        axis: Which y-axis to plot on (1, 2, or 3, defaults to 1).
        **kwargs: Additional matplotlib line properties, passed through to the
            plot call and set as attributes on the entry. ``label`` (display
            label for the plot line) is required; common optional keys include
            ``color`` (None for automatic color selection) and ``linewidth``
            (None for automatic width selection).

    Examples:
        Basic usage::

            # Simple data plot on axis 1
            Entry(data=temperature_array, label="Room Temperature")

            # Data plot with custom styling on axis 2
            Entry(data=power_array, label="Power Consumption",
                  color="red", fmt="--", axis=2)

            # Data plot on axis 3 with markers
            Entry(data=flow_array, label="Flow Rate",
                  color="blue", fmt="o-", axis=3)
    """

    def __init__(
        self,
        data=None,
        fmt=None,
        axis=1,
        **kwargs,
    ):
        # Check for common mistake: passing a method instead of calling it
        if callable(data):
            raise TypeError(
                "The 'data' parameter appears to be a callable (method/function). "
                "Did you forget to call .history()? "
                "Use .history() instead of .history to get the data array."
            )

        # Convert data to numpy array if necessary
        if isinstance(data, (list, pd.Series)):
            data = np.array(data)
        elif isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Set attributes
        self.data = data
        self.fmt = fmt
        self.axis = axis
        self.kwargs = kwargs

        for k, v in kwargs.items():
            setattr(self, k, v)

        # Validation
        if data is None:
            raise ValueError("'data' is required")
        assert "label" in kwargs, "'label' is required"


class Colors:
    colors = sns.color_palette("deep")
    blue = colors[0]
    orange = colors[1]
    green = colors[2]
    red = colors[3]
    purple = colors[4]
    brown = colors[5]
    pink = colors[6]
    grey = colors[7]
    beis = colors[8]
    sky_blue = colors[9]
    black = "black"


class PlotSettings:
    legend_loc = (0.5, 0.93)
    x = (0.45, 0.05)
    left_y = (0.025, 0.50)
    right_y_first = (0.86, 0.50)
    right_y_second = (0.975, 0.50)
    outward = 68

    @staticmethod
    def save_folder():
        save_folder, isfile = mkdir_in_root(["generated_files", "plots"])
        return save_folder


def on_pick(event, fig, graphs):
    legend = event.artist
    # isVisible = legend.get_visible()

    # Get the corresponding plot line from the graphs dictionary
    line = graphs[legend]
    isVisible = line.get_visible()

    isVisible = not isVisible
    line.set_visible(isVisible)

    # Toggle visibility and transparency of the legend line
    # legend.set_visible(not isVisible)
    if isVisible:
        legend.set_alpha(1)  # Make legend more transparent when line is hidden
    else:
        legend.set_alpha(0.2)  # Make legend more transparent when line is hidden

    # Redraw the figure
    fig.canvas.draw_idle()


def load_params():
    # usetex = True if sys.platform == "darwin" else False
    usetex = True if shutil.which("latex") else False
    params = {
        # 'figure.figsize': (fig_size_x, fig_size_y),
        #  'figure.dpi': 300,
        "axes.labelsize": 17,
        "axes.titlesize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "xtick.major.size": 10,
        "xtick.major.width": 1,
        "ytick.major.size": 10,
        "ytick.major.width": 1,
        "lines.linewidth": 2,  # 4,
        "figure.titlesize": 20,
        "mathtext.fontset": "cm",
        "legend.fontsize": 14,
        "axes.grid": True,
        "grid.color": "black",
        "grid.alpha": 0.2,
        "axes.unicode_minus": False,
        "legend.loc": "upper right",
        "legend.fancybox": False,
        "legend.facecolor": "white",
        "legend.framealpha": 1,
        "legend.edgecolor": "black",
        "font.family": "serif",
        "font.serif": "cmr10",  # Computer Modern
        "axes.formatter.use_mathtext": True,
        "text.usetex": usetex,
        # "text.latex.preamble": r"\usepackage{amsmath}",
        # "pgf.preamble": "\n".join([ # plots will use this preamble
        "text.latex.preamble": "\n".join(
            [  # plots will use this preamble
                r"\usepackage{amsmath}",
                r"\usepackage{bm}",
                r"\newcommand{\matrva}[1]{\bm{#1}}",
            ]
        ),
    }

    plt.style.use("ggplot")
    pylab.rcParams.update(params)
    # plt.rc('font', family='serif')


def get_file_name(name):
    name = name.replace(" ", "_").lower()
    return f"plot_{name}"


def bar_plot_line_format(label, evaluation_metric):
    """
    Convert time label to the format of pandas line plot
    """
    if evaluation_metric == "H":
        hour = "{:02d}".format(label.hour)
        if hour == "00":
            hour += f"\n{label.day_name()[:3]}"
        label = hour

    elif evaluation_metric == "D":
        day = label.day_name()[:3]
        if label.dayofweek == 0:
            day += f"\nweek {label.isocalendar()[1]}"
        label = day

    elif evaluation_metric == "W":
        week = "{:02d}".format(label.isocalendar()[1])
        if label.day <= 7:
            week += f"\n{label.month_name()[:3]}"

        label = week

    elif evaluation_metric == "M":
        month = label.month_name()[:3]
        if month == "Jan":
            month += f"\n{label.year}"
        label = month

    elif evaluation_metric == "A":
        year = label.month_name()[:3]
        label = year
    return label


def get_data(t):
    """
    Extract data, label, color, and format from an Entry.

    Args:
        t: Entry object

    Returns:
        tuple: (data, fmt, axis, kwargs)
    """
    if not isinstance(t, Entry):
        raise ValueError(f"Wrong input type. Got {type(t)}, expected Entry object")

    data = t.data
    fmt = t.fmt
    axis = t.axis
    kwargs = t.kwargs

    if isinstance(data, (list, pd.Series)):
        data = np.array(data)
    elif isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Handle 3D data - history() returns (n_t, n_s, n_c), need (n_s, n_t) for plotting
    if isinstance(data, np.ndarray) and data.ndim == 3:
        # History format: (n_t, n_s, n_c) - select first component and transpose to (n_s, n_t)
        data = data[:, :, 0].T  # Select n_c=0, then transpose (n_t, n_s) -> (n_s, n_t)

    if isinstance(data, np.ndarray) and data.ndim == 1:
        data = data.reshape(1, -1)

    return data, fmt, axis, kwargs


def filter_nans(time, data):
    valid_mask = ~pd.isna(time)
    time = time[valid_mask]
    data = data[valid_mask]
    return time, data


def plot(
    time,
    entries,
    ylabel_1axis=None,
    ylabel_2axis=None,
    ylabel_3axis=None,
    ylim_1axis=None,
    ylim_2axis=None,
    ylim_3axis=None,
    title=None,
    nticks=11,
    roundto_1axis=None,
    roundto_2axis=None,
    roundto_3axis=None,
    yoffset_1axis=None,
    yoffset_2axis=None,
    yoffset_3axis=None,
    align_zero=True,
    show=False,
):
    """
    General plot function with matplotlib-like API.

    Args:
        time: Time array/index for x-axis (required)
        entries (list): List of Entry objects specifying what to plot on which axis.
        ylabel_1axis (str, optional): Label for the first y-axis.
        ylabel_2axis (str, optional): Label for the second y-axis.
        ylabel_3axis (str, optional): Label for the third y-axis.
        ylim_1axis (tuple, optional): Y-axis limits for the first axis.
        ylim_2axis (tuple, optional): Y-axis limits for the second axis.
        ylim_3axis (tuple, optional): Y-axis limits for the third axis.
        title (str, optional): Plot title.
        nticks (int): Number of y-axis ticks on each axis (default 11).
        roundto_1axis (float, optional): Round the first axis' tick values to this resolution.
        roundto_2axis (float, optional): Round the second axis' tick values to this resolution.
        roundto_3axis (float, optional): Round the third axis' tick values to this resolution.
        yoffset_1axis (float, optional): Extra headroom offset applied to the first axis' limits.
        yoffset_2axis (float, optional): Extra headroom offset applied to the second axis' limits.
        yoffset_3axis (float, optional): Extra headroom offset applied to the third axis' limits.
        align_zero (bool): Align the zero (or common reference) level across the y-axes (default True).
        show (bool): Whether to display the plot.

    Returns:
        tuple: Figure and axes objects.
    """
    assert time is not None, "time parameter is required"
    assert entries is not None, "entries parameter is required"

    # Normalize time to list of arrays (batch mode) if it's a single time series
    if isinstance(time, (pd.Index, pd.Series)):
        time = [time]
    elif isinstance(time, np.ndarray):
        if time.ndim == 1 and (
            np.issubdtype(time.dtype, np.number)
            or np.issubdtype(time.dtype, np.datetime64)
        ):
            time = [time]
    elif isinstance(time, list):
        if len(time) > 0 and not isinstance(
            time[0], (list, np.ndarray, pd.Series, pd.Index)
        ):
            time = [time]

    if isinstance(entries, Entry):
        entries = [entries]

    # Separate entries by axis
    components_1axis = [e for e in entries if e.axis == 1 or e.axis is None]
    components_2axis = [e for e in entries if e.axis == 2]
    components_3axis = [e for e in entries if e.axis == 3]

    # Ensure we have at least some entries on axis 1
    if not components_1axis:
        raise ValueError("At least one entry must be specified for axis=1")
    load_params()
    fig, ax1 = plt.subplots(figsize=(12, 6))  # 12, 6
    if title:
        fig.suptitle(title, fontsize=20)

    y_formatter = ScalarFormatter(useOffset=False)
    ax1.yaxis.set_major_formatter(y_formatter)

    nticks_1axis = nticks
    nticks_2axis = nticks
    nticks_3axis = nticks

    axes = [ax1]
    nticks_list = [nticks_1axis]
    roundto_list = [roundto_1axis]
    yoffset_list = [yoffset_1axis]
    graphs = {}  # Will store mapping from legend entries to plot lines
    colors = [cycle(Colors.colors.copy()) for _ in range(len(time))]

    # Create axes upfront so entries can be plotted in original order
    ax2 = None
    ax3 = None
    if components_2axis:
        ax2 = ax1.twinx()
        ax2.yaxis.set_major_formatter(y_formatter)
        axes.append(ax2)
        nticks_list.append(nticks_2axis)
        roundto_list.append(roundto_2axis)
        yoffset_list.append(yoffset_2axis)
    if components_3axis:
        ax3 = ax1.twinx()
        ax3.yaxis.set_major_formatter(y_formatter)
        ax3.spines["right"].set_position(("outward", PlotSettings.outward))
        axes.append(ax3)
        nticks_list.append(nticks_3axis)
        roundto_list.append(roundto_3axis)
        yoffset_list.append(yoffset_3axis)

    # Axis label inference
    if len(components_1axis) > 1:
        assert (
            ylabel_1axis is not None
        ), "ylabel_1axis is required if multiple components are plotted on the first axis"
    elif ylabel_1axis is None:
        ylabel_1axis = components_1axis[0].label

    if components_2axis:
        if len(components_2axis) > 1:
            assert (
                ylabel_2axis is not None
            ), "ylabel_2axis is required if multiple components are plotted on the second axis"
        elif ylabel_2axis is None:
            ylabel_2axis = components_2axis[0].label

    if components_3axis:
        if len(components_3axis) > 1:
            assert (
                ylabel_3axis is not None
            ), "ylabel_3axis is required if multiple components are plotted on the third axis"
        elif ylabel_3axis is None:
            ylabel_3axis = components_3axis[0].label

    # Map axis number to axis object
    axis_map = {1: ax1, None: ax1}
    if ax2 is not None:
        axis_map[2] = ax2
    if ax3 is not None:
        axis_map[3] = ax3

    # Plot entries in the original order so the legend matches
    legend_lines = []
    legend_labels = []
    for entry in entries:
        data, fmt, axis, kwargs = get_data(entry)
        fmt = fmt if fmt is not None else "-"
        target_ax = axis_map[axis]
        assert data.shape[0] == len(
            time
        ), "data and time must have the same number of rows (batch size)"
        n_s = data.shape[0]
        for i in range(n_s):
            kwargs_copy = kwargs.copy()
            color = kwargs.get("color", next(colors[i]))
            kwargs_copy["color"] = color
            kwargs_copy["label"] = (
                kwargs["label"] if n_s == 1 else kwargs["label"] + f" [{i}]"
            )
            t_i, d = filter_nans(time[i], data[i, :])
            (line,) = target_ax.plot(t_i, d, fmt, **kwargs_copy)
            legend_lines.append(line)
            legend_labels.append(kwargs_copy["label"])

    # Set axis labels and limits
    ax1.set_xlabel("Time")
    if ylabel_1axis:
        ax1.set_ylabel(ylabel_1axis)
    if ylim_1axis:
        ax1.set_ylim(ylim_1axis)

    if ax2 is not None:
        if ylabel_2axis:
            ax2.set_ylabel(ylabel_2axis)
        if ylim_2axis:
            ax2.set_ylim(ylim_2axis)

    if ax3 is not None:
        if ylabel_3axis:
            ax3.set_ylabel(ylabel_3axis)
        if ylim_3axis:
            ax3.set_ylim(ylim_3axis)
        ax3.spines["right"].set_position(("outward", PlotSettings.outward))
        ax3.spines["right"].set_visible(True)
        ax3.spines["right"].set_color("black")

    lines = legend_lines
    labels = legend_labels

    legend = fig.legend(
        lines, labels, ncol=3, bbox_to_anchor=(0.5, 0.95), loc="upper center"
    )  # ,

    # Set up pick event and create mapping between legend entries and plot lines
    for legend_line, plot_line in zip(legend.get_lines(), lines):
        legend_line.set_picker(True)
        legend_line.set_pickradius(5)
        graphs[legend_line] = plot_line  # Map legend entry to corresponding plot line

    fig.canvas.mpl_connect("pick_event", lambda event: on_pick(event, fig, graphs))

    # Format x-axis
    for label in ax1.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(0)

    # Align y-axes
    ylim = axes[0].get_ylim()
    if all([yoffset is None for yoffset in yoffset_list]):
        yoffset_list[0] = (ylim[1] - ylim[0]) * 0.05

    alignYaxes(axes, nticks_list, roundto_list, yoffset_list, align_zero=align_zero)

    first_time = time[0][0]
    tz = first_time.tzinfo if hasattr(first_time, "tzinfo") else None
    for ax in axes:
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(
            mdates.AutoDateFormatter(locator, tz=tz)
        )

    # Save and show plot
    # component_ids = [comp[0] for comp in components_1axis + (components_2axis or []) + (components_3axis or [])]
    # plot_filename = os.path.join(PlotSettings.save_folder, f"{get_file_name('_'.join(component_ids))}.png")
    # fig.savefig(plot_filename, dpi=300, bbox_inches='tight')

    fig.tight_layout(rect=(0, 0, 1, 0.9))

    if show:
        plt.show()

    return fig, axes


def get_fig_axes(
    title_name,
    n_plots=1,
    cols=1,
    K=0.38,
    size_inches=(8, 4.3),
    offset=(0.12, 0.18),
    ax_dim=(0.65, 0.6),
    y_offset_add_default=0.04,
):
    """
    Create a figure with a grid of manually positioned axes.

    Args:
        title_name: Figure suptitle.
        n_plots: Number of axes to create.
        cols: Number of columns in the grid.
        K: Vertical scaling factor used for row offsets.
        size_inches: Figure size ``(width, height)`` in inches.
        offset: ``(x, y)`` offset of the first axes in figure fraction.
        ax_dim: ``(width, height)`` of each axes in figure fraction.
        y_offset_add_default: Extra vertical spacing between rows.

    Returns:
        tuple: Figure and list of axes (ordered top-left to bottom-right).
    """
    fig = plt.figure()
    fig.set_size_inches(size_inches)
    rows = math.ceil(n_plots / cols)
    x_offset = offset[0]
    y_offset = offset[1]  # /K
    ax_width = ax_dim[0]
    ax_height = ax_dim[1]  # /K
    axes = []
    for i in range(rows):
        frac_i = i / rows
        for j in range(cols):
            if i != 0:
                y_offset_add = -y_offset_add_default / K
            else:
                y_offset_add = 0
            frac_j = j / (cols + 1)
            if int(i * cols + j) < n_plots:
                rect = [
                    frac_j + x_offset,
                    frac_i + y_offset + i * y_offset_add,
                    ax_width,
                    ax_height,
                ]
                axes.append(fig.add_axes(rect))

    axes.reverse()
    fig.suptitle(title_name, fontsize=20)
    return fig, axes

