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
from itertools import cycle 
from collections import namedtuple

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
import twin4build.core as core
import twin4build.utils.types as tps
from twin4build.utils.mkdir_in_root import mkdir_in_root
from twin4build.utils.plot.align_y_axes import alignYaxes


# Named tuple for plot component specifications

class Entry:
    """
    A simple class for specifying plot data with named parameters (matplotlib-like API).
    
    Args:
        data: Data array/tensor to plot (required)
        label: Display label for the plot line (required)
        color: Plot line color (None for automatic color selection)
        fmt: Plot format string combining linestyle and marker (None for automatic style selection)
        axis: Which y-axis to plot on (1, 2, or 3, defaults to 1)
        linewidth: Plot line width (None for automatic width selection)
        
        # Deprecated parameters (for backward compatibility)
        component: Deprecated, use direct data instead
        port: Deprecated, use direct data instead
        io_type: Deprecated, use direct data instead
        input_idx: Deprecated, use direct data instead
        attribute: Deprecated, use label instead
        linestyle: Deprecated, use fmt instead
    
    Examples:
        # Simple data plot on axis 1
        Entry(data=temperature_array, label="Room Temperature")
        
        # Data plot with custom styling on axis 2
        Entry(data=power_array, label="Power Consumption", 
              color="red", fmt="--", axis=2)
        
        # Data plot on axis 3 with markers
        Entry(data=flow_array, label="Flow Rate", 
              color="blue", fmt="o-", axis=3)
    """
    def __init__(self, data=None, fmt=None, axis=1,
                 # Deprecated parameters for backward compatibility
                 component=None, port=None, io_type=None, input_idx=None, attribute=None, linestyle=None, **kwargs):
        
        # Handle deprecated parameters
        if component is not None or port is not None or io_type is not None or input_idx is not None:
            import warnings
            warnings.warn("Component-based plotting is deprecated. Use direct data arrays instead.", 
                         DeprecationWarning, stacklevel=2)
        
        if attribute is not None and label is None:
            import warnings
            warnings.warn("The 'attribute' parameter is deprecated. Use 'label' instead.", 
                         DeprecationWarning, stacklevel=2)
            label = attribute
        
        if linestyle is not None and fmt is None:
            import warnings
            warnings.warn("The 'linestyle' parameter is deprecated. Use 'fmt' instead.", 
                         DeprecationWarning, stacklevel=2)
            fmt = linestyle
        
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

    @classmethod
    @property
    def save_folder(cls):
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
    Extract data, label, color, and format from an Entry or legacy tuple.
    
    Args:
        t: Entry object or legacy tuple
        
    Returns:
        tuple: (data, label, color, fmt, linewidth, alpha)
    """    
    # Handle Entry class (and Option for backward compatibility)
    if isinstance(t, Entry):
        data = t.data
        fmt = t.fmt
        axis = t.axis
        kwargs = t.kwargs
        return data, fmt, axis, kwargs
    
    # Handle legacy tuple formats for backward compatibility
    elif isinstance(t, tuple):
        import warnings
        warnings.warn("Using tuples for plot data is deprecated. Use tb.plot.Entry() instead.", 
                     DeprecationWarning, stacklevel=3)
        
        if len(t) == 2:
            # Direct data case
            data, label = t
            assert isinstance(
                data, (torch.Tensor, np.ndarray, pd.Series, list)
            ), f"First element must be data array, got {type(data)}"
            assert isinstance(
                label, str
            ), f"Second element must be a string label, got {type(label)}"
            return data, fmt, axis, kwargs
        else:
            # For backward compatibility, we should not raise an error here
            # Instead, return None to indicate this needs component-based processing
            return None, None, None, None
    else:
        raise ValueError(f"Wrong input type. Got {type(t)}, expected Entry object or tuple")
        
    return data, fmt, axis, kwargs


def get_data_legacy(simulator, t):
    """
    Legacy function to extract data from component-based tuples.
    This maintains full backward compatibility.
    
    Args:
        simulator: The simulator object containing the model and time steps
        t: Legacy tuple with component information
        
    Returns:
        tuple: (data, label, color, fmt, linewidth, alpha)
    """
    color = None
    fmt = None
    linewidth = None
    alpha = None
    
    # Handle legacy tuple formats for backward compatibility
    if isinstance(t, tuple):
        import warnings
        warnings.warn("Using component-based tuples is deprecated. Use direct data arrays instead.", 
                     DeprecationWarning, stacklevel=3)
        
        if len(t) >= 3:
            if len(t) == 4:
                component, attribute, io_type, input_idx = t
            elif len(t) == 3:
                component, attribute, io_type = t
                input_idx = None
        elif len(t) == 2:
            data, attribute = t
            assert isinstance(
                data, (torch.Tensor, np.ndarray, pd.Series, list)
            ), f"If 2-tuple, first element must be a torch.Tensor or np.ndarray or pd.Series, got {type(data)}"
            assert isinstance(
                attribute, str
            ), f"If 2-tuple, second element must be a string, got {type(attribute)}"
        else:
            m = f"Wrong input output type specification. Got {t}, expected (component, attribute) or (component, attribute, 'input' or 'output')"
            raise (Exception(m))
    else:
        raise ValueError(f"Expected tuple for legacy component processing, got {type(t)}")

    # Handle component processing
    if isinstance(component, core.System):
        component = component
    elif isinstance(component, str):
        component = simulator.model.components[component]
    else:
        m = f"Wrong component type. Got {type(component)}, expected {core.System} or str"
        raise (Exception(m))

    assert isinstance(
        attribute, str
    ), f"Attribute must be a string, got {type(attribute)}"

    # Default io_type to "output" if not specified
    if io_type is None:
        io_type = "output"

    # Validate vector/scalar types based on input_idx
    if input_idx is not None:
        if io_type == "input":
            assert isinstance(component.input[attribute], tps.Vector), "Attribute must be a vector when input_idx is provided"
        else:
            assert isinstance(component.output[attribute], tps.Vector), "Attribute must be a vector when input_idx is provided"
    else:
        if io_type == "input":
            assert isinstance(component.input[attribute], tps.Scalar), "Attribute must be a scalar when input_idx is not provided"
        else:
            assert isinstance(component.output[attribute], tps.Scalar), "Attribute must be a scalar when input_idx is not provided"

    # Extract data
    if io_type == "input":
        data = component.input[attribute].history.detach()
        if input_idx is not None:
            data = data[:,input_idx]
            display_label = f"{attribute}[{input_idx}]"
        else:
            display_label = attribute
    elif io_type == "output":
        data = component.output[attribute].history.detach()
        if input_idx is not None:
            data = data[:,input_idx]
            display_label = f"{attribute}[{input_idx}]"
        else:
            display_label = attribute
    else:
        m = f"Wrong input output type specification. Got {io_type}, expected 'input' or 'output'"
        raise (Exception(m))
    
    # Use attribute name as label for legacy tuples
    final_label = display_label
        
    return data, final_label, color, fmt, linewidth, alpha


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
        show (bool): Whether to display the plot.

    Returns:
        tuple: Figure and axes objects.
    """
    assert time is not None, "time parameter is required"
    assert entries is not None, "entries parameter is required"
    
    # Separate entries by axis
    components_1axis = [e for e in entries if e.axis == 1]
    components_2axis = [e for e in entries if e.axis == 2]
    components_3axis = [e for e in entries if e.axis == 3]
    
    # Ensure we have at least some entries on axis 1
    if not components_1axis:
        raise ValueError("At least one entry must be specified for axis=1")
    load_params()
    fig, ax1 = plt.subplots(figsize=(12, 10)) # 12, 6
    if title:
        fig.suptitle(title, fontsize=20)
    # ax1.ticklabel_format(useOffset=False, style='plain')

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
    colors = cycle(Colors.colors.copy())

    if len(components_1axis) > 1:
        assert (
            ylabel_1axis is not None
        ), "ylabel_1axis is required if multiple components are plotted on the first axis"
    else:
        if ylabel_1axis is None:
            ylabel_1axis = components_1axis[0][1]

    # Plot components on the first axis
    for t in components_1axis:
        data, fmt, axis, kwargs = get_data(t)
        if data is None:
            # This should not happen in the new API since we only accept Entry objects or 2-tuples
            raise ValueError("Component-based tuples are not supported in the new plot() function. Use plot_component() for backward compatibility or convert to direct data arrays.")
        fmt = fmt if fmt is not None else "-"
        color = kwargs.get("color", next(colors))
        kwargs["color"] = color
        (line,) = ax1.plot(time, data, fmt, **kwargs)

    ax1.set_xlabel("Time")
    if ylabel_1axis:
        ax1.set_ylabel(ylabel_1axis)

    if ylim_1axis:
        ax1.set_ylim(ylim_1axis)


    colors = cycle(Colors.colors.copy())
    # Plot components on the second axis if provided
    if components_2axis:
        if len(components_2axis) > 1:
            assert (
                ylabel_2axis is not None
            ), "ylabel_2axis is required if multiple components are plotted on the second axis"
        else:
            if ylabel_2axis is None:
                ylabel_2axis = components_2axis[0][1]

        ax2 = ax1.twinx()
        ax2.yaxis.set_major_formatter(y_formatter)
        axes.append(ax2)
        nticks_list.append(nticks_2axis)
        roundto_list.append(roundto_2axis)
        yoffset_list.append(yoffset_2axis)
        for t in components_2axis:
            data, fmt, axis, kwargs = get_data(t)
            if data is None:
                raise ValueError("Component-based tuples are not supported in the new plot() function. Use plot_component() for backward compatibility or convert to direct data arrays.")
            fmt = fmt if fmt is not None else "-"
            color = kwargs.get("color", next(colors))
            kwargs["color"] = color
            (line,) = ax2.plot(time, data, fmt, **kwargs)   

        if ylabel_2axis:
            ax2.set_ylabel(ylabel_2axis)
        if ylim_2axis:
            ax2.set_ylim(ylim_2axis)


    colors = cycle(Colors.colors.copy())
    # Plot components on the third axis if provided
    if components_3axis:
        if len(components_3axis) > 1:
            assert (
                ylabel_3axis is not None
            ), "ylabel_3axis is required if multiple components are plotted on the third axis"
        else:
            if ylabel_3axis is None:
                ylabel_3axis = components_3axis[0][1]

        ax3 = ax1.twinx()
        ax3.yaxis.set_major_formatter(y_formatter)
        ax3.spines["right"].set_position(("outward", PlotSettings.outward))
        axes.append(ax3)
        nticks_list.append(nticks_3axis)
        roundto_list.append(roundto_3axis)
        yoffset_list.append(yoffset_3axis)
        for t in components_3axis:
            data, fmt, axis, kwargs = get_data(t)
            if data is None:
                raise ValueError("Component-based tuples are not supported in the new plot() function. Use plot_component() for backward compatibility or convert to direct data arrays.")
            fmt = fmt if fmt is not None else "-"
            color = kwargs.get("color", next(colors))
            kwargs["color"] = color
            (line,) = ax3.plot(time, data, fmt, **kwargs)

        if ylabel_3axis:
            ax3.set_ylabel(ylabel_3axis)
        if ylim_3axis:
            ax3.set_ylim(ylim_3axis)

        ax3.spines["right"].set_position(("outward", PlotSettings.outward))
        ax3.spines["right"].set_visible(True)
        ax3.spines["right"].set_color("black")

    # Combine legends
    lines, labels = [], []
    for ax in axes:
        ax_lines, ax_labels = ax.get_legend_handles_labels()
        lines.extend(ax_lines)
        labels.extend(ax_labels)

    legend = fig.legend(lines, labels, ncol=3, bbox_to_anchor=(0.5,0.95), loc="upper center") # , 

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

    for ax in axes:
        mylocator = mdates.HourLocator(interval=6, tz=None)
        ax.xaxis.set_minor_locator(mylocator)
        myFmt = mdates.DateFormatter("%H")
        ax.xaxis.set_minor_formatter(myFmt)

        mylocator = mdates.WeekdayLocator(
            byweekday=[
                mdates.MO,
                mdates.TU,
                mdates.WE,
                mdates.TH,
                mdates.FR,
                mdates.SA,
                mdates.SU,
            ],
            interval=1,
            tz=None,
        )
        ax.xaxis.set_major_locator(mylocator)
        myFmt = mdates.DateFormatter("%a")
        ax.xaxis.set_major_formatter(myFmt)
        ax.tick_params(axis="x", which="major", pad=10)  # move the tick labels

    # Save and show plot
    # component_ids = [comp[0] for comp in components_1axis + (components_2axis or []) + (components_3axis or [])]
    # plot_filename = os.path.join(PlotSettings.save_folder, f"{get_file_name('_'.join(component_ids))}.png")
    # fig.savefig(plot_filename, dpi=300, bbox_inches='tight')

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


def plot_component(
    simulator,
    components_1axis,
    components_2axis=None,
    components_3axis=None,
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
    Deprecated: Use plot() with time parameter and Entry objects instead.
    
    This function is maintained for backward compatibility but will be removed in a future version.
    """
    import warnings
    warnings.warn("plot_component() is deprecated. Use plot(time, entries) instead.", 
                 DeprecationWarning, stacklevel=2)
    
    # Extract time from simulator
    time = simulator.date_time_steps
    
    # Convert old-style components to Entry objects
    entries = []
    
    # Add axis 1 components
    for comp in components_1axis:
        if isinstance(comp, Entry):
            entries.append(comp)
        else:
            # Handle legacy tuples with full backward compatibility
            if len(comp) == 2:
                # Direct data tuple
                entries.append(Entry(data=comp[0], label=comp[1], axis=1))
            else:
                # Component-based tuple - extract data using legacy function
                data, label, color, fmt = get_data_legacy(simulator, comp)
                entries.append(Entry(data=data, label=label, color=color, fmt=fmt, axis=1))
    
    # Add axis 2 components
    if components_2axis:
        for comp in components_2axis:
            if isinstance(comp, Entry):
                entries.append(comp)
            else:
                if len(comp) == 2:
                    # Direct data tuple
                    entries.append(Entry(data=comp[0], label=comp[1], axis=2))
                else:
                    # Component-based tuple - extract data using legacy function
                    data, label, color, fmt = get_data_legacy(simulator, comp)
                    entries.append(Entry(data=data, label=label, color=color, fmt=fmt, axis=2))
    
    # Add axis 3 components
    if components_3axis:
        for comp in components_3axis:
            if isinstance(comp, Entry):
                entries.append(comp)
            else:
                if len(comp) == 2:
                    # Direct data tuple
                    entries.append(Entry(data=comp[0], label=comp[1], axis=3))
                else:
                    # Component-based tuple - extract data using legacy function
                    data, label, color, fmt = get_data_legacy(simulator, comp)
                    entries.append(Entry(data=data, label=label, color=color, fmt=fmt, axis=3))
    
    # Call the new plot function
    return plot(
        time=time,
        entries=entries,
        ylabel_1axis=ylabel_1axis,
        ylabel_2axis=ylabel_2axis,
        ylabel_3axis=ylabel_3axis,
        ylim_1axis=ylim_1axis,
        ylim_2axis=ylim_2axis,
        ylim_3axis=ylim_3axis,
        title=title,
        nticks=nticks,
        roundto_1axis=roundto_1axis,
        roundto_2axis=roundto_2axis,
        roundto_3axis=roundto_3axis,
        yoffset_1axis=yoffset_1axis,
        yoffset_2axis=yoffset_2axis,
        yoffset_3axis=yoffset_3axis,
        align_zero=align_zero,
        show=show,
    )
