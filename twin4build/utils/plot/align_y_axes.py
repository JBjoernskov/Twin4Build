r"""
Functions for calculating and aligning y-axis ticks in plots.

Mathematical Formulation:

1. Tick Spacing Calculation:
   For a data range :math:`[y_{min}, y_{max}]` and desired number of ticks :math:`n`:

   .. math::

      \Delta y = \frac{y_{max} - y_{min}}{n-1}

   The tick spacing is rounded to a "nice" number:

   .. math::

      \Delta y_{nice} = \lceil \frac{\Delta y}{10^m} \rceil \cdot 10^m

   where:
   - :math:`m = \lfloor \log_{10}(\Delta y) \rfloor` is the order of magnitude

2. Zero-Aligned Ticks:
   When aligning with zero at index :math:`i`:

   .. math::

      \begin{align}
      n_{below} &= i \\
      n_{above} &= n - i - 1 \\
      y_{min} &= -n_{below} \cdot \Delta y_{nice} \\
      y_{max} &= n_{above} \cdot \Delta y_{nice}
      \end{align}

3. Y-Axis Alignment:
   For multiple axes with ranges :math:`[y_{i,min}, y_{i,max}]`:

   .. math::

      y_{i,offset} = \frac{(y_{i,max} - y_{i,min}) \cdot y_{ref,offset}}{y_{ref,max} - y_{ref,min} + 2y_{ref,offset}} \cdot \frac{1}{1 - \frac{2y_{ref,offset}}{y_{ref,max} - y_{ref,min} + 2y_{ref,offset}}}

   where:
   - :math:`y_{ref,offset}` is the reference axis offset
   - :math:`y_{ref,min}, y_{ref,max}` are the reference axis limits
"""

# Third party imports
import numpy as np


def calculate_ticks(ax, nticks, round_to=None, zero_tick_idx=None):
    if round_to is None:
        # Calculate data range
        data_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        # Find appropriate order of magnitude
        magnitude = np.floor(np.log10(data_range / (nticks - 1)))
        # Set round_to to be 1, 2, or 5 times this magnitude
        candidates = np.array([1, 2, 5]) * 10**magnitude
        # Choose the one that gives closest to desired number of ticks
        divisions = data_range / candidates
        best_idx = np.argmin(np.abs(divisions - (nticks - 1)))
        round_to = candidates[best_idx]

    ylim = ax.get_ylim()
    data_min, data_max = ylim[0], ylim[1]

    if zero_tick_idx is not None and data_min < 0 < data_max:
        # Calculate ticks with zero at the specified index
        n_below = zero_tick_idx  # number of ticks below zero
        n_above = nticks - zero_tick_idx - 1  # number of ticks above zero

        # Calculate tick spacing based on the larger range
        max_range = max(abs(data_min), abs(data_max))
        tick_spacing = max_range / max(n_below, n_above)
        # Round the spacing to a nice number
        magnitude = np.floor(np.log10(tick_spacing))
        tick_spacing = np.ceil(tick_spacing / (10**magnitude)) * (10**magnitude)

        # Calculate bounds
        lower = -n_below * tick_spacing
        upper = n_above * tick_spacing

        # Adjust bounds to ensure data fits
        if data_min < lower:
            lower = np.floor(data_min / tick_spacing) * tick_spacing
        if data_max > upper:
            upper = np.ceil(data_max / tick_spacing) * tick_spacing

        ticks = np.linspace(lower, upper, nticks)
    else:
        # For non-zero-crossing cases, center the data range
        tick_spacing = (data_max - data_min) / (nticks - 1)
        magnitude = np.floor(np.log10(tick_spacing))
        tick_spacing = np.ceil(tick_spacing / (10**magnitude)) * (10**magnitude)

        lower = np.floor(data_min / tick_spacing) * tick_spacing
        upper = np.ceil(data_max / tick_spacing) * tick_spacing

        # Center the range if possible
        total_ticks = (upper - lower) / tick_spacing
        if total_ticks > nticks - 1:
            rem = np.remainder(total_ticks, nticks - 1)
            lower = lower + rem * tick_spacing / 2
            upper = upper - rem * tick_spacing / 2

        ticks = np.linspace(lower, upper, nticks)

    return ticks


def alignYaxes(axes_list, nticks_list, round_to_list, yoffset_list, align_zero=True):
    if align_zero:
        # Find axes that contain zero
        zero_axes = []
        for ax in axes_list:
            ylim = ax.get_ylim()
            if ylim[0] < 0 < ylim[1]:
                zero_axes.append(ax)

        if zero_axes:
            # Default to first tick being zero unless data suggests otherwise
            zero_tick_idx = 0
            for ax in zero_axes:
                ylim = ax.get_ylim()
                if abs(ylim[0]) > ylim[1]:
                    # More data below zero, put zero near the top
                    zero_tick_idx = min(nticks_list) - 2
                    break
        else:
            zero_tick_idx = None
    else:
        zero_tick_idx = None

    # Calculate ticks for each axis
    for ax, nticks, round_to in zip(axes_list, nticks_list, round_to_list):
        ticks = calculate_ticks(
            ax, nticks=nticks, round_to=round_to, zero_tick_idx=zero_tick_idx
        )
        ax.set_yticks(ticks)
        ax.set_ylim([ticks[0], ticks[-1]])

    # Continue with existing offset adjustment
    ybound_list = [ax.get_ylim() for ax in axes_list]
    yoffset_ybound_master_list = [
        [yoffset, ybound]
        for yoffset, ybound in zip(yoffset_list, ybound_list)
        if yoffset is not None
    ][0]
    yoffset_master = yoffset_ybound_master_list[0]
    ydiff_master = yoffset_ybound_master_list[1][1] - yoffset_ybound_master_list[1][0]
    ydiff_list = [ax.get_ylim()[1] - ax.get_ylim()[0] for ax in axes_list]
    yoffset_new_list = [
        ((ydiff * yoffset_master) / (ydiff_master + 2 * yoffset_master))
        / (1 - (2 * yoffset_master) / (ydiff_master + 2 * yoffset_master))
        for ydiff in ydiff_list
    ]

    for ax, yoffset_new in zip(axes_list, yoffset_new_list):
        ax.set_ylim([ax.get_ylim()[0] - yoffset_new, ax.get_ylim()[1] + yoffset_new])
