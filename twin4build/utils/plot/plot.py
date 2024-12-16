import matplotlib.dates as mdates
import matplotlib.pylab as pylab
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
from twin4build.utils.plot.align_y_axes import alignYaxes
from twin4build.utils.uppath import uppath
import sys
from matplotlib import cm
from matplotlib import colors as mplcolor
import matplotlib
import os
import itertools
import shutil
import twin4build.model.model as model
import corner
from matplotlib.colors import LinearSegmentedColormap
from twin4build.utils.bayesian_inference import generate_quantiles
from twin4build.utils.mkdir_in_root import mkdir_in_root

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

class PlotSettings:
    legend_loc = (0.5,0.93)
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
    isVisible = legend.get_visible()
    # graphs[legend].set_visible(not isVisible)
    for line in graphs[legend]:
        isVisible = line.get_visible()
        line.set_visible(not isVisible)
        
    legend.set_visible(not isVisible)
    # legend.set_alpha(1.0 if not isVisible else 0.2)
    fig.canvas.draw()

def load_params():
    # usetex = True if sys.platform == "darwin" else False
    usetex = True if shutil.which("latex") else False
    params = {
            # 'figure.figsize': (fig_size_x, fig_size_y),
            #  'figure.dpi': 300,
            'axes.labelsize': 17,
            'axes.titlesize': 15,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13,
            "xtick.major.size": 10,
            "xtick.major.width": 1,
            "ytick.major.size": 10,
            "ytick.major.width": 1,
            "lines.linewidth": 2, #4,
            "figure.titlesize": 20,
            "mathtext.fontset": "cm",
            "legend.fontsize": 14,
            "axes.grid": True,
            "grid.color": "black",
            "grid.alpha": 0.2,
            'axes.unicode_minus': False,
            "legend.loc": "upper right",
            "legend.fancybox": False,
            "legend.facecolor": "white",
            "legend.framealpha": 1,
            "legend.edgecolor": "black",
            "font.family": "serif",
            "font.serif": "cmr10", #Computer Modern
            "axes.formatter.use_mathtext": True,
            "text.usetex": usetex,
            # "text.latex.preamble": r"\usepackage{amsmath}",
            # "pgf.preamble": "\n".join([ # plots will use this preamble
            "text.latex.preamble": "\n".join([ # plots will use this preamble
                r"\usepackage{amsmath}",
                r"\usepackage{bm}",
                r"\newcommand{\matrva}[1]{\bm{#1}}"
                ])
            }
    
    plt.style.use("ggplot")
    pylab.rcParams.update(params)
    # plt.rc('font', family='serif')

def get_file_name(name):
    name = name.replace(" ","_").lower()
    return f"plot_{name}"

def bar_plot_line_format(label, evaluation_metric):
    """
    Convert time label to the format of pandas line plot
    """
    if evaluation_metric=="H":
        hour = "{:02d}".format(label.hour)
        if hour == '00':
            hour += f'\n{label.day_name()[:3]}'
        label = hour

    elif evaluation_metric=="D":
        day = label.day_name()[:3]
        if label.dayofweek == 0:
            day += f'\nweek {label.isocalendar()[1]}'
        label = day

    elif evaluation_metric=="W":
        week =  "{:02d}".format(label.isocalendar()[1])
        if label.day<=7:
            week += f'\n{label.month_name()[:3]}'
            
        label = week

    elif evaluation_metric=="M":
        month = label.month_name()[:3]
        if month == 'Jan':
            month += f'\n{label.year}'
        label = month

    elif evaluation_metric=="A":
        year = label.month_name()[:3]
        label = year
    return label


def plot_component(simulator,
                   components_1axis, 
                   components_2axis=None, 
                   components_3axis=None, 
                   ylabel_1axis=None,
                   ylabel_2axis=None,
                   ylabel_3axis=None,
                   ylim_1axis=None, 
                   ylim_2axis=None, 
                   ylim_3axis=None,
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
    General plot function for components.

    Args:
        simulator: The simulator object containing the model and time steps.
        components_1axis (list): List of tuples (component_id, attribute) for the first y-axis.
        components_2axis (list, optional): List of tuples for the second y-axis.
        components_3axis (list, optional): List of tuples for the third y-axis.
        show (bool): Whether to display the plot.
        firstAxisylim (tuple, optional): Y-axis limits for the first axis.
        secondAxisylim (tuple, optional): Y-axis limits for the second axis.
        thirdAxisylim (tuple, optional): Y-axis limits for the third axis.

    Returns:
        tuple: Figure and axes objects.
    """
    assert components_1axis is not None, "components_1axis is required"
    load_params()
    fig, ax1 = plt.subplots(figsize=(12, 6))

    model = simulator.model
    time = simulator.dateTimeSteps

    nticks_1axis = nticks
    nticks_2axis = nticks
    nticks_3axis = nticks

    axes = [ax1]
    nticks_list = [nticks_1axis]
    roundto_list = [roundto_1axis]
    yoffset_list = [yoffset_1axis]
    graphs = {}
    colors = Colors.colors.copy()

    if len(components_1axis)>1:
        assert ylabel_1axis is not None, "ylabel_1axis is required if multiple components are plotted on the first axis"
    else:
        if ylabel_1axis is None:
            ylabel_1axis = components_1axis[0][1]
        
    # Plot components on the first axis
    for component_id, attribute in components_1axis:
        color = colors[0]
        colors.remove(color)
        data = np.array(model.components[component_id].savedOutput[attribute])
        line, = ax1.plot(time, data, label=attribute, color=color)
        graphs[line] = [line]

    ax1.set_xlabel("Time")
    if ylabel_1axis:
        ax1.set_ylabel(ylabel_1axis)

    if ylim_1axis:
        ax1.set_ylim(ylim_1axis)

    # Plot components on the second axis if provided
    if components_2axis:
        if len(components_2axis)>1:
            assert ylabel_2axis is not None, "ylabel_2axis is required if multiple components are plotted on the second axis"
        else:
            if ylabel_2axis is None:
                ylabel_2axis = components_2axis[0][1]

        ax2 = ax1.twinx()
        axes.append(ax2)
        nticks_list.append(nticks_2axis)
        roundto_list.append(roundto_2axis)
        yoffset_list.append(yoffset_2axis)
        for component_id, attribute in components_2axis:
            color = colors[0]
            colors.remove(color)
            data = np.array(model.components[component_id].savedInput[attribute])
            line, = ax2.plot(time, data, label=attribute, color=color, linestyle='--')
            graphs[line] = [line]
        
        if ylabel_2axis:
            ax2.set_ylabel(ylabel_2axis)
        if ylim_2axis:
            ax2.set_ylim(ylim_2axis)

    # Plot components on the third axis if provided
    if components_3axis:
        if len(components_3axis)>1:
            assert ylabel_3axis is not None, "ylabel_3axis is required if multiple components are plotted on the third axis"
        else:
            if ylabel_3axis is None:
                ylabel_3axis = components_3axis[0][1]

        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', PlotSettings.outward))
        axes.append(ax3)
        nticks_list.append(nticks_3axis)
        roundto_list.append(roundto_3axis)
        yoffset_list.append(yoffset_3axis)
        for component_id, attribute in components_3axis:
            color = colors[0]
            colors.remove(color)
            data = np.array(model.components[component_id].savedInput[attribute])
            line, = ax3.plot(time, data, label=attribute, color=color, linestyle=':')
            graphs[line] = [line]
        
        if ylabel_3axis:
            ax3.set_ylabel(ylabel_3axis)
        if ylim_3axis:
            ax3.set_ylim(ylim_3axis)

        ax3.spines['right'].set_position(('outward', PlotSettings.outward))
        ax3.spines["right"].set_visible(True)
        ax3.spines["right"].set_color("black")

    # Combine legends
    lines, labels = [], []
    for ax in axes:
        ax_lines, ax_labels = ax.get_legend_handles_labels()
        lines.extend(ax_lines)
        labels.extend(ax_labels)
    
    legend = fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)

    # Set up pick event
    for line in lines:
        line.set_picker(True)
        line.set_pickradius(5)

    fig.canvas.mpl_connect('pick_event', lambda event: on_pick(event, fig, graphs))

    # Format x-axis
    for label in ax1.get_xticklabels():
        label.set_ha("center")
        label.set_rotation(0)

    # Align y-axes
    ylim = axes[0].get_ylim()
    if all([yoffset is None for yoffset in yoffset_list]):
        yoffset_list[0] = (ylim[1]-ylim[0])*0.05

    alignYaxes(axes, nticks_list, roundto_list, yoffset_list, align_zero=align_zero)

    for ax in axes:
        mylocator = mdates.HourLocator(interval=6, tz=None)
        ax.xaxis.set_minor_locator(mylocator)
        myFmt = mdates.DateFormatter('%H')
        ax.xaxis.set_minor_formatter(myFmt)

        mylocator = mdates.WeekdayLocator(
            byweekday=[mdates.MO, mdates.TU, mdates.WE, mdates.TH, mdates.FR, mdates.SA, mdates.SU], interval=1,
            tz=None)
        ax.xaxis.set_major_locator(mylocator)
        myFmt = mdates.DateFormatter('%a')
        ax.xaxis.set_major_formatter(myFmt)
        ax.tick_params(axis='x', which='major', pad=10)  # move the tick labels

    # Save and show plot
    # component_ids = [comp[0] for comp in components_1axis + (components_2axis or []) + (components_3axis or [])]
    # plot_filename = os.path.join(PlotSettings.save_folder, f"{get_file_name('_'.join(component_ids))}.png")
    # fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig, axes

def get_fig_axes(title_name, n_plots=1, cols=1, K=0.38, size_inches=(8,4.3), offset=(0.12,0.18), ax_dim=(0.65,0.6), y_offset_add_default=0.04):
    fig = plt.figure()
    fig.set_size_inches(size_inches)
    rows = math.ceil(n_plots/cols)
    x_offset = offset[0]
    y_offset = offset[1] #/K
    ax_width = ax_dim[0]
    ax_height = ax_dim[1] #/K
    axes = []
    for i in range(rows):
        frac_i = i/rows
        for j in range(cols):
            if i!=0:
                y_offset_add = -y_offset_add_default/K
            else:
                y_offset_add = 0
            frac_j = j/(cols+1)
            if int(i*cols + j) < n_plots:
                rect = [frac_j + x_offset, frac_i + y_offset + i*y_offset_add, ax_width, ax_height]
                axes.append(fig.add_axes(rect))

    axes.reverse()
    fig.suptitle(title_name,fontsize=20)
    return fig, axes

def flip(items, ncol):
    return itertools.chain(*[items[i::ncol] for i in range(ncol)])

def plot_bayesian_inference(intervals, time, ydata, show=True, subset=None, save_plot:bool=False, plotargs=None, single_plot=False, addmodel=True, addmodelinterval=True, addnoisemodel=False, addnoisemodelinterval=False, addMetrics=True, summarizeMetrics = True, ylabels=None):
    load_params()
    new_intervals = []
    new_ydata = []
    metricsList = []

    if subset is not None:
        for ii, interval in enumerate(intervals):
            if interval["id"] in subset:
                new_intervals.append(interval)
                new_ydata.append(ydata[:,ii])
        intervals = new_intervals
        ydata = np.array(new_ydata).transpose()

    facecolor = tuple(list(Colors.beis)+[0.5])
    edgecolor = tuple(list((0,0,0))+[0.1])
    # cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
    # cmap = sns.color_palette("Dark2", as_cmap=True)
    # cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
    # cmap = sns.color_palette("crest", as_cmap=True)

    limits = [99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    n_limits = len(limits)
    n = 5
    cmap = sns.dark_palette((50,50,90), input="husl", reverse=True, n_colors=n_limits+n)# 0,0,74
    cmap = cmap[:-n]

    data_display = dict(
        marker=None,
        color=Colors.red,
        linewidth=1,
        linestyle="solid",
        mfc='none',
        label=r'Observations: $\matrva{Y}$')
    
    model_display = dict(
        color=Colors.blue,
        linestyle="dashed",
        label=f"Model",
        linewidth=2
        )
    
    noisemodel_display = dict(
                        color="black",
                        linestyle="dashed", 
                        label=r"Median of posterior predictive distribution: $Q_{50\%}\Big(\matrva{Y}^p_y\Big)$",
                        linewidth=2
                        )

    interval_display = dict(alpha=None, edgecolor=edgecolor, linestyle="solid")
    
    modelintervalset = dict(
        limits=limits,
        colors=cmap,
        # cmap=cmap,
        alpha=0.5)
    
    noisemodelintervalset = dict(
        limits=limits,
        colors=cmap,
        # cmap=cmap,
        alpha=0.2)
    
    if single_plot:
        figs = []
        axes = []
        for i in range(len(intervals)):
            fig, ax = plt.subplots()
            figs.append(fig)
            axes.append(ax)
    else:
        fig, axes = plt.subplots(len(intervals), ncols=1, sharex=True)
        figs = [fig]*len(intervals)
        axes = [axes] if len(intervals)==1 else axes
    
    for ii, (interval, fig, ax) in enumerate(zip(intervals, figs, axes)):
        id = intervals[ii]["id"]
        fig, ax, metrics = plot_intervals(intervals=interval,
                                            time=time,
                                            ydata=ydata[:,ii],
                                            data_display=data_display,
                                            model_display=model_display,
                                            noisemodel_display=noisemodel_display,
                                            interval_display=interval_display,
                                            modelintervalset=modelintervalset,
                                            noisemodelintervalset=noisemodelintervalset,
                                            fig=fig,
                                            ax=ax,
                                            adddata=True,
                                            addlegend=False,
                                            addmodel=addmodel,
                                            addnoisemodel=addnoisemodel,
                                            addmodelinterval=addmodelinterval,
                                            addnoisemodelinterval=addnoisemodelinterval, ##
                                            figsize=(15, 4))
        pos = ax.get_position()
        pos.x0 = 0.15       # for example 0.2, choose your value
        pos.x1 = 0.99       # for example 0.2, choose your value
        ax.set_position(pos)

        if ylabels is not None and id in ylabels:
            ax.set_ylabel(ylabels[id], color="black", fontsize=14, rotation=0, ha='right', labelpad=10)
        ax.xaxis.label.set_color('black')

        limit_list = []
        inside_fraction_list = []
        mylocator = mdates.HourLocator(interval=6, tz=None)
        ax.xaxis.set_minor_locator(mylocator)
        myFmt = mdates.DateFormatter('%H')
        ax.xaxis.set_minor_formatter(myFmt)

        mylocator = mdates.WeekdayLocator(
            byweekday=[mdates.MO, mdates.TU, mdates.WE, mdates.TH, mdates.FR, mdates.SA, mdates.SU], interval=1,
            tz=None)
        ax.xaxis.set_major_locator(mylocator)
        myFmt = mdates.DateFormatter('%a')
        ax.xaxis.set_major_formatter(myFmt)
        ax.tick_params(axis='x', which='major', pad=10)  # move the tick labels

        if addmodelinterval and addnoisemodelinterval == False:
            text_list = [r'$\mu_{%.0f}=%.2f$' % (
            modelintervalset["limits"][0], metrics["is_inside_fraction_model_list"][0],)]
            for limit, is_inside_fraction in zip(modelintervalset["limits"][1:],
                                                    metrics["is_inside_fraction_model_list"][1:]):
                text_list.append(r'$\mu_{%.0f}=%.2f$' % (limit, is_inside_fraction,))
                limit_list.append(limit)
                inside_fraction_list.append(is_inside_fraction)

        elif addnoisemodelinterval:
            textstr = r'$\mu_{%.0f}=%.2f$' % (
            noisemodelintervalset["limits"][0], metrics["is_inside_fraction_noisemodel_list"][0],)
            text_list = [textstr]
            for limit, is_inside_fraction in zip(noisemodelintervalset["limits"][1:],
                                                    metrics["is_inside_fraction_noisemodel_list"][1:]):
                text_list.append(r'$\mu_{%.0f}=%.2f$' % (limit, is_inside_fraction,))
                limit_list.append(limit)
                inside_fraction_list.append(is_inside_fraction)

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        if addmodelinterval or addnoisemodelinterval:
            textstr = "    ".join(text_list)
            # these are matplotlib.patch.Patch properties
            # place a text box in upper left in axes coords
            if addMetrics:
                ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=props)

        text_list = [r'$\textrm{MAE}=%.2f$' % (metrics["mae"],)]
        text_list.append(r'$\textrm{RMSE}=%.2f$' % (metrics["rmse"],))
        text_list.append(r'$\textrm{CVRMSE}=%.2f$' % (metrics["cvrmse"],))
        text_list.append(r'$\textrm{MAPE}=%.2f$' % (metrics["mape"],))
        text_list.append(r'$\textrm{mean_y}=%.2f$' % (metrics["mean_y"],))
        textstr = "    ".join(text_list)
        if addMetrics:
            ax.text(0.05, 0.70, textstr, transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)

        if summarizeMetrics:
            metricsDict = {
                "ID": id,
                "MAE": metrics["mae"],
                "RMSE": metrics["rmse"],
            }

            i = 0
            # print(inside_fraction_list)
            # print(limit_list)
            for i in range(len(limit_list)):
                metricsDict["ACE" + str(limit_list[i])] = (inside_fraction_list[i]*100-limit_list[i])
                i = i + 2
            # for key in metricsDict:
            #     print(key, metricsDict[key])
            

            metricsList.append(metricsDict)

    if single_plot == False:
        figs[0].subplots_adjust(hspace=0.3)
        figs[0].set_size_inches((17, 10))
        cb = figs[0].colorbar(mappable=None, cmap=matplotlib.colors.ListedColormap(cmap), location="right", ax=axes)
        # cb = fig.colorbar(mappable=None, cmap=matplotlib.colors.ListedColormap(cmap), location="right", ax=ax) 
        cb.set_label(label=r"PI", size=25)#, weight='bold')
        cb.solids.set(alpha=1)
        # fig_trace_beta.tight_layout()
        vmin = 0
        vmax = 1
        dist = (vmax-vmin)/(n_limits)/2
        tick_start = vmin+dist
        tick_end = vmax-dist
        tick_locs = np.linspace(tick_start, tick_end, n_limits)[::-1]
        cb.set_ticks(tick_locs)
        labels = limits
        ticklabels = reversed([str(round(float(label))) + "%" if isinstance(label, str) == False else label for label in
                               labels])  # round(x, 2)
        cb.set_ticklabels(ticklabels, size=12)

        for tick in cb.ax.get_yticklabels():
            tick.set_fontsize(12)
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = 3
        axes[0].legend(flip(handles, ncol), flip(labels, ncol), loc="upper center", bbox_to_anchor=(0.5,1.4), prop={'size': 12}, ncol=ncol)
        axes[-1].set_xlabel("Time")
        if save_plot:
            id = intervals[0]["id"]
            plot_filename = f"bayesian_inference_{id}.png"
            figs[0].savefig(plot_filename, dpi=300)
            plt.close(figs[0])
    else:
        for interval, fig, ax in zip(intervals, figs, axes):
            id = interval["id"]
            fig.suptitle(f"{id}")
            fig.subplots_adjust(hspace=0.3)
            # fig.set_size_inches((15,10))
            cb = fig.colorbar(mappable=None, cmap=matplotlib.colors.ListedColormap(cmap), location="right", ax=ax) 
            cb.set_label(label=r"PI", size=15)#, weight='bold')
            cb.solids.set(alpha=1)
            # fig_trace_beta.tight_layout()
            vmin = 0
            vmax = 1
            dist = (vmax - vmin) / (n_limits) / 2
            tick_start = vmin + dist
            tick_end = vmax - dist
            tick_locs = np.linspace(tick_start, tick_end, n_limits)[::-1]
            cb.set_ticks(tick_locs)
            labels = limits
            ticklabels = reversed(
                [str(round(float(label))) + "%" if isinstance(label, str) == False else label for label in
                 labels])  # round(x, 2)
            cb.set_ticklabels(ticklabels, size=12)

            for tick in cb.ax.get_yticklabels():
                tick.set_fontsize(12)
            
            handles, labels = ax.get_legend_handles_labels()
            ncol = 3
            ax.legend(flip(handles, ncol), flip(labels, ncol), loc="upper center", bbox_to_anchor=(0.5,1.4), prop={'size': 12}, ncol=ncol)
            ax.set_xlabel("Time")

            if save_plot:
                id = interval["id"]
                plot_filename = f"bayesian_inference_{id}.png"
                fig.savefig(plot_filename, dpi=300)
                plt.close(fig)
    if show:
        plt.show()


    if summarizeMetrics:
        return figs, axes, metricsList
    else:
        return figs, axes

# def plot_bayesian_inference(intervals, time, ydata, show=True, subset=None, save_plot:bool=False, plotargs=None, single_plot=False, addmodel=True, addmodelinterval=True, addnoisemodel=False, addnoisemodelinterval=False, addMetrics=True):
#     load_params()

#     new_intervals = []
#     new_ydata = []
#     if subset is not None:
#         for ii, interval in enumerate(intervals):
#             if interval["id"] in subset:
#                 new_intervals.append(interval)
#                 new_ydata.append(ydata[:,ii])
#     intervals = new_intervals
#     ydata = np.array(new_ydata).transpose()
#     facecolor = tuple(list(Colors.beis)+[0.5])
#     edgecolor = tuple(list((0,0,0))+[0.1])
#     # cmap = sns.dark_palette("#69d", reverse=True, as_cmap=True)
#     # cmap = sns.color_palette("Dark2", as_cmap=True)
#     # cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
#     # cmap = sns.color_palette("crest", as_cmap=True)

#     limits = [99, 95, 90, 85, 80, 75, 70, 65, 60, 55, 50]
#     n_limits = len(limits)
#     n = 5
#     cmap = sns.dark_palette((50,50,90), input="husl", reverse=True, n_colors=n_limits+n)# 0,0,74
#     cmap = cmap[:-n]
    
#     data_display = dict(
#         marker=None,
#         color=Colors.red,
#         linewidth=1,
#         linestyle="solid",
#         mfc='none',
#         label=r'Observations: $\matrva{Y}$')
    
#     model_display = dict(
#         color=Colors.blue,
#         linestyle="dashed",
#         label=f"Model",
#         linewidth=2
#         )
    
#     noisemodel_display = dict(
#                         color="black",
#                         linestyle="dashed", 
#                         label=r"Median of posterior predictive distribution: $Q_{50\%}\Big(\matrva{Y}^p_y\Big)$",
#                         linewidth=2
#                         )

#     interval_display = dict(alpha=None, edgecolor=edgecolor, linestyle="solid")
    
#     modelintervalset = dict(
#         limits=limits,
#         colors=cmap,
#         # cmap=cmap,
#         alpha=0.5)
    
#     noisemodelintervalset = dict(
#         limits=limits,
#         colors=cmap,
#         # cmap=cmap,
#         alpha=0.2)
    
#     if single_plot:
#         figs = []
#         axes = []
#         for i in range(len(intervals)):
#             fig, ax = plt.subplots()
#             figs.append(fig)
#             axes.append(ax)
#     else:
#         fig, axes = plt.subplots(len(intervals), ncols=1, sharex=True)
#         figs = [fig]*len(intervals)
#         axes = [axes] if len(intervals)==1 else axes
    
    
    
    

#     for ii, (interval, fig, ax) in enumerate(zip(intervals, figs, axes)):

#         fig, ax, metrics = plot_intervals(intervals=interval,
#                                             time=time,
#                                             ydata=ydata[:,ii],
#                                             data_display=data_display,
#                                             model_display=model_display,
#                                             noisemodel_display=noisemodel_display,
#                                             interval_display=interval_display,
#                                             modelintervalset=modelintervalset,
#                                             noisemodelintervalset=noisemodelintervalset,
#                                             fig=fig,
#                                             ax=ax,
#                                             adddata=True,
#                                             addlegend=False,
#                                             addmodel=addmodel,
#                                             addnoisemodel=addnoisemodel,
#                                             addmodelinterval=addmodelinterval,
#                                             addnoisemodelinterval=addnoisemodelinterval, ##
#                                             figsize=(15, 4))
#         pos = ax.get_position()
#         pos.x0 = 0.15       # for example 0.2, choose your value
#         pos.x1 = 0.99       # for example 0.2, choose your value
#         ax.set_position(pos)

#         if addMetrics:
#             if addmodelinterval and addnoisemodelinterval==False:
#                 text_list = [r'$\mu_{%.0f}=%.2f$' % (modelintervalset["limits"][0], metrics["is_inside_fraction_model_list"][0], )]
#                 for limit, is_inside_fraction in zip(modelintervalset["limits"][1:], metrics["is_inside_fraction_model_list"][1:]):
#                     text_list.append(r'$\mu_{%.0f}=%.2f$' % (limit, is_inside_fraction, ))
#             elif addnoisemodelinterval:
#                 textstr = r'$\mu_{%.0f}=%.2f$' % (noisemodelintervalset["limits"][0], metrics["is_inside_fraction_noisemodel_list"][0], )
#                 text_list = [textstr]
#                 for limit, is_inside_fraction in zip(noisemodelintervalset["limits"][1:], metrics["is_inside_fraction_noisemodel_list"][1:]):
#                     text_list.append(r'$\mu_{%.0f}=%.2f$' % (limit, is_inside_fraction, ))

#             props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#             if addmodelinterval or addnoisemodelinterval:
#                 textstr = "    ".join(text_list)
#                 # these are matplotlib.patch.Patch properties
#                 # place a text box in upper left in axes coords
#                 ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#                 verticalalignment='top', bbox=props)

#             text_list = [r'$\textrm{MAE}=%.2f$' % (metrics["mae"], )]
#             text_list.append(r'$\textrm{RMSE}=%.2f$' % (metrics["rmse"], ))
#             text_list.append(r'$\textrm{CVRMSE}=%.2f$' % (metrics["cvrmse"], ))
#             text_list.append(r'$\textrm{MAPE}=%.2f$' % (metrics["mape"], ))
#             text_list.append(r'$\textrm{mean_y}=%.2f$' % (metrics["mean_y"], ))
#             textstr = "    ".join(text_list)
#             ax.text(0.05, 0.70, textstr, transform=ax.transAxes, fontsize=10,
#             verticalalignment='top', bbox=props)

#         mylocator = mdates.HourLocator(interval=6, tz=None)
#         ax.xaxis.set_minor_locator(mylocator)
#         myFmt = mdates.DateFormatter('%H')
#         ax.xaxis.set_minor_formatter(myFmt)
        
#         mylocator = mdates.WeekdayLocator(byweekday=[mdates.MO, mdates.TU, mdates.WE, mdates.TH, mdates.FR, mdates.SA, mdates.SU], interval=1, tz=None)
#         ax.xaxis.set_major_locator(mylocator)
#         myFmt = mdates.DateFormatter('%a')
#         ax.xaxis.set_major_formatter(myFmt)


#     if single_plot==False:
#         figs[0].subplots_adjust(hspace=0.3)
#         figs[0].set_size_inches((15,10))
#         cb = figs[0].colorbar(mappable=None, cmap=matplotlib.colors.ListedColormap(cmap), location="right", ax=axes)
#         # cb = fig.colorbar(mappable=None, cmap=matplotlib.colors.ListedColormap(cmap), location="right", ax=ax) 
#         cb.set_label(label=r"PI", size=25)#, weight='bold')
#         cb.solids.set(alpha=1)
#         # fig_trace_beta.tight_layout()
#         vmin = 0
#         vmax = 1
#         dist = (vmax-vmin)/(n_limits)/2
#         tick_start = vmin+dist
#         tick_end = vmax-dist
#         tick_locs = np.linspace(tick_start, tick_end, n_limits)[::-1]
#         cb.set_ticks(tick_locs)
#         labels = limits
#         ticklabels = reversed([str(round(float(label)))+"%" if isinstance(label, str)==False else label for label in labels]) #round(x, 2)
#         cb.set_ticklabels(ticklabels, size=12)

#         for tick in cb.ax.get_yticklabels():
#             tick.set_fontsize(12)
#         handles, labels = axes[0].get_legend_handles_labels()
#         ncol = 3
#         axes[0].legend(flip(handles, ncol), flip(labels, ncol), loc="upper center", bbox_to_anchor=(0.5,1.3), prop={'size': 12}, ncol=ncol)
#         axes[-1].set_xlabel("Time")
#         if save_plot:
#             id = intervals[0]["id"]
#             plot_filename = f"bayesian_inference_{id}.png"
#             figs[0].savefig(plot_filename, dpi=300)
#             plt.close(figs[0])
#     else:
#         for interval, fig, ax in zip(intervals, figs, axes):
#             id = interval["id"]
#             fig.suptitle(f"{id}")
#             fig.subplots_adjust(hspace=0.3)
#             # fig.set_size_inches((15,10))
#             cb = fig.colorbar(mappable=None, cmap=matplotlib.colors.ListedColormap(cmap), location="right", ax=ax) 
#             cb.set_label(label=r"PI", size=15)#, weight='bold')
#             cb.solids.set(alpha=1)
#             # fig_trace_beta.tight_layout()
#             vmin = 0
#             vmax = 1
#             dist = (vmax-vmin)/(n_limits)/2
#             tick_start = vmin+dist
#             tick_end = vmax-dist
#             tick_locs = np.linspace(tick_start, tick_end, n_limits)[::-1]
#             cb.set_ticks(tick_locs)
#             labels = limits
#             ticklabels = reversed([str(round(float(label)))+"%" if isinstance(label, str)==False else label for label in labels]) #round(x, 2)
#             cb.set_ticklabels(ticklabels, size=12)

#             for tick in cb.ax.get_yticklabels():
#                 tick.set_fontsize(12)
            
#             handles, labels = ax.get_legend_handles_labels()
#             ncol = 3
#             ax.legend(flip(handles, ncol), flip(labels, ncol), loc="upper center", bbox_to_anchor=(0.5,1.3), prop={'size': 12}, ncol=ncol)
#             ax.set_xlabel("Time")

#             if save_plot:
#                 id = interval["id"]
#                 plot_filename = f"bayesian_inference_{id}.png"
#                 fig.savefig(plot_filename, dpi=300)
#                 plt.close(fig)


#     if show and save_plot==False:
#         plt.show()
#     return figs, axes

# This code has been adapted from the ptemcee package https://github.com/willvousden/ptemcee
def plot_intervals(intervals, time, ydata=None, xdata=None,
                   limits=[95],
                   adddata=None, 
                   addmodel=True, 
                   addnoisemodel=True, 
                   addlegend=True, 
                   addmodelinterval=True, 
                   addnoisemodelinterval=True,
                   data_display={}, 
                   model_display={}, 
                   noisemodel_display={}, 
                   interval_display={},
                   fig=None, 
                   ax=None, 
                   figsize=None, 
                   legloc='upper left',
                   modelintervalset=None, 
                   noisemodelintervalset=None,
                   return_settings=False):
    '''
    Plot propagation intervals in 2-D

    This routine takes the model distributions generated using the
    :func:`~calculate_intervals` method and then plots specific
    quantiles.  The user can plot just the intervals, or also include the
    median model response and/or observations.  Specific settings for
    credible intervals are controlled by defining the `ciset` dictionary.
    Likewise, for prediction intervals, settings are defined using `noisemodelintervalset`.

    The setting options available for each interval are as follows:
        - `limits`: This should be a list of numbers between 0 and 100, e.g.,
          `limits=[50, 90]` will result in 50% and 90% intervals.
        - `cmap`: The program is designed to "try" to choose colors that
          are visually distinct.  The user can specify the colormap to choose
          from.
        - `colors`: The user can specify the color they would like for each
          interval in a list, e.g., ['r', 'g', 'b'].  This list should have
          the same number of elements as `limits` or the code will revert
          back to its default behavior.

    Args:
        * **intervals** (:py:class:`dict`): Interval dictionary generated
          using :meth:`calculate_intervals` method.
        * **time** (:class:`~numpy.ndarray`): Independent variable, i.e.,
          x-axis of plot

    Kwargs:
        * **ydata** (:class:`~numpy.ndarray` or None): Observations, expect
          1-D array if defined.
        * **xdata** (:class:`~numpy.ndarray` or None): Independent values
          corresponding to observations.  This is required if the observations
          do not align with your times of generating the model response.
        * **limits** (:py:class:`list`): Quantile limits that correspond to
          percentage size of desired intervals.  Note, this is the default
          limits, but specific limits can be defined using the `ciset` and
          `noisemodelintervalset` dictionaries.
        * **adddata** (:py:class:`bool`): Flag to include data
        * **addmodel** (:py:class:`bool`): Flag to include median model
          response
        * **addlegend** (:py:class:`bool`): Flag to include legend
        * **addcredible** (:py:class:`bool`): Flag to include credible
          intervals
        * **addprediction** (:py:class:`bool`): Flag to include prediction
          intervals
        * **model_display** (:py:class:`dict`): Display settings for median
          model response
        * **data_display** (:py:class:`dict`): Display settings for data
        * **interval_display** (:py:class:`dict`): General display settings
          for intervals.
        * **fig**: Handle of previously created figure object
        * **figsize** (:py:class:`tuple`): (width, height) in inches
        * **legloc** (:py:class:`str`): Legend location - matplotlib help for
          details.
        * **ciset** (:py:class:`dict`): Settings for credible intervals
        * **noisemodelintervalset** (:py:class:`dict`): Settings for prediction intervals
        * **return_settings** (:py:class:`bool`): Flag to return ciset and
          noisemodelintervalset along with fig and ax.

    Returns:
        * (:py:class:`tuple`) with elements
            1) Figure handle
            2) Axes handle
            3) Dictionary with `ciset` and `noisemodelintervalset` inside (only
               outputted if `return_settings=True`)
    '''

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise("Please supply both fig and ax as arguments")


    if fig is None and ax is None:
        fig, ax = plt.subplots()

    if figsize is not None:
        fig.set_size_inches(figsize)
    

    # unpack dictionary
    noise = intervals['noise']
    model = intervals['model']
    prediction = intervals['prediction']
    

    # Check user-defined settings
    modelintervalset = __setup_iset(modelintervalset,
                                        default_iset=dict(
                                                limits=limits,
                                                cmap=None,
                                                colors=None))
    noisemodelintervalset = __setup_iset(noisemodelintervalset,
                                        default_iset=dict(
                                                limits=limits,
                                                cmap=None,
                                                colors=None))
    
    
    # Check limits
    modelintervalset['limits'] = _check_limits(modelintervalset['limits'], limits)
    noisemodelintervalset['limits'] = _check_limits(noisemodelintervalset['limits'], limits)
    # convert limits to ranges
    modelintervalset['quantiles'] = _convert_limits(modelintervalset['limits'])
    noisemodelintervalset['quantiles'] = _convert_limits(noisemodelintervalset['limits'])
    # setup display settings
    interval_display, model_display, data_display = setup_display_settings(
            interval_display, model_display, data_display)
    # Define colors
    modelintervalset['colors'] = setup_interval_colors(modelintervalset, inttype='ci')
    noisemodelintervalset['colors'] = setup_interval_colors(noisemodelintervalset, inttype='pi')
    # Define labels
    modelintervalset['labels'] = _setup_labels(modelintervalset['limits'], type_='CI')
    noisemodelintervalset['labels'] = _setup_labels(noisemodelintervalset['limits'], type_=None)


    is_inside_fraction_model_list = []
    is_inside_fraction_noisemodel_list = []


    # add model (median model response)
    if addmodel is True:
        # ci = generate_mode(model, n_bins=20)
        if model.shape[0] == 1:
            ci = model[0]
        else:
            ci = generate_quantiles(model, p=np.array([0.5]))[0]
        ax.plot(time, ci, **model_display)

        if addnoisemodel==False:
            rmse = np.sqrt(np.mean((ydata-ci)**2))
            mae = np.mean(np.abs(ydata-ci))
            cvrmse = rmse/np.mean(ydata)
            mean_y = np.mean(ydata)
            non_zero_indices = ydata>0.01
            mape = np.mean(np.abs(ydata[non_zero_indices]-ci[non_zero_indices])/ydata[non_zero_indices])


        # Individual noise samples
        # ax_twin = ax.twinx()
        # for noise_ in noise:
        #     ax_twin.plot(time, noise_, color=Colors.pink)

    if addnoisemodel:
        # pi = generate_mode(prediction, n_bins=20)
        pi = generate_quantiles(prediction, p=np.array([0.5]))[0]
        # pi = generate_mean(prediction)
        ax.plot(time, pi, **noisemodel_display)

        rmse = np.sqrt(np.mean((ydata-pi)**2))
        mae = np.mean(np.abs(ydata-pi))
        cvrmse = rmse/np.mean(ydata)
        mean_y = np.mean(ydata)
        non_zero_indices = ydata>0.01
        mape = np.mean(np.abs(ydata[non_zero_indices]-pi[non_zero_indices])/ydata[non_zero_indices])

        # for pred in prediction:
        #     ax.plot(time, pred, color=Colors.green, alpha=0.3, linewidth=0.5)

    # for i in range(prediction.shape[0]):
    #     ax.plot(time, credible[i,:], color="black", alpha=0.2, linewidth=0.5)
    
    # add data to plot
    if ydata is not None and adddata is None:
        adddata = True
    if adddata is True and ydata is not None:
        if xdata is None:
            ax.plot(time, ydata, **data_display)
        else:
            ax.plot(xdata, ydata, **data_display)

    
            
    # add credible intervals
    if addmodelinterval is True:
        for ii, quantile in enumerate(modelintervalset['quantiles']):
            ci = generate_quantiles(model, np.array(quantile))
            ax.fill_between(time, ci[0], ci[1], facecolor=modelintervalset['colors'][ii],
                            label=modelintervalset['labels'][ii], **interval_display)
            is_inside = np.logical_and(ydata>=ci[0], ydata<=ci[1])
            is_inside_fraction = np.sum(is_inside)/is_inside.size
            is_inside_fraction_model_list.append(is_inside_fraction)

    # time = time.reshape(time.size,)
    # add prediction intervals
    if addnoisemodelinterval is True:
        for ii, quantile in enumerate(noisemodelintervalset['quantiles']):
            pi = generate_quantiles(prediction, np.array(quantile))
            # ax.fill_between(time, pi[0], pi[1], facecolor=noisemodelintervalset['colors'][ii],
            #                 label=noisemodelintervalset['labels'][ii], **interval_display)
            ax.fill_between(time, pi[0], pi[1], facecolor=noisemodelintervalset['colors'][ii], **interval_display)
            is_inside = np.logical_and(ydata>=pi[0], ydata<=pi[1])
            is_inside_fraction = np.sum(is_inside)/is_inside.size
            is_inside_fraction_noisemodel_list.append(is_inside_fraction)

    metrics = dict(rmse=rmse, cvrmse=cvrmse, mae=mae, mean_y=mean_y, mape=mape, is_inside_fraction_model_list=is_inside_fraction_model_list, is_inside_fraction_noisemodel_list=is_inside_fraction_noisemodel_list)
    # add legend
    if addlegend is True:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=legloc)

    if return_settings is True:
        return fig, ax, metrics, dict(modelintervalset=modelintervalset, noisemodelintervalset=noisemodelintervalset)
    else:
        return fig, ax, metrics


def plot_ls_inference(predictions, time, ydata, targetMeasuringDevices, show=True):
    """
    Plot the results of a least squares inference.
    
    :param predictions: Predicted values from the model. The shape of the predictions is (n, 4) where n is the number of time steps. The columns are the same as the contents of targetMeasuringDevices.
    :param time: Time steps for the predictions.
    :param ydata: Actual observed data.
    :param targetMeasuringDevices: Target measuring devices used in the model.
    :param show: Whether to display the plot.
    :param plotargs: Additional arguments for plotting.
    :return: The figure and axes of the plot.
    """
    # Load parameters and set up styles (if needed)
    # load_params()  # Uncomment if you have a function to set up plot parameters

    # Define display properties
    data_display = dict(
        marker=None,
        color=Colors.red,
        linewidth=1,
        linestyle="solid",
        mfc='none',
        label='Observed')
    model_display = dict(
        color="black",
        linestyle="dashed", 
        label=f"Estimate",
        linewidth=1
        )
    # Create a figure and axes
    fig, axes = plt.subplots(len(targetMeasuringDevices), ncols=1, figsize=(10, 6))
    if len(targetMeasuringDevices) == 1:
        axes = [axes]  # Ensure axes is always a list

    for i, (measuring_device, ax) in enumerate(zip(targetMeasuringDevices, axes)):
        # Plot observed data
        ax.plot(time, ydata[:, i], **data_display)

        # Plot model predictions
        ax.plot(time, predictions[:, i], **model_display)

        # Formatting
        ax.set_title(f"Measuring Device: {measuring_device.id}")
        ax.set_ylabel("Value")
        myFmt = mdates.DateFormatter('%H:%M')
        ax.xaxis.set_major_formatter(myFmt)

    axes[-1].set_xlabel("Time")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5,1.1), prop={'size': 12}, ncol=2)
    # Make the graphs background light grey, add a grid and make sure every component is tight
    for ax in axes:
        ax.set_facecolor('lightgrey')
        ax.grid()
        ax.autoscale(enable=True, axis='x', tight=True)
        
    # Adjust the layout to fit all titles and labels
    fig.tight_layout()
    plt.show()

    if show:
        plt.show()

    return fig, axes


def check_settings(default_settings, user_settings=None):
    '''
    Check user settings with default.

    Recursively checks elements of user settings against the defaults and updates settings
    as it goes.  If a user setting does not exist in the default, then the user setting
    is added to the settings.  If the setting is defined in both the user and default
    settings, then the user setting overrides the default.  Otherwise, the default
    settings persist.

    Args:
        * **default_settings** (:py:class:`dict`): Default settings for particular method.
        * **user_settings** (:py:class:`dict`): User defined settings.

    Returns:
        * (:py:class:`dict`): Updated settings.
    '''
    settings = default_settings.copy()  # initially define settings as default
    options = list(default_settings.keys())  # get default settings
    if user_settings is None:  # convert to empty dict
        user_settings = {}
    user_options = list(user_settings.keys())  # get user settings
    for uo in user_options:  # iterate through settings
        if uo in options:
            # check if checking a dictionary
            if isinstance(settings[uo], dict):
                settings[uo] = check_settings(settings[uo], user_settings[uo])
            else:
                settings[uo] = user_settings[uo]
        if uo not in options:
            settings[uo] = user_settings[uo]
    return settings

def setup_display_settings(interval_display, model_display, data_display):
    '''
    Compare user defined display settings with defaults and merge.

    Args:
        * **interval_display** (:py:class:`dict`): User defined settings for interval display.
        * **model_display** (:py:class:`dict`): User defined settings for model display.
        * **data_display** (:py:class:`dict`): User defined settings for data display.

    Returns:
        * **interval_display** (:py:class:`dict`): Settings for interval display.
        * **model_display** (:py:class:`dict`): Settings for model display.
        * **data_display** (:py:class:`dict`): Settings for data display.
    '''
    # Setup interval display
    default_interval_display = dict(
            linestyle=':',
            linewidth=1,
            alpha=1.0,
            edgecolor='k')
    interval_display = check_settings(default_interval_display, interval_display)
    # Setup model display
    default_model_display = dict(
            linestyle='-',
            color='k',
            marker='',
            linewidth=2,
            markersize=5,
            label='Model')
    model_display = check_settings(default_model_display, model_display)
    # Setup data display
    default_data_display = dict(
            linestyle='',
            color='b',
            marker='.',
            linewidth=1,
            markersize=5,
            label='Data')
    data_display = check_settings(default_data_display, data_display)
    return interval_display, model_display, data_display


def setup_interval_colors(iset, inttype='CI'):
    '''
    Setup colors for empirical intervals

    This routine attempts to distribute the color of the UQ intervals
    based on a normalize color map.  Or, it will assign user-defined
    colors; however, this only happens if the correct number of colors
    are specified.

    Args:
        * **iset** (:py:class:`dict`):  This dictionary should contain the
          following keys - `limits`, `cmap`, and `colors`.

    Kwargs:
        * **inttype** (:py:class:`str`): Type of uncertainty interval

    Returns:
        * **ic** (:py:class:`list`): List containing color for each interval
    '''
    limits, cmap, colors = iset['limits'], iset['cmap'], iset['colors']
    norm = __setup_cmap_norm(limits)
    cmap = __setup_default_cmap(cmap, inttype)
    # assign colors using color map or using colors defined by user
    ic = []
    if colors is None:  # No user defined colors
        for limits in limits:
            ic.append(cmap(norm(limits)))
    else:
        if len(colors) == len(limits):  # correct number of colors defined
            for color in colors:
                ic.append(color)
        else:  # User defined the wrong number of colors
            print('Note, user-defined colors were ignored. Using color map. '
                  + 'Expected a list of length {}, but received {}'.format(
                          len(limits), len(colors)))
            for limits in limits:
                ic.append(cmap(norm(limits)))
    return ic


def _setup_labels(limits, type_='CI'):
    '''
    Setup labels for prediction/credible intervals.
    '''
    labels = []
    for limit in limits:
        if type_ is None:
            labels.append(str('{}%'.format(limit)))
        else:
            labels.append(str('{}% {}'.format(limit, type_)))
    return labels


def _check_limits(limits, default_limits):
    if limits is None:
        limits = default_limits
    limits.sort(reverse=True)
    return limits


def _convert_limits(limits):
    rng = []
    for limit in limits:
        limit = limit/100
        rng.append([0.5 - limit/2, 0.5 + limit/2])
    return rng


def __setup_iset(iset, default_iset):
    '''
    Setup interval settings by comparing user input to default
    '''
    if iset is None:
        iset = {}
    iset = check_settings(default_iset, iset)
    return iset


def __setup_cmap_norm(limits):
    if len(limits) == 1:
        norm = mplcolor.Normalize(vmin=0, vmax=100)
    else:
        norm = mplcolor.Normalize(vmin=min(limits), vmax=max(limits))
    return norm


def __setup_default_cmap(cmap, inttype):
    if cmap is None:
        if inttype.upper() == 'CI':
            cmap = cm.autumn
        else:
            cmap = cm.winter
    return cmap

#---------------------------------------
def get_attr_list(model: model.Model, subset=None):
    '''This function takes a model, the model should contain a chain_log, otherwise it does not work 
    0, number of steps
    1, number of temperatures
    2, number of walkers
    3, number of parameters
    '''

    component_id = np.array(model.result["component_id"])
    attr_list = np.array(model.result["component_attr"])

    if subset is None:
        subset = list(component_id)
    l = [(component_id[i], attr_list[i], i) for i in range(len(component_id)) if component_id[i] in subset]
    attr_list = [x[1] for x in l]
    component_id = [x[0] for x in l]
    idx = np.array([x[2] for x in l])
    model.result["component_id"] = component_id
    model.result["component_attr"] = attr_list
    # model.result["theta_mask"] = theta_mask
    x = model.result["chain_x"]
    x = x[:, :, :, model.result["theta_mask"]]
    model.result["chain_x"] = x[:, :, :, idx]


    ####################################################
    # Temp fix to only plot the first 2 variables
    # model.result["chain_x"] = model.result["chain_x"][:, :, :, :2]
    # attr_list = attr_list[:2]
    ####################################################


    # records_array = np.array(model.result["theta_mask"])
    # vals, inverse, count = np.unique(records_array, return_inverse=True,
    #                           return_counts=True)
    # idx_vals_repeated = np.where(count > 1)[0]
    # vals_repeated = vals[idx_vals_repeated]
    # rows, cols = np.where(inverse == idx_vals_repeated[:, np.newaxis])
    # _, inverse_rows = np.unique(rows, return_index=True)
    # res = np.split(cols, inverse_rows[1:])
    # d_idx = []
    # for i in res:
    #     d_idx.extend(list(i[1:]))

    # component_id = np.array(model.result["component_id"])
    # attr_list = np.array(model.result["component_attr"])
    # attr_list = np.delete(attr_list, np.array(d_idx).astype(int)) #res is an array of duplicates, so its size should always be larger than 1
    # component_id = np.delete(component_id, np.array(d_idx).astype(int))
    # model.result["chain_x"] = np.delete(model.result["chain_x"], np.array(d_idx).astype(int), axis=3)

    
    # model.result["component_id"] = component_id
    # model.result["component_attr"] = attr_list


    return attr_list

def logl_plot(model: model.Model, show=True):
    'The function shows a logl-plot from a model, the model needs to have estimated parameters'

    ntemps = model.result["chain_x"].shape[1]
    nwalkers = model.result["chain_x"].shape[2]

    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")

    fig_logl, ax_logl = plt.subplots(layout='compressed')
    fig_logl.set_size_inches((17 / 4, 12 / 4))
    fig_logl.suptitle("Log-likelihood", fontsize=20)
    logl = model.result["chain.logl"]
    logl[np.abs(logl) > 1e+9] = np.nan

    indices = np.where(logl[:, 0, :] == np.nanmax(logl[:, 0, :]))
    s0 = indices[0][0]
    s1 = indices[1][0]

    n_it = model.result["chain.logl"].shape[0]
    for i_walker in range(nwalkers):
        for i in range(ntemps):
            # if i_walker == 0:  #######################################################################
            ax_logl.plot(range(n_it), logl[:, i, i_walker], color=cm_sb[i])

    if show:
        plt.show()


def trace_plot(model, n_subplots=20, one_plot=False, burnin=0, max_cols=3, save_plot=False, file_name='TracePlot_2', subset=None, show=True, do_iac_plot=True, correlationFactor = 10, plot_title = 'Trace Plot'):

    flat_attr_list_ = get_attr_list(model, subset)

    ntemps = model.result["chain_x"].shape[1]
    nwalkers = model.result["chain_x"].shape[2]

    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")
    cm_sb_rev = list(reversed(cm_sb))
    cm_mpl_rev = LinearSegmentedColormap.from_list("seaborn_rev", cm_sb_rev, N=ntemps)

    vmin = np.min(model.result["chain.betas"])
    vmax = np.max(model.result["chain.betas"])
    burnin = burnin

    chain_logl = model.result["chain.logl"]
    bool_ = chain_logl < -5e+9
    chain_logl[bool_] = np.nan
    chain_logl[np.isnan(chain_logl)] = np.nanmin(chain_logl)

    num_attributes = len(flat_attr_list_)
    max_cols = max_cols

    if one_plot:
        n_subplots = len(flat_attr_list_)

    for start in range(0, num_attributes, n_subplots):
        end = min(start + n_subplots, num_attributes)
        current_attrs = flat_attr_list_[start:end]
        num_current_attrs = len(current_attrs)

        num_cols = max_cols
        num_rows = math.ceil(num_current_attrs / num_cols)

        fig, axes_trace = plt.subplots(num_rows, num_cols)
        fig.set_size_inches(22, 12)

        # Ensure axes_trace is always a 2D array
        if num_rows == 1 and num_cols == 1:
            axes_trace = np.array([[axes_trace]])
        elif num_rows == 1 or num_cols == 1:
            axes_trace = axes_trace.reshape((num_rows, num_cols))

        for nt in reversed(range(ntemps)):
            for nw in range(nwalkers):
                x = model.result["chain_x"][:, nt, nw, :]
                beta = model.result["chain.betas"][:, nt]

                for j, attr in enumerate(current_attrs):
                    row, col = divmod(j, num_cols)
                    ax = axes_trace[row, col]
                    if ntemps > 1:
                        sc = ax.scatter(range(x[:, start + j].shape[0]), x[:, start + j], c=beta, vmin=vmin, vmax=vmax,
                                        s=0.3, cmap=cm_mpl_rev, alpha=0.1)
                    else:
                        sc = ax.scatter(range(x[:, start + j].shape[0]), x[:, start + j], s=0.3, color=cm_sb[0],
                                        alpha=0.1)
                    ax.axvline(burnin, color="black", linewidth=1, alpha=0.8)

        if do_iac_plot:
            axes_iac = np.empty_like(axes_trace, dtype=object)
            for j in range(num_current_attrs):
                row, col = divmod(j, num_cols)
                axes_iac[row, col] = axes_trace[row, col].twinx()

            iac = model.result["integratedAutoCorrelatedTime"][:-1]
            n_it = iac.shape[0]
            for i in range(ntemps):
                beta = model.result["chain.betas"][:, i]
                for j, attr in enumerate(current_attrs):
                    row, col = divmod(j, num_cols)
                    if ntemps > 1:
                        axes_iac[row, col].plot(range(n_it), iac[:, i, j], color='red', alpha=1, zorder=1)
                    else:
                        axes_iac[row, col].plot(range(n_it), iac[:, i, j], color='red', alpha=1, zorder=1)

            heuristic_line = np.arange(n_it) / correlationFactor
            for j, attr in enumerate(current_attrs):
                row, col = divmod(j, num_cols)
                axes_iac[row, col].plot(range(n_it), heuristic_line, color="black", linewidth=1, linestyle="dashed", alpha=1, label=r"$\tau=N/50$")
                axes_iac[row, col].set_ylim([0 - 0.05 * iac.max(), iac.max() + 0.05 * iac.max()])

        x_left = 0.1
        x_mid_left = 0.515
        x_right = 0.9
        x_mid_right = 0.58
        dx_left = x_mid_left - x_left
        dx_right = x_right - x_mid_right

        fontsize = 12
        for j, attr in enumerate(current_attrs):
            row, col = divmod(j, num_cols)
            ax = axes_trace[row, col]
            ax.axvline(burnin, color="black", linestyle=":", linewidth=1.5, alpha=0.5)
            y = np.array([-np.inf, np.inf])
            x1 = -burnin
            ax.fill_betweenx(y, x1, x2=0)
            ax.text(x_left + dx_left / 2, 0.44, 'Burn-in', ha='center', va='center',
                    rotation='horizontal', fontsize=fontsize, transform=ax.transAxes)

            ax.text(x_mid_right + dx_right / 2, 0.44, 'Posterior', ha='center', va='center',
                    rotation='horizontal', fontsize=fontsize, transform=ax.transAxes)

            ax.set_ylabel(attr, fontsize=20)
            ax.ticklabel_format(style='plain', useOffset=False)
            # ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_y_ticks))  # Apply the formatter here

        if ntemps > 1:
            cb = fig.colorbar(sc, ax=axes_trace.ravel().tolist())
            cb.set_label(label=r"$T$", size=30)
            cb.solids.set(alpha=1)
            dist = (vmax - vmin) / (ntemps) / 2
            tick_start = vmin + dist
            tick_end = vmax - dist
            tick_locs = np.linspace(tick_start, tick_end, ntemps)[::-1]
            cb.set_ticks(tick_locs)
            labels = list(model.result["chain.T"][0, :])
            inf_label = r"$\infty$"
            labels[-1] = inf_label
            ticklabels = [str(round(float(label), 1)) if not isinstance(label, str) else label for label in labels]
            cb.set_ticklabels(ticklabels, size=12)

            for tick in cb.ax.get_yticklabels():
                tick.set_fontsize(12)
                txt = tick.get_text()
                if (txt == inf_label):
                    tick.set_fontsize(20)

        # Add a title to the plot
        fig.suptitle(plot_title, fontsize=24)

        # Adjust the layout
        plt.tight_layout(pad=2.0, h_pad=2.0, w_pad=2.0)
        fig.subplots_adjust(top=0.9)  # Adjust the top to make space for the title

        if save_plot:
            fig.savefig(file_name + str(start + 1) + ".png")
            plt.close(fig)

        if ntemps == 1:
            plt.tight_layout()
    if show:
        plt.show()

def corner_plot(model, subsample_factor=None, burnin:int=0, save_plot:bool=False,
                            file_name="corner_plot", param_blocks=None, subset=None, show=True, labels=None):
    """
    Makes a corner plot for every parameter block on the same plot. The dataset can be thinned by using: subsample_factor,
    this will take the n-th datapoint.
    """
    load_params()
    burnin = burnin
    flat_attr_list_ = get_attr_list(model, subset)
    ntemps = model.result["chain_x"].shape[1]
    
    if labels is not None:
        assert len(labels) == len(flat_attr_list_), f"The length of the labels ({len(labels)}) does not match the number of parameters ({len(flat_attr_list_)})"
        flat_attr_list_ = labels

    cm_sb = sns.diverging_palette(210, 0, s=50, l=50, n=ntemps, center="dark")

    parameter_chain = model.result["chain_x"][burnin:, 0, :, :]
    if subsample_factor is not None:
        parameter_chain = parameter_chain[::subsample_factor]
    parameter_chain = parameter_chain.reshape(parameter_chain.shape[0] * parameter_chain.shape[1],
                                              parameter_chain.shape[2])
    

    if param_blocks is not None:
        num_params = parameter_chain.shape[1]
        num_full_blocks = num_params // param_blocks
        remaining_params = num_params % param_blocks
        block_indices = [range(i * param_blocks, (i + 1) * param_blocks) for i in range(num_full_blocks)]

        if remaining_params > 0:
            block_indices.append(range(num_full_blocks * param_blocks, num_params))

        for i in range(len(block_indices)):
            for j in range(i + 1, len(block_indices)):
                block1 = parameter_chain[:, list(block_indices[i])]
                block2 = parameter_chain[:, list(block_indices[j])]

                fig_corner = corner.corner(np.hstack((block1, block2)), 
                                           fig=None, labels=[flat_attr_list_[idx] for idx in
                                                                                          list(block_indices[i]) + list(
                                                                                              block_indices[j])],
                                           labelpad=-0.3, 
                                           show_titles=True, 
                                           color=cm_sb[0], 
                                           plot_contours=True,
                                           bins=15, 
                                           hist_bin_factor=5, 
                                           max_n_ticks=3, 
                                           quantiles=[0.16, 0.5, 0.84],
                                           title_kwargs={"fontsize": 10, "ha": "left", "position": (0.03, 1.01)},
                                           title_fmt=".2E")
                fig_corner.set_size_inches((16, 16))
                pad = 0.12
                fig_corner.subplots_adjust(left=pad, bottom=pad, right=1 - pad, top=1 - pad, wspace=0.15, hspace=0.15)
                axes = fig_corner.get_axes()
                for ax in axes:
                    ax.set_xticks([], minor=True)
                    ax.set_xticks([])
                    ax.set_yticks([], minor=True)
                    ax.set_yticks([])
                    ax.xaxis.set_ticklabels([])
                    ax.yaxis.set_ticklabels([])

                median = np.median(np.hstack((block1, block2)), axis=0)
                corner.overplot_lines(fig_corner, median, color='red', linewidth=0.5)
                corner.overplot_points(fig_corner, median.reshape(1, median.shape[0]), marker="s", color='red')
                plt.suptitle(f"{file_name}_block{i}_block{j}")

                if save_plot:
                    fig_corner.savefig(f"{file_name}_block{i}_block{j}.png")
                    plt.close(fig_corner)
    else:
        fig_corner = corner.corner(parameter_chain, fig=None, labels=flat_attr_list_, labelpad=-0.25, show_titles=True,
                                   color=cm_sb[0], plot_contours=True, bins=15, hist_bin_factor=5, max_n_ticks=3,
                                   quantiles=[0.16, 0.5, 0.84],
                                   title_kwargs={"fontsize": 10, "ha": "left", "position": (0.03, 1.01)})

        # data_kwargs = {"ms": 5} #Markersize of points
        # hist_kwargs = {"linewidth": 2} #Linewidth of histogram
        # fig_corner = corner.corner(parameter_chain, fig=None, labels=flat_attr_list_, labelpad=-0.29, show_titles=True,
        #                            color=cm_sb[0], plot_contours=True, bins=15, hist_bin_factor=5, max_n_ticks=3,
        #                            quantiles=[0.16, 0.5, 0.84],
        #                            title_kwargs={"fontsize": 30, "ha": "left", "position": (0.03, 1.01)}, data_kwargs=data_kwargs, hist_kwargs=hist_kwargs) # Used for plotting zoom-in plots
        fig_corner.set_size_inches((12, 12))
        pad = 0.025
        pad = 0.05
        fig_corner.subplots_adjust(left=pad, bottom=pad, right=1 - pad, top=1 - pad, wspace=0.08, hspace=0.08)
        axes = fig_corner.get_axes()
        for ax in axes:
            ax.set_xticks([], minor=True)
            ax.set_xticks([])
            ax.set_yticks([], minor=True)
            ax.set_yticks([])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])

            ax.xaxis.label.set_size(35) ################
            ax.yaxis.label.set_size(35) ################


        median = np.median(parameter_chain, axis=0)
        corner.overplot_lines(fig_corner, median, color='red', linewidth=0.5)
        corner.overplot_points(fig_corner, median.reshape(1, median.shape[0]), marker="s", color='red')
        # corner.overplot_points(fig_corner, median.reshape(1, median.shape[0]), marker="s", color='red', ms=20) # Used for plotting zoom-in plots



        if save_plot == True:
            fig_corner.savefig(file_name + ".png")
    if show:
        plt.show()