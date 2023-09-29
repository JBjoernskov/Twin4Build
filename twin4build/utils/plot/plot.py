import matplotlib.dates as mdates
import matplotlib.pylab as pylab
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
from twin4build.utils.plot.align_y_axes import alignYaxes
import sys
from matplotlib import cm
from matplotlib import colors as mplcolor
from scipy.interpolate import interp1d


global global_colors
global global_blue
global global_orange
global global_green
global global_red
global global_purple
global global_brown
global global_pink
global global_grey
global global_beis
global global_sky_blue
global global_legend_loc
global global_outward
global_colors = sns.color_palette("deep")
global_blue = global_colors[0]
global_orange = global_colors[1]
global_green = global_colors[2]
global_red = global_colors[3]
global_purple = global_colors[4]
global_brown = global_colors[5]
global_pink = global_colors[6]
global_grey = global_colors[7]
global_beis = global_colors[8]
global_sky_blue = global_colors[9]
global_legend_loc = (0.5,0.93)
global_x = (0.45, 0.05)
global_left_y = (0.025, 0.50)
global_right_y_first = (0.86, 0.50)
global_right_y_second = (0.975, 0.50)
global_outward = 68

def load_params():
    
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
            "legend.loc": "upper right",
            "legend.fancybox": False,
            "legend.facecolor": "white",
            "legend.framealpha": 1,
            "legend.edgecolor": "black"
            }

    plt.style.use("ggplot")
    pylab.rcParams.update(params)
    plt.rc('font', family='serif')
    font = {"fontname": "serif"}

def get_fig_axes(title_name, n_plots=1, cols=1):
    fig = plt.figure()
    K = 0.38
    fig.set_size_inches(8,4.3)
    rows = math.ceil(n_plots/cols)
    x_offset = 0.12
    y_offset = 0.18
    ax_width = 0.65
    ax_height = 0.6
    axes = []
    for i in range(rows):
        frac_i = i/rows
        for j in range(cols):
            if i!=0:
                y_offset_add = -0.04/K
            else:
                y_offset_add = 0
            frac_j = j/(cols+1)
            if int(i*cols + j) < n_plots:
                rect = [frac_j + x_offset, frac_i + y_offset + i*y_offset_add, ax_width, ax_height]
                axes.append(fig.add_axes(rect))

    axes.reverse()
    fig.suptitle(title_name,fontsize=20)

    return fig, axes

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

def test_plot(model, simulator):

    global_colors = sns.color_palette("deep")
    global_blue = global_colors[0]
    global_orange = global_colors[1]
    global_green = global_colors[2]
    global_red = global_colors[3]
    global_purple = global_colors[4]
    global_brown = global_colors[5]
    global_pink = global_colors[6]
    global_grey = global_colors[7]
    global_beis = global_colors[8]
    global_sky_blue = global_colors[9]
    

    space_name_list = ["Ø20-601b-2"]

    fig = plt.figure()
    fig.set_size_inches(15,5) 

    n_plots = 1#NN_output_matrix.shape[1] 20 ######################################################################################## 4
    cols = 1 ################################################# 4
    rows = math.ceil(n_plots/cols)
    # fig.suptitle("Winter Period, 24-Hour Forecast", fontsize=60)

    x_offset = 0.07
    y_offset = 0.17
    ax_width = 0.23
    ax_height = 0.5
    ax_room = []
    for i in range(rows):
        frac_i = i/rows
        for j in range(cols):
            if j!=0:
                x_offset_add = -0.08
            else:
                x_offset_add = 0
            frac_j = j/(cols+1)
            if int(i*cols + j) < n_plots:
                # ax_room.append(fig.add_subplot(grid[i, j]))#, xticklabels=[])#, sharey=main_ax)
                # ax_room.append(fig.add_subplot(rows, cols+10, int(i*cols + j + 1)))#, xticklabels=[])#, sharey=main_ax)
                rect = [frac_j + x_offset + j*x_offset_add, frac_i + y_offset, ax_width, ax_height]
                ax_room.append(fig.add_axes(rect))

    frac_i = i/rows
    frac_j = cols/(cols+1)
    rect = [1-ax_width-x_offset-0.01, frac_i + y_offset, ax_width, ax_height]
    ax_weather = fig.add_axes(rect)

    # Plotting
    line_list = []
    for i,ax_i in enumerate(ax_room):
        
        row = math.floor(i/cols)
        col = int(i-cols*row)


        space_name = space_name_list[i]
        indoor_temperature_setpoint_schedule_name = "Temperature setpoint schedule"
        weather_station_name = "Outdoor environment"
        
    
        ax_i.set_title(space_name_list[i],fontsize=25)


        ax_i.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedOutput["indoorTemperature"], color="black",label="Temperature predicted", linestyle="dashed")
        ax_i.plot(simulator.dateTimeSteps, model.component_dict[indoor_temperature_setpoint_schedule_name].savedOutput["scheduleValue"], color=global_brown,label="Temperature setpoint", linestyle="dashed")

        ax_twin_0_1 = ax_i.twinx()


        # ax_twin_CO2 = ax_i.twinx()
        # ax_twin_CO2.set_ylim([-50, 1050])
        
        


        ax_twin_0_1.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["valvePosition"], color=global_red, label = "Valve position")
        ax_twin_0_1.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["damperPosition"], color=global_blue, label = "Damper position")
        ax_twin_0_1.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["shadePosition"], color=global_sky_blue, label = "Shades position")
        # ax_i.legend()
        # ax_i.set_ylim([20, 24]) #Winter
        ax_i.set_ylim([18, 29]) #Summer
        ax_twin_0_1.set_ylim([-0.05, 1.05])

        lines_labels1 = ax_i.get_legend_handles_labels()
        lines_labels2 = ax_twin_0_1.get_legend_handles_labels()
        # lines_labels3 = ax_twin_CO2.get_legend_handles_labels()
        lines_labels = [lines_labels1, lines_labels2]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        line_list.append(lines)


        # ax_i.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
        # ax_i.xaxis.set_tick_params(rotation=0)
        # # ticklabels = ["%s" % (i) if len("%s" % (i))==2 else "0%s" % (i) for i in range(24)]
        # # 
        # locator = mdates.HourLocator(byhour=range(0,25,2))
        # ax_i.xaxis.set_major_locator(locator)

        if col>0:
            for tick in ax_i.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            
        
        if col<cols-1:
            # for tick in ax_twin_CO2.yaxis.get_major_ticks():
            #     tick.tick1line.set_visible(False)
            #     tick.tick2line.set_visible(False)
            #     tick.label1.set_visible(False)
            #     tick.label2.set_visible(False)

            for tick in ax_twin_0_1.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            
        # else:
        #     offset = 90
        #     ax_twin_CO2.spines['right'].set_position(('outward', offset))
        #     ax_twin_CO2.spines["right"].set_visible(True)
        #     ax_twin_CO2.spines["right"].set_color(global_orange)
        #     ax_twin_CO2.tick_params(axis='y', colors=global_orange) 
            

        if row!=rows-1:
            for tick in ax_i.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        # formatter = mdates.DateFormatter(r"%m/%d %H")
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            # label.set_ha("right")
            label.set_ha("center")
            label.set_rotation(0) 

        # ax_i.get_xaxis().set_tick_params(labelright=True, labelrotation=45)

        # ax_i.set_xticklabels(ax_i.get_xticks(), rotation = 45,ha='right')



    # fig.text(0.015, 0.43, r"$\mathrm{Temperature [^\circ C]}$", va='center', ha='center', rotation='vertical', fontsize=40)
    # fig.text(0.655, 0.43, r"$\mathrm{CO_{2} [ppm]}$", va='center', ha='center', rotation='vertical', fontsize=40, color = global_orange)
    # fig.text(0.325, 0.05, r"$\mathrm{Hour \; of \; day}$", va='center', ha='center', rotation='horizontal', fontsize=40)

    # fig.text(0.71, 0.43, r"$\mathrm{Temperature [^\circ C]}$", va='center', ha='center', rotation='vertical', fontsize=40)
    # fig.text(0.975, 0.43, r"$\mathrm{Radiation [W/m^2]}$", va='center', ha='center', rotation='vertical', fontsize=40)
    # fig.text(0.84, 0.05, r"$\mathrm{Hour \; of \; day}$", va='center', ha='center', rotation='horizontal', fontsize=40)

    

    fig.text(0.015, 0.45, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=20)
    # fig.text(0.655, 0.43, r"Position", va='center', ha='center', rotation='vertical', fontsize=40, color = global_orange)
    fig.text(0.325, 0.05, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=20)

    fig.text(0.63, 0.45, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=20)
    fig.text(0.985, 0.45, r"Irradiance [W/m$^2$]", va='center', ha='center', rotation='vertical', fontsize=20)
    fig.text(0.82, 0.05, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=20)


    ax_weather.set_title("Weather input",fontsize=20)
    ax_weather.plot(simulator.dateTimeSteps, model.component_dict[weather_station_name].savedOutput["outdoorTemperature"], color=global_green, label = "Outdoor air temperature")

    ax_weather_twin = ax_weather.twinx()
    ax_weather_twin.plot(simulator.dateTimeSteps, np.array(model.component_dict[weather_station_name].savedOutput["globalIrradiation"])/3.6, color=global_red, label = "Shortwave solar irradiance")

    ax_weather.set_ylim([-5, 35])
    # ax_weather.set_ylim([18, 29])
    ax_weather_twin.set_ylim([-50, 1050])

    # formatter = mdates.DateFormatter(r"%m/%d %H")
    formatter = mdates.DateFormatter(r"%H")
    ax_weather.xaxis.set_major_formatter(formatter)
    for label in ax_weather.get_xticklabels():
        # label.set_ha("right")
        label.set_ha("center")
        label.set_rotation(0) 
    # ax_weather.set_xticklabels(ax_weather.get_xticks(), rotation = 45)

    lines_labels1 = ax_i.get_legend_handles_labels()
    lines_labels2 = ax_twin_0_1.get_legend_handles_labels()
    # lines_labels3 = ax_twin_CO2.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=3, loc = "upper center", bbox_to_anchor=(0.35,0.97))
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(20)
        graphs[legend_lines[i]] = []
        for lines in line_list:
            graphs[legend_lines[i]].append(lines[i])


    lines_labels1 = ax_weather.get_legend_handles_labels()
    lines_labels2 = ax_weather_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=1, loc = "upper center", bbox_to_anchor=(0.84,0.97))
    legend_lines = legend.get_lines()
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(20)
        graphs[legend_lines[i]] = [lines[i]]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        # graphs[legend].set_visible(not isVisible)
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
            
        legend.set_visible(not isVisible)
        # legend.set_alpha(1.0 if not isVisible else 0.2)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)




     ##############################################################################################################################################
    fig = plt.figure()
    # fig.suptitle(self.id, fontsize=60)
    # figManager = plt.get_current_fig_manager() ################
    # figManager.window.showMaximized() #######################
    fig.set_size_inches(15,5) 

    ax_DELTA = []

    for i in range(rows):
        frac_i = i/rows
        for j in range(cols):
            if j!=0:
                x_offset_add = -0.08
            else:
                x_offset_add = 0
            frac_j = j/(cols+1)
            if int(i*cols + j) < n_plots:
                # ax_room.append(fig.add_subplot(grid[i, j]))#, xticklabels=[])#, sharey=main_ax)
                # ax_room.append(fig.add_subplot(rows, cols+10, int(i*cols + j + 1)))#, xticklabels=[])#, sharey=main_ax)
                rect = [frac_j + x_offset + j*x_offset_add, frac_i + y_offset, ax_width, ax_height]
                ax_DELTA.append(fig.add_axes(rect))

    frac_i = i/rows
    frac_j = cols/(cols+1)
    rect = [1-ax_width-x_offset-0.01, frac_i + y_offset, ax_width, ax_height]
    ax_weather = fig.add_axes(rect)

    # Plotting
    for i,ax_i in enumerate(ax_DELTA):
        row = math.floor(i/cols)
        col = int(i-cols*row)
        space_name = space_name_list[i]
        ax_i.set_title(space_name_list[i],fontsize=25)
        ax_i.fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,0], color=global_green, alpha=0.5)
        ax_i.fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,1], color=global_orange, alpha=0.5)
        ax_i.fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,2], color=global_red, alpha=0.5)
        ax_i.fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,3], color=global_blue, alpha=0.5)
        ax_i.set_ylim([-0.05, 0.05])

        if col>0:
            for tick in ax_i.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)
            
        if row!=rows-1:
            for tick in ax_i.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            # label.set_ha("right")
            label.set_ha("center")
            label.set_rotation(0) 


    plt.show()

def plot_space_energy(model, simulator, space_name):
    fig = plt.figure()
    K = 0.65
    fig.set_size_inches(8,10*K) 

    n_plots = 2
    cols = 1 
    rows = math.ceil(n_plots/cols)
    # fig.suptitle("Winter Period, 24-Hour Forecast", fontsize=60)

    x_offset = 0.15
    y_offset = 0.1/K
    ax_width = 0.55
    ax_height = 0.23/K
    axes = []
    for i in range(rows):
        frac_i = i/rows
        for j in range(cols):
            if i!=0:
                y_offset_add = -0.04/K
            else:
                y_offset_add = 0
            frac_j = j/(cols+1)
            if int(i*cols + j) < n_plots:
                # ax_room.append(fig.add_subplot(grid[i, j]))#, xticklabels=[])#, sharey=main_ax)
                # ax_room.append(fig.add_subplot(rows, cols+10, int(i*cols + j + 1)))#, xticklabels=[])#, sharey=main_ax)
                rect = [frac_j + x_offset, frac_i + y_offset + i*y_offset_add, ax_width, ax_height]
                axes.append(fig.add_axes(rect))

    axes.reverse()

    import numpy as np
    model.component_dict[space_name].x_list = np.array(model.component_dict[space_name].x_list)
    axes[0].set_title("Predicted temperature change",fontsize=20)
    axes[0].fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,0], color=global_green, alpha=0.5, label = r"$\Delta T_{W}$")
    axes[0].fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,1], color=global_orange, alpha=0.5, label = r"$\Delta T_{\Phi}$")
    axes[0].fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,2], color=global_red, alpha=0.5, label = r"$\Delta T_{SH}$")
    axes[0].fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,3], color=global_blue, alpha=0.5, label = r"$\Delta T_{V}$")


    energy = np.array(model.component_dict[space_name].x_list[:,2])*14667975/1000/600
    axes[1].fill_between(simulator.dateTimeSteps, 0, energy, color=global_red, alpha=0.5, label = r"$Approx. Energy$")


    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%D")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0) 


    fig.text(0.025, 0.7, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=20)
    # fig.text(0.655, 0.43, r"Position", va='center', ha='center', rotation='vertical', fontsize=40, color = global_orange)
    fig.text(0.45, 0.025, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=20)


def plot_space_wDELTA(model, simulator, space_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np

    params = {
        # 'figure.figsize': (fig_size_x, fig_size_y),
        #  'figure.dpi': 300,
         'axes.labelsize': 15,
         'axes.titlesize': 15,
         'xtick.labelsize': 15,
         'ytick.labelsize': 15,
         "xtick.major.size": 15,
         "xtick.major.width": 2,
         "ytick.major.size": 15,
         "ytick.major.width": 2,
         "lines.linewidth": 2, #4,
         "figure.titlesize": 40,
         "mathtext.fontset": "cm",
         "legend.fontsize": 14,
         "axes.grid": True,
         "grid.color": "black",
         "grid.alpha": 0.2,
         "legend.loc": "upper right",
         "legend.fancybox": False,
         "legend.facecolor": "white",
         "legend.framealpha": 1,
         "legend.edgecolor": "black"
         }

    # print(plt.style.available)
    plt.style.use("ggplot")
    pylab.rcParams.update(params)
    plt.rc('font', family='serif')
    font = {"fontname": "serif"}

    global_colors = sns.color_palette("deep")
    global_blue = global_colors[0]
    global_orange = global_colors[1]
    global_green = global_colors[2]
    global_red = global_colors[3]
    global_purple = global_colors[4]
    global_brown = global_colors[5]
    global_pink = global_colors[6]
    global_grey = global_colors[7]
    global_beis = global_colors[8]
    global_sky_blue = global_colors[9]
    

    

    fig = plt.figure()
    fig.set_size_inches(8,10) 

    n_plots = 3
    cols = 1 
    rows = math.ceil(n_plots/cols)
    # fig.suptitle("Winter Period, 24-Hour Forecast", fontsize=60)

    x_offset = 0.15
    y_offset = 0.1
    ax_width = 0.55
    ax_height = 0.23
    axes = []
    for i in range(rows):
        frac_i = i/rows
        for j in range(cols):
            if i!=0:
                y_offset_add = -0.04
            else:
                y_offset_add = 0
            frac_j = j/(cols+1)
            if int(i*cols + j) < n_plots:
                # ax_room.append(fig.add_subplot(grid[i, j]))#, xticklabels=[])#, sharey=main_ax)
                # ax_room.append(fig.add_subplot(rows, cols+10, int(i*cols + j + 1)))#, xticklabels=[])#, sharey=main_ax)
                rect = [frac_j + x_offset, frac_i + y_offset + i*y_offset_add, ax_width, ax_height]
                axes.append(fig.add_axes(rect))




    axes.reverse()

    # Plotting
    indoor_temperature_setpoint_schedule_name = f"{space_name}| Temperature setpoint schedule"
    weather_station_name = "Outdoor environment"

    axes[0].set_title(space_name,fontsize=20)
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[space_name].savedOutput["indoorTemperature"], color="black",label=r"$T_{predicted}$", linestyle="dashed")
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[indoor_temperature_setpoint_schedule_name].savedOutput["scheduleValue"], color=global_brown,label=r"$T_{setpoint}$", linestyle="dashed")

    ax_twin_0_1 = axes[0].twinx()


    
    ax_twin_0_1.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["valvePosition"], color=global_red, label = r"$u_{v}$")
    ax_twin_0_1.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["supplyDamperPosition"], color=global_blue, label = r"$u_{d}$")
    ax_twin_0_1.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["shadePosition"], color=global_sky_blue, label = r"$u_{s}$")
    # ax_i.legend()
    # ax_i.set_ylim([20, 24]) #Winter
    axes[0].set_ylim([18, 30]) #Summer
    ax_twin_0_1.set_ylim([-0.05, 1.05])


    for tick in axes[0].xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    for tick in axes[1].xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    # formatter = mdates.DateFormatter(r"%m/%d %H")


    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%D")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0) 


    fig.text(0.025, 0.2, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=20)
    fig.text(0.025, 0.8, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=20)
    # fig.text(0.655, 0.43, r"Position", va='center', ha='center', rotation='vertical', fontsize=40, color = global_orange)

    fig.text(0.025, 0.55, r"$\Delta T$ [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=20)
    fig.text(0.82, 0.2, r"Irradiance [W/m$^2$]", va='center', ha='center', rotation='vertical', fontsize=20)
    fig.text(0.45, 0.025, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=20)

    import numpy as np
    model.component_dict[space_name].x_list = np.array(model.component_dict[space_name].x_list)
    axes[1].set_title("Predicted temperature change",fontsize=20)
    axes[1].fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,0], color=global_green, alpha=0.5, label = r"$\Delta T_{W}$")
    axes[1].fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,1], color=global_orange, alpha=0.5, label = r"$\Delta T_{\Phi}$")
    axes[1].fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,2], color=global_red, alpha=0.5, label = r"$\Delta T_{SH}$")
    axes[1].fill_between(simulator.dateTimeSteps, 0, model.component_dict[space_name].x_list[:,3], color=global_blue, alpha=0.5, label = r"$\Delta T_{V}$")
    # axes[1].set_ylim([-0.1, 0.1])


    axes[2].set_title("Weather input",fontsize=20)
    axes[2].plot(simulator.dateTimeSteps, model.component_dict[weather_station_name].savedOutput["outdoorTemperature"], color=global_green, label = r"$T_{amb}$")

    ax_weather_twin = axes[2].twinx()
    ax_weather_twin.plot(simulator.dateTimeSteps, np.array(model.component_dict[weather_station_name].savedOutput["globalIrradiation"])/3.6, color=global_red, label = r"$\Phi$")

    axes[2].set_ylim([-5, 35])
    ax_weather_twin.set_ylim([-50, 1050])

    formatter = mdates.DateFormatter(r"%D")
    axes[2].xaxis.set_major_formatter(formatter)
    for label in axes[2].get_xticklabels():
        label.set_ha("center")
        label.set_rotation(0)

    axes[0].get_shared_x_axes().join(axes[0], axes[1], axes[2])




    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_twin_0_1.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=1, loc = "upper center", bbox_to_anchor=(0.91,0.9))
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    lines_labels1 = axes[1].get_legend_handles_labels()
    lines_labels = [lines_labels1]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=1, loc = "upper center", bbox_to_anchor=(0.91,0.6))
    legend_lines = legend.get_patches()
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        # legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]

    # print(axes[1].get_children())
    # print(legend.get_children())
    # aa


    lines_labels1 = axes[2].get_legend_handles_labels()
    lines_labels2 = ax_weather_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=1, loc = "upper center", bbox_to_anchor=(0.91,0.24))
    legend_lines = legend.get_lines()
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        # graphs[legend].set_visible(not isVisible)
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
            
        legend.set_visible(not isVisible)
        # legend.set_alpha(1.0 if not isVisible else 0.2)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)



def plot_space(model, simulator, space_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np

    load_params()

    global_colors = sns.color_palette("deep")
    global_blue = global_colors[0]
    global_orange = global_colors[1]
    global_green = global_colors[2]
    global_red = global_colors[3]
    global_purple = global_colors[4]
    global_brown = global_colors[5]
    global_pink = global_colors[6]
    global_grey = global_colors[7]
    global_beis = global_colors[8]
    global_sky_blue = global_colors[9]
    

    

    fig = plt.figure()
    K = 0.65
    fig.set_size_inches(8,10*K) 

    n_plots = 2
    cols = 1 
    rows = math.ceil(n_plots/cols)
    # fig.suptitle("Winter Period, 24-Hour Forecast", fontsize=60)

    x_offset = 0.15
    y_offset = 0.1/K
    ax_width = 0.55
    ax_height = 0.23/K
    axes = []
    for i in range(rows):
        frac_i = i/rows
        for j in range(cols):
            if i!=0:
                y_offset_add = -0.04/K
            else:
                y_offset_add = 0
            frac_j = j/(cols+1)
            if int(i*cols + j) < n_plots:
                # ax_room.append(fig.add_subplot(grid[i, j]))#, xticklabels=[])#, sharey=main_ax)
                # ax_room.append(fig.add_subplot(rows, cols+10, int(i*cols + j + 1)))#, xticklabels=[])#, sharey=main_ax)
                rect = [frac_j + x_offset, frac_i + y_offset + i*y_offset_add, ax_width, ax_height]
                axes.append(fig.add_axes(rect))

    axes.reverse()
    

    axes[0].set_title("BuildingSpace",fontsize=20)
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[space_name].savedOutput["indoorTemperature"], color="black",label=r"$T_{z}$", linestyle="dashed")
    # axes[0].plot(simulator.dateTimeSteps, model.component_dict[indoor_temperature_setpoint_schedule_name].savedOutput["scheduleValue"], color=global_brown,label=r"$T_{setpoint}$", linestyle="dashed")

    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["valvePosition"], color=global_red, label = r"$u_{valve}$")
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["damperPosition"], color=global_blue, label = r"$u_{damper}$")
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["shadePosition"], color=global_sky_blue, label = r"$u_{shade}$")
    
    axes[1].plot(simulator.dateTimeSteps, model.component_dict[space_name].savedOutput["indoorCo2Concentration"], color="black", label = r"$C_{z}$", linestyle="dashed")
    ax_1_twin_0 = axes[1].twinx()
    ax_1_twin_1 = axes[1].twinx()
    ax_1_twin_0.plot(simulator.dateTimeSteps, np.array(model.component_dict[space_name].savedInput["numberOfPeople"]), color=global_blue, label = r"$N_{occ}$")
    ax_1_twin_1.plot(simulator.dateTimeSteps, np.array(model.component_dict[space_name].savedInput["supplyAirFlowRate"]), color=global_orange, label = r"$\dot{m}_{a}$")
    ax_1_twin_1.spines['right'].set_position(('outward', 50))
    ax_1_twin_1.spines["right"].set_visible(True)


    axes[0].set_ylim([18, 30]) #Summer
    ax_0_twin.set_ylim([-0.05, 1.05])


    for tick in axes[0].xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

    # formatter = mdates.DateFormatter(r"%m/%d %H")


    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0) 


    fig.text(0.025, 0.35, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(0.025, 0.8, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(0.82, 0.35, r"Irradiance [W/m$^2$]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(0.45, 0.025, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    formatter = mdates.DateFormatter(r"%H")
    axes[1].xaxis.set_major_formatter(formatter)
    for label in axes[1].get_xticklabels():
        label.set_ha("center")
        label.set_rotation(0)




    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=1, loc = "upper center", bbox_to_anchor=(0.91,0.9))
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]

    # print(axes[1].get_children())
    # print(legend.get_children())
    # aa


    lines_labels1 = axes[1].get_legend_handles_labels()
    lines_labels2 = ax_1_twin_0.get_legend_handles_labels()
    lines_labels3 = ax_1_twin_1.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2, lines_labels3]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=1, loc = "upper center", bbox_to_anchor=(0.91,0.4))
    legend_lines = legend.get_lines()
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]

    

    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        # graphs[legend].set_visible(not isVisible)
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
            
        legend.set_visible(not isVisible)
        # legend.set_alpha(1.0 if not isVisible else 0.2)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)

    fig.savefig(f"plot_{space_name}.png", dpi=300)

def plot_space_temperature_old(model, simulator, space_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(space_name)

    
    ax_0_twin = axes[0].twinx()
    

    
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[space_name].savedOutput["indoorTemperature"], color="black",label=r"$T_{z}$", linestyle="dashed")
    # axes[0].plot(simulator.dateTimeSteps, model.component_dict[indoor_temperature_setpoint_schedule_name].savedOutput["scheduleValue"], color=global_brown,label=r"$T_{setpoint}$", linestyle="dashed")

    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["valvePosition"], color=global_red, label = r"$u_{valve}$")
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["damperPosition"], color=global_blue, label = r"$u_{damper}$")
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["shadePosition"], color=global_sky_blue, label = r"$u_{shade}$")


    # axes[0].set_ylim([18, 30]) #Summer
    


    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0) 


    # fig.text(*global_left_y, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Position", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Temperature [$^\circ$C]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Position", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)


    ax_0_twin.set_ylim([0, 1])
    axes_list = axes + [ax_0_twin]
    nticks_list = [6,6]
    round_to_list = [0.1,0.1]
    y_offset_list = [None,0.05]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(space_name)}_temperature.png", dpi=300)


def plot_space_temperature(model, simulator, space_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(space_name)

    outdoor_environment_name = "Outdoor environment"    

    
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[space_name].savedOutput["indoorTemperature"], color="black",label=r"$T_{z}$", linestyle="dashed")
    # axes[0].plot(simulator.dateTimeSteps, model.component_dict[outdoor_environment_name].savedOutput["outdoorTemperature"], color=global_green, label = r"$T_{amb}$")
    # axes[0].plot(simulator.dateTimeSteps, model.component_dict[indoor_temperature_setpoint_schedule_name].savedOutput["scheduleValue"], color=global_brown,label=r"$T_{setpoint}$", linestyle="dashed")

    ax_0_twin = axes[0].twinx()
    # ax_0_twin_1 = axes[0].twinx()
    
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["valvePosition"], color=global_red, label = r"$u_{valve}$")
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["supplyDamperPosition"], color=global_blue, label = r"$u_{damper}$")
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_name].savedInput["shadePosition"], color=global_sky_blue, label = r"$u_{shade}$")
    # ax_0_twin_1.plot(simulator.dateTimeSteps, np.array(model.component_dict[outdoor_environment_name].savedOutput["globalIrradiation"])/3.6, color=global_orange, label = r"$\Phi$")

    
    # ax_0_twin_1.spines['right'].set_position(('outward', global_outward))
    # ax_0_twin_1.spines["right"].set_visible(True)
    # ax_0_twin_1.spines["right"].set_color("black")
    


    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0) 


    # fig.text(*global_left_y, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Position", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    # axes[0].set_ylabel(r"Temperature [$^\circ$C]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Position", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    # ax_0_twin_1.set_ylabel(r"Solar irradiance [W/m$^2$]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    # lines_labels3 = ax_0_twin_1.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)


    ax_0_twin.set_ylim([0, 1])
    # ax_0_twin_1.set_ylim([0, 300])
    axes_list = axes + [ax_0_twin]
    nticks_list = [6,6]
    round_to_list = [0.1,0.1]
    y_offset_list = [None,0.05]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(space_name)}_temperature.png", dpi=300)

def plot_space_CO2(model, simulator, space_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(space_name)

    
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[space_name].savedOutput["indoorCo2Concentration"], color="black", label = r"$C_{z}$", linestyle="dashed")
    ax_0_twin_0 = axes[0].twinx()
    ax_0_twin_1 = axes[0].twinx()
    ax_0_twin_0.plot(simulator.dateTimeSteps, np.array(model.component_dict[space_name].savedInput["numberOfPeople"]), color=global_orange, label = r"$N_{occ}$")
    ax_0_twin_1.plot(simulator.dateTimeSteps, np.array(model.component_dict[space_name].savedInput["supplyAirFlowRate"]), color=global_blue, label = r"$\dot{m}_{a}$")
    ax_0_twin_1.spines['right'].set_position(('outward', global_outward))
    ax_0_twin_1.spines["right"].set_visible(True)
    ax_0_twin_1.spines["right"].set_color("black")
    # ax_0_twin_1.tick_params(axis='y', colors=global_blue) 

    # axes[0].spines[:].set_visible(True)
    # axes[0].spines[:].set_color("black")
    # axes[0].spines[:].set_edgecolor("black")
    # axes[0].spines[:].set_facecolor("black")
    # axes[0].spines[:].set_fill(True)
    # axes[0].spines[:].set_linewidth(3)
    # axes[0].spines[:].set_linestyle("-")
    # axes[0].spines[:].set_hatch("O")



    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0) 


    # fig.text(*global_left_y, r"CO2-level [ppm]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(global_right_y_first[0]-0.02, global_right_y_first[1], r"Occupancy", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_second, r"Airflow [kg/s]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"CO2-level [ppm]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin_0.set_ylabel(r"Occupancy", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin_1.set_ylabel(r"Airflow [kg/s]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin_0.get_legend_handles_labels()
    lines_labels3 = ax_0_twin_1.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2, lines_labels3]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)

    axes[0].set_ylim([400, 900])
    ax_0_twin_0.set_ylim([0, 45])
    ax_0_twin_1.set_ylim([0, 1])
    axes_list = axes + [ax_0_twin_0,ax_0_twin_1]
    nticks_list = [6,6,6]
    round_to_list = [100,3,0.1]
    y_offset_list = [None,3,None]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(space_name)}_co2.png", dpi=300)


def plot_weather_station(model, simulator):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np

    load_params()
    fig, axes = get_fig_axes("Outdoor environment")
    outdoor_environment_name = "Outdoor environment"

    axes[0].plot(simulator.dateTimeSteps, model.component_dict[outdoor_environment_name].savedOutput["outdoorTemperature"], color=global_green, label = r"$T_{amb}$")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, np.array(model.component_dict[outdoor_environment_name].savedOutput["globalIrradiation"])/3.6, color=global_orange, label = r"$\Phi$")

    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%D")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0) 


    # fig.text(*global_left_y, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Irradiance [W/m$^2$]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Temperature [$^\circ$C]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Irradiance [W/m$^2$]", fontsize=pylab.rcParams['axes.labelsize'], color="black")



    

    


    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)
    

    axes[0].set_ylim([0, 8])
    ax_0_twin.set_ylim([0, 300])
    axes_list = axes + [ax_0_twin]
    nticks_list = [6,6]
    round_to_list = [0.1,10]
    y_offset_list = [None,10]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(outdoor_environment_name)}.png", dpi=300)



def plot_space_heater(model, simulator, space_heater_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(space_heater_name)

    axes[0].plot(simulator.dateTimeSteps, np.array(model.component_dict[space_heater_name].savedOutput["Power"])/1000, color="black",label=r"$\dot{Q}_h$", linestyle="dashed")
    ax_0_twin_0 = axes[0].twinx()
    ax_0_twin_1 = axes[0].twinx()
    ax_0_twin_0.plot(simulator.dateTimeSteps, model.component_dict[space_heater_name].savedInput["waterFlowRate"], color=global_blue, label = r"$\dot{m}_w$")
    # ax_0_twin_1.plot(simulator.dateTimeSteps, model.component_dict[space_heater_name].savedInput["supplyWaterTemperature"], color=global_red,label=r"$T_{w,in}$", linestyle="solid")

    ax_0_twin_1.spines['right'].set_position(('outward', global_outward))
    ax_0_twin_1.spines["right"].set_visible(True)
    ax_0_twin_1.spines["right"].set_color("black")


    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)

    # fig.text(*global_left_y, r"Power [kW]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Waterflow [kg/s]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Power [kW]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin_0.set_ylabel(r"Waterflow [kg/s]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin_0.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)


    axes[0].set_ylim([0, 4])
    ax_0_twin_0.set_ylim([0, 0.25])
    axes_list = axes + [ax_0_twin_0]
    nticks_list = [6,6]
    round_to_list = [0.1,0.02]
    y_offset_list = [None,0.01]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(space_heater_name)}.png", dpi=300)


def plot_space_heater_energy(model, simulator, space_heater_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes("Space Heater")

    axes[0].plot(simulator.dateTimeSteps, np.array(model.component_dict[space_heater_name].savedOutput["Energy"]), color="black",label=r"$E_h$", linestyle="dashed")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[space_heater_name].savedInput["waterFlowRate"], color=global_blue, label = r"$\dot{m}_w$")


    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)

    # fig.text(*global_left_y, r"Power [kW]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Waterflow [kg/s]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Energy [kWh]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Waterflow [kg/s]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    # axes[0].set_ylim([0, 170])

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)

    fig.savefig(f"{get_file_name(space_heater_name)}_energy.png", dpi=300)


    
def plot_temperature_controller(model, simulator, temperature_controller_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(temperature_controller_name)
    
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[temperature_controller_name].savedOutput["inputSignal"], color="black",label=r"$u_v$", linestyle="dashed")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[temperature_controller_name].savedInput["actualValue"], color=global_blue, label = r"$T_z$")
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[temperature_controller_name].savedInput["setpointValue"], color=global_red, label = r"$T_{z,set}$")
    axes[0].set_ylim([-0.05, 1.05])


    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)


    # fig.text(*global_left_y, r"Position", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Position", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Temperature [$^\circ$C]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    


    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)


    axes[0].set_ylim([0, 1])
    axes_list = axes + [ax_0_twin]
    nticks_list = [6,6]
    round_to_list = [0.1,0.1]
    y_offset_list = [0.05,None]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(temperature_controller_name)}.png", dpi=300)


def plot_CO2_controller(model, simulator, CO2_controller_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(CO2_controller_name)

    axes[0].plot(simulator.dateTimeSteps, model.component_dict[CO2_controller_name].savedOutput["inputSignal"], color="black",label=r"$u_{d}$", linestyle="dashed")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[CO2_controller_name].savedInput["actualValue"], color=global_blue, label = r"$C_z$")
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[CO2_controller_name].savedInput["setpointValue"], color=global_red, label = r"$C_{z,set}$")
    axes[0].set_ylim([-0.05, 1.05])

    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)


    # fig.text(*global_left_y, r"Position", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"CO$_2$ [ppm]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Position", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"CO$_2$ [ppm]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)


    axes[0].set_ylim([0, 1])
    ax_0_twin.set_ylim([400, 900])
    axes_list = axes + [ax_0_twin]
    nticks_list = [6,6,6]
    round_to_list = [0.1,100]
    y_offset_list = [0.05,None]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(CO2_controller_name)}.png", dpi=300)

def plot_CO2_controller_rulebased(model, simulator, CO2_controller_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(CO2_controller_name)

    axes[0].plot(simulator.dateTimeSteps, model.component_dict[CO2_controller_name].savedOutput["inputSignal"], color="black",label=r"$u_{d}$", linestyle="dashed")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[CO2_controller_name].savedInput["actualValue"], color=global_blue, label = r"$C_z$")
    axes[0].set_ylim([-0.05, 1.05])

    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)


    # fig.text(*global_left_y, r"Position", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"CO$_2$ [ppm]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Position", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"CO$_2$ [ppm]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)


    axes[0].set_ylim([0, 1])
    ax_0_twin.set_ylim([400, 900])
    axes_list = axes + [ax_0_twin]
    nticks_list = [6,6,6]
    round_to_list = [0.1,100]
    y_offset_list = [0.05,None]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(CO2_controller_name)}.png", dpi=300)

def plot_heat_recovery_unit(model, simulator, air_to_air_heat_recovery_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(air_to_air_heat_recovery_name)

    axes[0].plot(simulator.dateTimeSteps, model.component_dict[air_to_air_heat_recovery_name].savedOutput["primaryTemperatureOut"], color="black",label=r"$T_{a,sup,out}$", linestyle="dashed")
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[air_to_air_heat_recovery_name].savedInput["primaryTemperatureIn"], color=global_green, label = r"$T_{a,sup,in}$")
    axes[0].plot(simulator.dateTimeSteps, model.component_dict[air_to_air_heat_recovery_name].savedInput["secondaryTemperatureIn"], color=global_red, label = r"$T_{a,exh,in}$")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[air_to_air_heat_recovery_name].savedInput["primaryAirFlowRate"], color=global_blue, label = r"$\dot{m}_{a}$")
    
    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)


    # fig.text(*global_left_y, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Airflow [kg/s]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Temperature [$^\circ$C]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Airflow [kg/s]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)


    axes[0].set_ylim([0, None])
    ax_0_twin.set_ylim([0, 1])
    axes_list = axes + [ax_0_twin]
    nticks_list = [6,6]
    round_to_list = [0.1,0.01]
    y_offset_list = [None,0.05]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)
    fig.savefig(f"{get_file_name(air_to_air_heat_recovery_name)}.png", dpi=300)


def plot_heating_coil(model, simulator, heating_coil_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(heating_coil_name)

    axes[0].plot(simulator.dateTimeSteps, np.array(model.component_dict[heating_coil_name].savedOutput["Power"])/1000, color="black", label = r"$\dot{Q}_{hc}$", linestyle="dashed")
    ax_0_twin_0 = axes[0].twinx()
    ax_0_twin_1 = axes[0].twinx()
    ax_0_twin_0.plot(simulator.dateTimeSteps, model.component_dict[heating_coil_name].savedInput["airTemperatureIn"], color=global_green,label=r"$T_{a,in}$", linestyle="solid")
    ax_0_twin_0.plot(simulator.dateTimeSteps, model.component_dict[heating_coil_name].savedInput["airTemperatureOutSetpoint"], color=global_red,label=r"$T_{a,set}$", linestyle="solid")
    ax_0_twin_1.plot(simulator.dateTimeSteps, model.component_dict[heating_coil_name].savedInput["airFlowRate"], color=global_blue, label = r"$\dot{m}_{a}$")

    ax_0_twin_1.spines['right'].set_position(('outward', global_outward))
    ax_0_twin_1.spines["right"].set_visible(True)
    ax_0_twin_1.spines["right"].set_color("black")

    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)

    # fig.text(*global_left_y, r"Power [kW]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(global_right_y_first[0]-0.02, global_right_y_first[1], r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(global_right_y_second[0]-0.02, global_right_y_second[1], r"Airflow [kg/s]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Power [kW]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin_0.set_ylabel(r"Temperature [$^\circ$C]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin_1.set_ylabel(r"Airflow [kg/s]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    
    

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin_0.get_legend_handles_labels()
    lines_labels3 = ax_0_twin_1.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2, lines_labels3]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)

    axes[0].set_ylim([0, None])
    # ax_0_twin_0.set_ylim([0, 0.22])
    ax_0_twin_1.set_ylim([0, 1])
    axes_list = axes + [ax_0_twin_0,ax_0_twin_1]
    nticks_list = [6,6,6]
    round_to_list = [0.1,0.1,0.02]
    y_offset_list = [None,None,0.05]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)

    fig.savefig(f"{get_file_name(heating_coil_name)}.png", dpi=300)



def plot_fan(model, simulator, fan_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(fan_name)

    axes[0].plot(simulator.dateTimeSteps, np.array(model.component_dict[fan_name].savedOutput["Power"])/1000, color="black", label = r"$\dot{W}_{fan}$", linestyle="dashed")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[fan_name].savedInput["airFlowRate"], color=global_blue, label = r"$\dot{m}_{a}$")

    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)


    # fig.text(*global_left_y, r"Power [kW]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Power [kW]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Airflow [kg/s]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)

    axes[0].set_ylim([0, 4])
    ax_0_twin.set_ylim([0, 6])
    axes_list = axes + [ax_0_twin]
    nticks_list = [6,6]
    round_to_list = [0.1,0.3]
    y_offset_list = [None,0.25]
    alignYaxes(axes_list, nticks_list, round_to_list, y_offset_list)

    fig.savefig(f"{get_file_name(fan_name)}.png", dpi=300)


def plot_supply_fan_energy(model, simulator, supply_fan_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(supply_fan_name)

    axes[0].plot(simulator.dateTimeSteps, np.array(model.component_dict[supply_fan_name].savedOutput["Energy"]), color="black", label = r"${E}_{fan}$", linestyle="dashed")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[supply_fan_name].savedInput["airFlowRate"], color=global_blue, label = r"$\dot{m}_{a}$")

    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)


    # fig.text(*global_left_y, r"Power [kW]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Temperature [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Energy [kWh]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Massflow [kg/s]", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)

    fig.savefig(f"{get_file_name(supply_fan_name)}_energy.png", dpi=300)


def plot_supply_damper(model, simulator, supply_damper_name):
    import matplotlib.dates as mdates
    import matplotlib.pylab as pylab
    import seaborn as sns
    import numpy as np
    load_params()
    fig, axes = get_fig_axes(supply_damper_name)

    axes[0].plot(simulator.dateTimeSteps, np.array(model.component_dict[supply_damper_name].savedOutput["airFlowRate"]), color="black", label = r"$\dot{m}_{a}$", linestyle="dashed")
    ax_0_twin = axes[0].twinx()
    ax_0_twin.plot(simulator.dateTimeSteps, model.component_dict[supply_damper_name].savedInput["damperPosition"], color=global_blue, label = r"$u_d$")

    for ax_i in axes:
        formatter = mdates.DateFormatter(r"%H")
        ax_i.xaxis.set_major_formatter(formatter)
        for label in ax_i.get_xticklabels():
            label.set_ha("center")
            label.set_rotation(0)


    # fig.text(*global_left_y, r"Massflow [kg/s]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    # fig.text(*global_right_y_first, r"Position [$^\circ$C]", va='center', ha='center', rotation='vertical', fontsize=pylab.rcParams['axes.labelsize'])
    fig.text(*global_x, r"Hour of day", va='center', ha='center', rotation='horizontal', fontsize=pylab.rcParams['axes.labelsize'])

    axes[0].set_ylabel(r"Massflow [kg/s]", fontsize=pylab.rcParams['axes.labelsize'], color="black")
    ax_0_twin.set_ylabel(r"Position", fontsize=pylab.rcParams['axes.labelsize'], color="black")

    lines_labels1 = axes[0].get_legend_handles_labels()
    lines_labels2 = ax_0_twin.get_legend_handles_labels()
    lines_labels = [lines_labels1, lines_labels2]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    legend = fig.legend(lines, labels, ncol=len(labels), loc = "upper center", bbox_to_anchor=global_legend_loc)
    legend_lines = legend.get_lines()
    graphs = {}
    for i in range(len(legend_lines)):
        legend_lines[i].set_picker(True)
        legend_lines[i].set_pickradius(10)
        graphs[legend_lines[i]] = [lines[i]]


    def on_pick(event):
        legend = event.artist
        isVisible = legend.get_visible()
        for line in graphs[legend]:
            isVisible = line.get_visible()
            line.set_visible(not isVisible)
        legend.set_visible(not isVisible)
        fig.canvas.draw()
    plt.connect('pick_event', on_pick)

    fig.savefig(f"{get_file_name(supply_damper_name)}.png", dpi=300)


# This code has been adapted from the ptemcee package https://github.com/willvousden/ptemcee
def plot_intervals(intervals, time, ydata=None, xdata=None,
                   limits=[95],
                   adddata=None, addmodel=True, addlegend=True,
                   addcredible=True, addprediction=True,
                   data_display={}, model_display={}, interval_display={},
                   fig=None, ax=None, figsize=None, legloc='upper left',
                   ciset=None, piset=None,
                   return_settings=False):
    '''
    Plot propagation intervals in 2-D

    This routine takes the model distributions generated using the
    :func:`~calculate_intervals` method and then plots specific
    quantiles.  The user can plot just the intervals, or also include the
    median model response and/or observations.  Specific settings for
    credible intervals are controlled by defining the `ciset` dictionary.
    Likewise, for prediction intervals, settings are defined using `piset`.

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
          `piset` dictionaries.
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
        * **piset** (:py:class:`dict`): Settings for prediction intervals
        * **return_settings** (:py:class:`bool`): Flag to return ciset and
          piset along with fig and ax.

    Returns:
        * (:py:class:`tuple`) with elements
            1) Figure handle
            2) Axes handle
            3) Dictionary with `ciset` and `piset` inside (only
               outputted if `return_settings=True`)
    '''

    if (fig is None and ax is not None) or (fig is not None and ax is None):
        raise("Please supply both fig and ax as arguments")


    if fig is None and ax is None:
        fig, ax = plt.subplots()
    # unpack dictionary
    credible = intervals['credible']
    prediction = intervals['prediction']
    # Check user-defined settings
    ciset = __setup_iset(ciset,
                         default_iset=dict(
                                 limits=limits,
                                 cmap=None,
                                 colors=None))
    piset = __setup_iset(piset,
                         default_iset=dict(
                                 limits=limits,
                                 cmap=None,
                                 colors=None))
    # Check limits
    ciset['limits'] = _check_limits(ciset['limits'], limits)
    piset['limits'] = _check_limits(piset['limits'], limits)
    # convert limits to ranges
    ciset['quantiles'] = _convert_limits(ciset['limits'])
    piset['quantiles'] = _convert_limits(piset['limits'])
    # setup display settings
    interval_display, model_display, data_display = setup_display_settings(
            interval_display, model_display, data_display)
    # Define colors
    ciset['colors'] = setup_interval_colors(ciset, inttype='ci')
    piset['colors'] = setup_interval_colors(piset, inttype='pi')
    # Define labels
    ciset['labels'] = _setup_labels(ciset['limits'], inttype='CI')
    piset['labels'] = _setup_labels(piset['limits'], inttype='PI')

    time = time.reshape(time.size,)
    # add prediction intervals
    if addprediction is True:
        for ii, quantile in enumerate(piset['quantiles']):
            pi = generate_quantiles(prediction, np.array(quantile))
            ax.fill_between(time, pi[0], pi[1], facecolor=piset['colors'][ii],
                            label=piset['labels'][ii], **interval_display)
    # add credible intervals
    if addcredible is True:
        for ii, quantile in enumerate(ciset['quantiles']):
            ci = generate_quantiles(credible, np.array(quantile))
            ax.fill_between(time, ci[0], ci[1], facecolor=ciset['colors'][ii],
                            label=ciset['labels'][ii], **interval_display)
    # add model (median model response)
    if addmodel is True:
        ci = generate_mode(credible, n_bins=50)
        ax.plot(time, ci, **model_display)
    # add data to plot
    if ydata is not None and adddata is None:
        adddata = True
    if adddata is True and ydata is not None:
        if xdata is None:
            ax.plot(time, ydata, **data_display)
        else:
            ax.plot(xdata, ydata, **data_display)
    # add legend
    if addlegend is True:
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc=legloc)
    if return_settings is True:
        return fig, ax, dict(ciset=ciset, piset=piset)
    else:
        return fig, ax

def check_s2chain(s2chain, nsimu):
    '''
    Check size of s2chain

    Args:
        * **s2chain** (:py:class:`float`, :class:`~numpy.ndarray`, or `None`):
            Observation error variance chain or value
        * **nsimu** (:py:class:`int`): No. of elements in chain

    Returns:
        * **s2chain** (:class:`~numpy.ndarray` or `None`)
    '''
    if s2chain is None:
        return None
    else:
        if isinstance(s2chain, float):
            s2chain = np.ones((nsimu,))*s2chain
        if s2chain.ndim == 2:
            if s2chain.shape[0] != nsimu:
                s2chain = s2chain * np.ones((nsimu, s2chain.size))
        else:
            if s2chain.size != nsimu:  # scalars provided for multiple QoI
                s2chain = s2chain * np.ones((nsimu, s2chain.size))
        return s2chain


# --------------------------------------------
def observation_sample(s2, y, sstype):
    '''
    Calculate model response with observation errors.

    Args:
        * **s2** (:class:`~numpy.ndarray`): Observation error(s).
        * **y** (:class:`~numpy.ndarray`): Model responses.
        * **sstype** (:py:class:`int`): Flag to specify sstype.

    Returns:
        * **opred** (:class:`~numpy.ndarray`): Model responses with observation errors.
    '''
    if sstype == 0:
        opred = y + np.random.standard_normal(y.shape) * np.sqrt(s2)
    elif sstype == 1:  # sqrt
        opred = (np.sqrt(y) + np.random.standard_normal(y.shape) * np.sqrt(s2))**2
    elif sstype == 2:  # log
        opred = y*np.exp(np.random.standard_normal(y.shape) * np.sqrt(s2))
    else:
        sys.exit('Unknown sstype')
    return opred


# --------------------------------------------
def define_sample_points(nsample, nsimu):
    '''
    Define indices to sample from posteriors.

    Args:
        * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
        * **nsimu** (:py:class:`int`): Number of MCMC simulations.

    Returns:
        * **iisample** (:class:`~numpy.ndarray`): Array of indices in posterior set.
        * **nsample** (:py:class:`int`): Number of samples to draw from posterior.
    '''
    # define sample points
    if nsample >= nsimu:
        iisample = range(nsimu)  # sample all points from chain
        nsample = nsimu
    else:
        # randomly sample from chain
        iisample = np.ceil(np.random.rand(nsample)*nsimu) - 1
        iisample = iisample.astype(int)
    return iisample, nsample


# --------------------------------------------
def generate_quantiles(x, p=np.array([0.25, 0.5, 0.75])):
    '''
    Calculate empirical quantiles.

    Args:
        * **x** (:class:`~numpy.ndarray`): Observations from which to generate quantile.
        * **p** (:class:`~numpy.ndarray`): Quantile limits.

    Returns:
        * (:class:`~numpy.ndarray`): Interpolated quantiles.
    '''
    
    # extract number of rows/cols from np.array
    n = x.shape[0]
    # define vector valued interpolation function
    xpoints = np.arange(0, n, 1)
    interpfun = interp1d(xpoints, np.sort(x, 0), axis=0)
    # evaluation points
    itpoints = (n - 1)*p
    return interpfun(itpoints)

def generate_mode(x, n_bins=50):
    '''
    Calculate empirical mode.

    Args:
        * **x** (:class:`~numpy.ndarray`): Observations from which to generate mode.
        * **p** (:class:`~numpy.ndarray`): Number of bins.

    Returns:
        * (:class:`~numpy.ndarray`): Mode from histogram.
    '''
    print(x.shape)
    n_timesteps = x.shape[1]
    hist = [np.histogram(x[:,i], bins=n_bins) for i in range(n_timesteps)]
    frequency = np.array([el[0] for el in hist])
    edges = np.array([el[1] for el in hist])
    mode_indices = np.argmax(frequency,axis=1)
    print(mode_indices.shape)
    modes = edges[np.arange(n_timesteps), mode_indices]
    print(edges.shape)
    print(modes.shape)
    return modes


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


# --------------------------------------------
def _setup_labels(limits, inttype='CI'):
    '''
    Setup labels for prediction/credible intervals.
    '''
    labels = []
    for limit in limits:
        labels.append(str('{}% {}'.format(limit, inttype)))
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
