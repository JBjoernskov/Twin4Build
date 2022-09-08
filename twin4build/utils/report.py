import matplotlib.pyplot as plt
import math
import matplotlib.pylab as pylab
import matplotlib.dates as mdates
# from matplotlib.pyplot import cm
from itertools import cycle
import numpy as np
import seaborn as sns

params = {
        # 'figure.figsize': (fig_size_x, fig_size_y),
        #  'figure.dpi': 300,
         'axes.labelsize': 25,
         'axes.titlesize': 25,
         'xtick.labelsize': 20,
         'ytick.labelsize': 20,
         "xtick.major.size": 15,
         "xtick.major.width": 2,
         "ytick.major.size": 15,
         "ytick.major.width": 2,
         "lines.linewidth": 4, #4,
         "figure.titlesize": 40,
         "mathtext.fontset": "cm",
         "legend.fontsize": 20,
        #  "figure.autolayout": True, #################
         "axes.grid": True,
         "grid.color": "black",
         "grid.alpha": 0.2,
         "legend.loc": "upper right",
         "legend.fancybox": False,
        #  "legend.shadow": True,
         "legend.facecolor": "white",
         "legend.framealpha": 1,
         "legend.edgecolor": "black"
        #  'hatch.linewidth': 
        #  'text.usetex': True
         }
plt.style.use("ggplot")
pylab.rcParams.update(params)
plt.rc('font', family='serif')
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


class Report:
    def __init__(self,
                savedInput = None,
                savedOutput = None,
                createReport = False,
                **kwargs):
        self.savedInput = savedInput ###
        self.savedOutput = savedOutput ###
        self.createReport = createReport ###
        # super().__init__(**kwargs)

    def update_report(self):
        if self.createReport:
            for key in self.input:
                if key not in self.savedInput:
                    self.savedInput[key] = [self.input[key]]
                else:
                    self.savedInput[key].append(self.input[key])
                

            for key in self.output:
                if key not in self.savedOutput:
                    self.savedOutput[key] = [self.output[key]]
                else:
                    self.savedOutput[key].append(self.output[key])


    def plot_report(self, time_list):
        n_plots = 2
        cols = 2
        rows = math.ceil(n_plots/cols)
        
        fig = plt.figure()
        fig.suptitle(self.systemId, fontsize=60)
        # figManager = plt.get_current_fig_manager() ################
        # figManager.window.showMaximized() #######################
        fig.set_size_inches(40, 13) 

        grid = plt.GridSpec(rows, cols, hspace=0.1, wspace=0.1) #0.2
        # ax_loss = fig.add_subplot(grid[0, 0:2])

        x_offset = 0.05
        y_offset = 0.1
        ax_width = 0.3
        ax_height = 0.6
        ax = []
        ax_twin_AirFlowRate = []
        ax_twin_Power = []
        ax_twin_Signal = []
        ax_twin_Radiation = []
        for i in range(rows):
            frac_i = i/rows
            for j in range(cols):
                if j!=0:
                    x_offset_add = 0.05
                else:
                    x_offset_add = 0
                frac_j = j/(cols)
                if int(i*cols + j) < n_plots:
                    rect = [frac_j + x_offset + j*x_offset_add, frac_i + y_offset, ax_width, ax_height]
                    added_ax = fig.add_axes(rect)
                    ax.append(added_ax)
                    # ax_twin_AirFlowRate.append(added_ax.twinx())
                    # ax_twin_Power.append(added_ax.twinx())
                    # ax_twin_Signal.append(added_ax.twinx())
                    # ax_twin_Radiation.append(added_ax.twinx())

        axis_priority_list = ["indoorTemperature", "radiatorOutletTemperature", "Temperature", "Power", "People", "Position", "flowRate", "FlowRate", "Radiation", "waterFlowRate", "Co2Concentration", "Energy", "Value", "Signal"]
        color_list = ["black", *global_colors, *global_colors]
        linecycler_list = [cycle(["-","--","-.",":"]) for i in range(len(axis_priority_list))]
        normalize_list = [1, 1, 1, 1/1000, 1, 1, 3600/1.225, 3600/1.225, 1/3.6, 1, 1, 1, 1, 1]
        unit_list = ["[$^\circ$C]", "[$^\circ$C]", "[$^\circ$C]", "[kW]", "", "", "[m$^3$/h]", "[m$^3$/h]", "[W/m$^2$]", "[kg/s]", "[ppm]", "[kWh]", "", ""]
        y_lim_min_list = [15, None, -5, -0.5, -0.5, -0.05, -50, -0.01, -10, 0, 0, 0, 0, 0]
        y_lim_max_list = [30, None, 35, 10, 20, 1.05, 3500, None, 1000, 30, 1000, 0, 0, 0]
        data_list = [self.savedInput, self.savedOutput]

        #The amount of secondary axes is limited to 3 to keep the plot readable
        #The axes are prioritized following "axis_priority_list"
        secondary_axis_limit = 99

        lines_list = []
        legend_list = []
        for i,data in enumerate(data_list):

            if i!=0:
                x_offset_add = 0.05
            else:
                x_offset_add = 0

            ax_list = [None]*len(axis_priority_list)
            
            if len(list(data.keys())) != 0:
                stripped_input_list = []
                data_keys = list(data.keys())
                for ii in range(len(data_keys)):
                    for axis_name in axis_priority_list:
                        if data_keys[ii].find(axis_name)!=-1:
                            stripped_input_list.append(axis_name)
                            data_keys[ii] = ""

                # stripped_input_list = [jj for ii in list(data.keys()) for jj in axis_priority_list if ii.find(jj)!=-1]
                first_axis_priority_index_list = [axis_priority_list.index(ii) for ii in stripped_input_list]
                smallest_n_values_list = sorted(list(set(first_axis_priority_index_list)))[:secondary_axis_limit+1] #+1 for first axis'
                min_index = min(first_axis_priority_index_list)
                first_axis_index_list = [i for i, x in enumerate(first_axis_priority_index_list) if x == min_index]
                offset_change = 100
                offset = 0
                for j,key in enumerate(data):
                    label_string = axis_priority_list[first_axis_priority_index_list[j]] + " " + unit_list[first_axis_priority_index_list[j]]
                    frac_i = i/(cols)
                    # color = color_list[first_axis_priority_index_list[j]]
                    value = np.array(data[key])*normalize_list[first_axis_priority_index_list[j]]
                    if first_axis_priority_index_list[j] in smallest_n_values_list:
                        if j in first_axis_index_list:
                            color = "black"
                            ax[i].plot(time_list, value, label=key, color=color, linestyle=next(linecycler_list[first_axis_priority_index_list[j]]))
                            ax[i].set_ylabel(label_string, color = color)
                            ax[i].set_ylim([y_lim_min_list[first_axis_priority_index_list[j]], y_lim_max_list[first_axis_priority_index_list[j]]])
                            # offset_y_label = frac_i + x_offset/6 + x_offset_add
                            

                        else:
                            color = color_list[first_axis_priority_index_list[j]]
                            if ax_list[first_axis_priority_index_list[j]] is None:
                                ax_list[first_axis_priority_index_list[j]] = ax[i].twinx()
                                
                                ax_list[first_axis_priority_index_list[j]].spines['right'].set_position(('outward', offset))
                                ax_list[first_axis_priority_index_list[j]].spines["right"].set_visible(True)
                                ax_list[first_axis_priority_index_list[j]].spines["right"].set_color(color)
                                ax_list[first_axis_priority_index_list[j]].tick_params(axis='y', colors=color)

                                ax_list[first_axis_priority_index_list[j]].yaxis.labelpad = 0
                                ax_list[first_axis_priority_index_list[j]].set_ylabel(label_string, color = color)
                                ax_list[first_axis_priority_index_list[j]].set_ylim([y_lim_min_list[first_axis_priority_index_list[j]], y_lim_max_list[first_axis_priority_index_list[j]]])
                                
                                offset += offset_change
                            # offset_y_label = frac_i + ax_width + x_offset + offset/2300 + x_offset_add
                            ax_list[first_axis_priority_index_list[j]].plot(time_list, value, label=key, color=color, linestyle=next(linecycler_list[first_axis_priority_index_list[j]]))
                    # label_string = first_axis_priority_list[first_axis_priority_index_list[j]] + " " + unit_list[first_axis_priority_index_list[j]]
                    # fig.text(offset_y_label, 0.4, label_string, va='center', ha='center', rotation='vertical', fontsize=25, color = color)

                lines_labels_list = []
                lines_labels = ax[i].get_legend_handles_labels()
                lines_labels_list.append(lines_labels)
                for ax_i in ax_list:
                    if ax_i is not None:
                        lines_labels = ax_i.get_legend_handles_labels()
                        lines_labels_list.append(lines_labels)
                lines, labels = [sum(lol, []) for lol in zip(*lines_labels_list)]
                legend = fig.legend(lines, labels, ncol=2, loc = "upper center", bbox_to_anchor=(frac_i+0.25,0.9))

                lines_list.append(lines)
                legend_list.append(legend)



        graphs = {}
        for j,legend in enumerate(legend_list):
            legend_lines = legend.get_lines()
            for i in range(len(legend_lines)):
                legend_lines[i].set_picker(True)
                legend_lines[i].set_pickradius(20)
                graphs[legend_lines[i]] = [lines_list[j][i]]

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



        ax[0].set_title("Inputs",fontsize=40)
        ax[1].set_title("Outputs",fontsize=40)

        formatter = mdates.DateFormatter(r"%H")

        for ax_i in ax:
            ax_i.xaxis.set_major_formatter(formatter)
            for label in ax_i.get_xticklabels():
                # label.set_ha("right")
                # label.set_ha("right")
                # label.set_rotation(30)
                label.set_ha("center")
                label.set_rotation(0)
            