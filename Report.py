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
         'xtick.labelsize': 25,
         'ytick.labelsize': 30,
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
plt.style.use(plt.style.available[7])
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
        super().__init__(**kwargs)

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
        fig.suptitle(self.__class__.__name__, fontsize=60)
        # figManager = plt.get_current_fig_manager() ################
        # figManager.window.showMaximized() #######################
        fig.set_size_inches(40, 13) 

        grid = plt.GridSpec(rows, cols, hspace=0.1, wspace=0.1) #0.2
        # ax_loss = fig.add_subplot(grid[0, 0:2])

        x_offset = 0.1
        y_offset = 0.15
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
                    x_offset_add = 0
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

        first_axis_priority_list = ["Temperature", "Power", "AirFlowRate", "Signal", "Radiation", "waterFlowRate", "Co2Concentration", "People"]
        color_list = ["black",
                    *global_colors]
        normalize_list = [1, 1000, 1, 1, 1, 1, 1, 1]
        unit_list = ["[$^\circ$C]", "[kW]", "[kg/s]", "", "[W/m$^2$]", "[kg/s]", "[ppm]", ""]
        y_lim_min_list = [-5, -0.5, -0.5, 0.05, -50, -0.01, -10, 0]
        y_lim_max_list = [35, 70, 20, 1.05, 3500, 0.11, 1000, 30]
        data_list = [self.savedInput, self.savedOutput]
        
        for i,data in enumerate(data_list):
            print(data.keys())
            ax_list = [None]*len(first_axis_priority_list)
            linecycler_list = [cycle(["-","--","-.",":"]) for i in range(len(first_axis_priority_list))]
            if len(list(data.keys())) != 0:
                print(data.keys())
                stripped_input_list = [ii for ii in first_axis_priority_list for jj in list(data.keys()) if jj.find(ii)!=-1]
                first_axis_priority_index_list = [first_axis_priority_list.index(ii) for ii in stripped_input_list]
                min_index = min(first_axis_priority_index_list)
                first_axis_index_list = [i for i, x in enumerate(first_axis_priority_index_list) if x == min_index]
                offset_change = 90
                offset = 0
                for j,key in enumerate(data):
                    frac_i = i/(cols)
                    
                    value = np.array(data[key])/normalize_list[first_axis_priority_index_list[j]]
                    if j in first_axis_index_list:
                        color = "black"
                        ax[i].plot(time_list, value, label=key, color=color, linestyle=next(linecycler_list[first_axis_priority_index_list[j]]))
                        # ax[i].set_ylim([y_lim_min_list[first_axis_priority_index_list[j]], y_lim_max_list[first_axis_priority_index_list[j]]])
                        offset_y_label = frac_i + x_offset/3
                        

                    else:
                        color = color_list[first_axis_priority_index_list[j]]
                        if ax_list[first_axis_priority_index_list[j]] is None:
                            ax_list[first_axis_priority_index_list[j]] = ax[i].twinx()
                            
                            ax_list[first_axis_priority_index_list[j]].spines['right'].set_position(('outward', offset))
                            ax_list[first_axis_priority_index_list[j]].spines["right"].set_visible(True)
                            ax_list[first_axis_priority_index_list[j]].spines["right"].set_color(color)
                            ax_list[first_axis_priority_index_list[j]].tick_params(axis='y', colors=color)
                            # ax_list[first_axis_priority_index_list[j]].set_ylim([y_lim_min_list[first_axis_priority_index_list[j]], y_lim_max_list[first_axis_priority_index_list[j]]])
                            
                            offset += offset_change
                        offset_y_label = frac_i + ax_width + x_offset + offset/1200
                        ax_list[first_axis_priority_index_list[j]].plot(time_list, value, label=key, color=color, linestyle=next(linecycler_list[first_axis_priority_index_list[j]]))
                    label_string = first_axis_priority_list[first_axis_priority_index_list[j]] + " " + unit_list[first_axis_priority_index_list[j]]
                    fig.text(offset_y_label, 0.5, label_string, va='center', ha='center', rotation='vertical', fontsize=40, color = color)

                lines_labels_list = []
                lines_labels = ax[i].get_legend_handles_labels()
                lines_labels_list.append(lines_labels)
                for ax_i in ax_list:
                    if ax_i is not None:
                        lines_labels = ax_i.get_legend_handles_labels()
                        lines_labels_list.append(lines_labels)
                lines, labels = [sum(lol, []) for lol in zip(*lines_labels_list)]
                legend = fig.legend(lines, labels, ncol=2, loc = "upper center", bbox_to_anchor=(frac_i+0.25,0.9))



        ax[0].set_title("Inputs",fontsize=40)
        ax[1].set_title("Outputs",fontsize=40)

        formatter = mdates.DateFormatter(r"%D %H")

        for ax_i in ax:
            ax_i.xaxis.set_major_formatter(formatter)
            for label in ax_i.get_xticklabels():
                # label.set_ha("right")
                label.set_ha("right")
                label.set_rotation(30)
            