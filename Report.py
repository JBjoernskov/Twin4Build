import matplotlib.pyplot as plt
import math
import matplotlib.pylab as pylab
import matplotlib.dates as mdates

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
         "legend.fontsize": 30,
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
        fig.set_size_inches(40, 11) 

        grid = plt.GridSpec(rows, cols, hspace=0.1, wspace=0.1) #0.2
        # ax_loss = fig.add_subplot(grid[0, 0:2])

        x_offset = 0.05
        y_offset = 0.2
        ax_width = 0.4
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
                    ax_twin_AirFlowRate.append(added_ax.twinx())
                    ax_twin_Power.append(added_ax.twinx())
                    ax_twin_Signal.append(added_ax.twinx())
                    ax_twin_Radiation.append(added_ax.twinx())




        for key in self.savedInput:
            print("---savedInput---")
            print(key)

            value = self.savedInput[key]
            if key.find("Temperature")!=-1:
                ax[0].plot(time_list, value, label=key, linestyle="dashed")
                ax[0].set_ylim([-5, 35])
            elif key.find("AirFlowRate")!=-1:
                ax_twin_AirFlowRate[0].plot(time_list, value, label=key, linestyle="dashed")
                ax_twin_AirFlowRate[0].set_ylim([0, 30])
            elif key.find("Power")!=-1:
                ax_twin_Power[0].plot(time_list, value, label=key, linestyle="dashed")
                ax_twin_Power[0].set_ylim([-50, 50000])
            elif key.find("Signal")!=-1:
                ax_twin_Signal[0].plot(time_list, value, label=key, linestyle="dashed")
                ax_twin_Signal[0].set_ylim([-0.05, 1.05])
            elif key.find("Radiation")!=-1:
                ax_twin_Radiation[0].plot(time_list, value, label=key, linestyle="dashed")
                ax_twin_Radiation[0].set_ylim([-0.05, 1.05])
            

        for key in self.savedOutput:
            print("---savedOutput---")
            print(key)
            value = self.savedOutput[key]
            if key.find("Temperature")!=-1:
                print("Temperature")
                ax[1].plot(time_list, value, label=key, linestyle="dashed")
                ax[1].set_ylim([-5, 35])
            elif key.find("AirFlowRate")!=-1:
                print("AirFlowRate")
                ax_twin_AirFlowRate[1].plot(time_list, value, label=key, linestyle="dashed")
                ax_twin_AirFlowRate[1].set_ylim([0, 30])
            elif key.find("Power")!=-1:
                print("Power")
                ax_twin_Power[1].plot(time_list, value, label=key, linestyle="dashed")
                ax_twin_Power[1].set_ylim([-50, 50000])
            elif key.find("Signal")!=-1:
                print("Signal")
                ax_twin_Signal[1].plot(time_list, value, label=key, linestyle="dashed")
                ax_twin_Signal[1].set_ylim([-0.05, 1.05])
            elif key.find("Radiation")!=-1:
                print("Radiation")
                ax_twin_Radiation[1].plot(time_list, value, label=key, linestyle="dashed")
                ax_twin_Radiation[1].set_ylim([-50, 3500])


        ax[0].set_title("Inputs",fontsize=40)
        ax[1].set_title("Outputs",fontsize=40)
        ax[0].legend()
        ax[1].legend()

        formatter = mdates.DateFormatter(r"%D %H")

        for ax_i in ax:
            ax_i.xaxis.set_major_formatter(formatter)
            for label in ax_i.get_xticklabels():
                # label.set_ha("right")
                label.set_ha("right")
                label.set_rotation(45)
            