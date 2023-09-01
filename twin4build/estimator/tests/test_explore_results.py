import os
import sys
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import json
import matplotlib.patches as mpatches
# Only for tersting before distributing package.
# If the package is installed, this is not needed.
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.utils.rsetattr import rsetattr  
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.utils.schedule import Schedule
from twin4build.utils.node import Node
from twin4build.logger.Logging import Logging
from twin4build.model.tests.test_LBNL_model import extend_model
from twin4build.utils.plot.plot import load_params
from twin4build.monitor.monitor import Monitor
logger = Logging.get_logger("ai_logfile")


def change_pos(ax):
    pos1 = ax.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0 + 0.05,  pos1.width, pos1.height] 
    ax.set_position(pos2) # set a new position

def test():
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
    load_params()
    stepSize = 60
    Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False)
    estimator = Estimator(model)

    coil = model.component_dict["coil"]
    valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]


    n_days = 10
    startPeriod = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
    startPeriod_test = datetime.datetime(year=2022, month=2, day=1, hour=1, minute=0, second=0) 
    endPeriod_test = datetime.datetime(year=2022, month=2, day=28, hour=0, minute=0, second=0)
    sol_list = []
    startPeriod_list = [startPeriod]*(n_days-1)
    endPeriod_list = [startPeriod + datetime.timedelta(days=dt) for dt in range(1, n_days, 2)]

    days_list = [dt for dt in range(1, n_days, 2)]
    
    
    color_list = [blue, red]
    filename_list = ["DryCoilDiscretized_test_fmu_valve.json"]
    legend_name_list = ["DryCoilDiscretized"]
    width_list = [-0.4, 0.4]
    
    id_list = ["fan power meter", "coil outlet air temperature sensor", "coil outlet water temperature sensor"]
    
    for i,com in enumerate(id_list):
        fig,ax = plt.subplots()
        change_pos(ax)
        patch_list = []
        ax.set_title(com, fontdict={'fontsize': 20})
        for color, filename, legend_name, width in zip(color_list, filename_list, legend_name_list, width_list):
            patch_list.append(mpatches.Patch(color=color, label=legend_name))
            with open(filename, 'r') as f:
                sol = json.load(f)
            for i,result in sol.items():
                if i!="x0":
                    ax.bar(i,result["RMSE"][com], color=color, align="edge", width=width)
        y_lim = ax.get_ylim()
        y_lim = [y_lim[0], y_lim[1]]
        y_lim[1] = y_lim[1]*1.2
        ax.set_ylim(y_lim)
        ax.set_xticklabels(days_list, rotation=0)
        ax.set_xlabel("Number of training days")
        if i==0:
            labelpad = 0.03
        else:
            labelpad = 0
        ax.set_ylabel("RMSE $^\circ$C", labelpad=labelpad)
        ax.legend(handles=patch_list)
        fig.savefig(f"{com}.png", dpi=300)
    


    fig,ax = plt.subplots()
    patch_list = []
    ax.set_title("Objective evaluations", fontdict={'fontsize': 20})
    change_pos(ax)
    for color, filename, legend_name, width in zip(color_list, filename_list, legend_name_list, width_list):
        patch_list.append(mpatches.Patch(color=color, label=legend_name))
        with open(filename, 'r') as f:
            sol = json.load(f)
        
        for i,result in sol.items():
            if i!="x0":
                ax.bar(i,result["n_obj_eval"], color=color, align="edge", width=width)
    y_lim = ax.get_ylim()
    y_lim = [y_lim[0], y_lim[1]]
    y_lim[1] = y_lim[1]*1.2
    ax.set_ylim(y_lim)
    ax.set_xticklabels(days_list, rotation=0)
    ax.set_xlabel("Number of training days")
    ax.legend(handles=patch_list)
    fig.savefig(f"Objective evaluations.png", dpi=300)
    plt.show()


def test_m():

    Model.extend_model = extend_model
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False)
    estimator = Estimator(model)

    coil = model.component_dict["coil"]
    valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]
    controller = model.component_dict["controller"]

    filename = 'DryCoilDiscretized_test_fmu_valve.json'
    # filename = 'DryCoilEffectivenessNTU.json'

    # with open(filename, 'r') as f:
    #     sol_dict = json.load(f)

    n = 0
    if filename=='DryCoilEffectivenessNTU.json':
        # x = {coil: sol_dict[str(n)]["coil"],
        #     valve: sol_dict[str(n)]["valve"],
        #     fan: sol_dict[str(n)]["fan"]}
        x_flat = [val for lst in x.values() for val in lst]
        targetParameters = {coil: ["r_nominal", "nominalSensibleCapacity.hasValue", "m1_flow_nominal", "m2_flow_nominal", "T_a1_nominal", "T_a2_nominal"],
                                        valve: ["workingPressure.hasValue", "flowCoefficient.hasValue", "waterFlowRateMax"],
                                        fan: ["c1", "c2", "c3", "c4"]}
    
    if filename=="DryCoilDiscretized_test_fmu_valve.json":
        # x = {coil: sol_dict[str(n)]["coil"],
        #     valve: sol_dict[str(n)]["valve"],
        #     fan: sol_dict[str(n)]["fan"],
        #     controller: sol_dict[str(n)]["controller"]}
        # x_flat = [val for lst in x.values() for val in lst]
        targetParameters = {coil: ["m1_flow_nominal", "m2_flow_nominal", "tau1", "tau2", "tau_m", "nominalUa.hasValue"],
                                    valve: ["workingPressure.hasValue", "flowCoefficient.hasValue", "waterFlowRateMax"],
                                    fan: ["c1", "c2", "c3", "c4", "eps_motor", "f_motorToAir"],
                                    controller: ["kp", "Ti", "Td"]}

#     x_flat = [ 1.40353308e+00,  1.06362295e+00,  2.70573159e+02,  1.82713852e+00,
#   1.12615421e+01,  2.51661829e+03,  6.76750235e-01,  4.12863219e+00,
#   9.11228031e-02, -3.73065645e-02,  8.54549554e-01, -8.19300187e-02,
#   9.99919185e-01,  9.99999364e-01,]
    
#     x_flat = [ 1.40584326e+00,  5.04118780e-01,  2.71262736e+02,  3.47480993e+00,
#   1.44971271e+00,  2.51854718e+03,  5.00396431e-01,  3.60620605e+00,
#   1.05132916e-01, -1.83889218e-01,  1.31465538e+00, -5.24579640e-01,
#   9.99998881e-01,  9.99991603e-01,] # No flow dependent
    
#     x_flat = [ 5.08853472e-01,  8.29269529e+00,  2.31854242e+02,  1.00002240e+00,
#   1.28782192e+02,  3.33797557e+03,  5.05317854e-01,  3.53916841e+00,
#   1.05749371e-01, -1.87608034e-01,  1.32248188e+00, -5.30335304e-01,
#   3.58337208e-01,  9.99999660e-01] #flow dependent
    
#     x_flat = [ 6.54338088e-01,  9.59287917e+00,  1.96084884e+02,  1.02024368e+00,
#   1.09087869e+02,  3.84930042e+03,  5.04976787e-01,  3.54456981e+00,
#   1.06572615e-01, -1.96536741e-01,  1.35053465e+00, -5.57396259e-01,
#   5.04569992e-03,  9.99996673e-01,]


    #### CONTROL ####
#     x_flat = [ 7.44687491e+00,  6.96325606e-01,  9.49438291e+01,  1.00000551e+00,
#   1.61492213e+02,  5.40605017e+03,  5.00000845e-01,  4.32957157e+00,
#   9.97712023e-02, -2.27992645e-01,  1.57851466e+00, -8.78021930e-01,
#   9.85642376e-01,  9.71561739e-01,  3.13069943e+00,  1.94367614e-01,
#   1.70224070e+00,]
    
    # x_flat = [ 7.44671226e+00,  6.96386947e-01,  9.65113699e+01,  2.00928158e+00,
    #     1.61502385e+02,  5.40944338e+03,  5.00000843e-01,  4.32964025e+00,
    #     9.98000239e-02, -2.28142258e-01,  1.57866772e+00, -8.77944694e-01,
    #     9.85645849e-01,  9.71561529e-01,  3.13069943e+00,  1.94098024e-01,
    #     1.70237301e+00]
    

    #Newest
    # x_flat = [ 2.35085742e+00,  7.91557152e+00,  2.69684584e+02,  4.36178308e+01,
    #             5.24730720e+01,  8.31008499e+03,  6.02943060e+03,  6.07260379e+03,
    #             3.32350667e+00,  4.24228575e-02,  3.22053433e-01, -3.90442400e-02,
    #             6.06855442e-01,  6.92221024e-01,  5.45169395e-01,  3.72090420e+01,
    #             6.75978875e-01,  9.75841443e-01,]
    




    # old
    x_flat = [   2.277,    0.984,  273.414,   16.246,  245.437, 2676.313,  214.288, 7701.148,
    4.873,    0.058,    0.109,    0.038,    0.729,    0.783,    0.327,    0.051,
   44.103,   79.179,]


    x_flat = [   2.518,    0.591,  425.478,   17.327,  218.535, 3492.542,  107.84,  2881.862,
    4.028,    0.058,    0.145,    0.3,      0.588,    0.93,     0.486,    0.405,       
   57.246,   66.391,]



    flat_component_list = [obj for obj, attr_list in targetParameters.items() for i in range(len(attr_list))]
    flat_attr_list = [attr for attr_list in targetParameters.values() for attr in attr_list]

    for i, (obj, attr) in enumerate(zip(flat_component_list, flat_attr_list)):
        print(attr, x_flat[i])
        rsetattr(obj, attr, x_flat[i])


    stepSize = 60
    # startPeriod_test = datetime.datetime(year=2022, month=2, day=23, hour=0, minute=0, second=0)
    # endPeriod_test = datetime.datetime(year=2022, month=2, day=24, hour=0, minute=0, second=0)

    startPeriod_test = datetime.datetime(year=2022, month=2, day=1, hour=0, minute=0, second=0)
    endPeriod_test = datetime.datetime(year=2022, month=2, day=2, hour=0, minute=0, second=0)
    monitor = Monitor(model)
    monitor.monitor(startPeriod=startPeriod_test,
                        endPeriod=endPeriod_test,
                        stepSize=stepSize,
                        do_plot=True)
    id_list = ["coil outlet air temperature sensor", "fan power meter", "coil outlet water temperature sensor", "valve position sensor"]
    for id_ in id_list:
        fig,axes = monitor.plot_dict[id_]
        
        
        for ax in axes:
            # change_pos(ax)
            ax.set_xlabel("Hour of day")
            h, l = ax.get_legend_handles_labels()
            n = len(l)
            
            
            ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.15), prop={'size': 8}, ncol=n)
            # ax.yaxis.label.set_size(15)
            # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        l = 0.03
        box = axes[0].get_position()
        axes[0].set_position([box.x0, box.y0+l, box.width, box.height-l])
        box = axes[1].get_position()
        axes[1].set_position([box.x0, box.y0+2*l, box.width, box.height-l])
    monitor.save_plots()


    fig,axes = monitor.plot_dict["coil outlet air temperature sensor"]
    axes[0].plot(monitor.simulator.dateTimeSteps, model.component_dict["Supply air temperature setpoint"].savedOutput["scheduleValue"])
    plt.show()

if __name__ == '__main__':
    # import cProfile
    # import pstats
    # import io
    # from pstats import SortKey
    # cProfile.run("test()", 'teststats')
    # p = pstats.Stats('teststats')
    # p.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats()
    test_m()


