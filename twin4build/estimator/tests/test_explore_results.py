import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import json
import matplotlib.patches as mpatches
import unittest
from twin4build.utils.rsetattr import rsetattr  
from twin4build.estimator.estimator import Estimator
from twin4build.model.model import Model
from twin4build.logger.Logging import Logging
from twin4build.model.tests.test_LBNL_model import fcn
from twin4build.utils.plot.plot import load_params
from twin4build.monitor.monitor import Monitor
logger = Logging.get_logger("ai_logfile")


def change_pos(ax):
    pos1 = ax.get_position() # get the original position 
    pos2 = [pos1.x0, pos1.y0 + 0.05,  pos1.width, pos1.height] 
    ax.set_position(pos2) # set a new position


@unittest.skipIf(True, 'Currently not used')
def test_explore_results():
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
    Model.fcn = fcn
    model = Model(id="model", saveSimulationResult=True)
    model.load_model(infer_connections=False)
    estimator = Estimator(model)

    coil = model.component_dict["coil"]
    valve = model.component_dict["valve"]
    fan = model.component_dict["fan"]


    n_days = 10
    startTime = datetime.datetime(year=2022, month=1, day=1, hour=0, minute=0, second=0)
    startTime_test = datetime.datetime(year=2022, month=2, day=1, hour=1, minute=0, second=0) 
    endTime_test = datetime.datetime(year=2022, month=2, day=28, hour=0, minute=0, second=0)
    sol_list = []
    startTime_list = [startTime]*(n_days-1)
    endTime_list = [startTime + datetime.timedelta(days=dt) for dt in range(1, n_days, 2)]

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

