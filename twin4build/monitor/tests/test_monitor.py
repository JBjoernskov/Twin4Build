import os
import sys
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import matplotlib.ticker as ticker
###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.monitor.monitor import Monitor
from twin4build.model.model import Model
from twin4build.utils.plot.plot import bar_plot_line_format
def test():
    model = Model(id="model", saveSimulationResult=True)
    model.load_model()
    model.prepare_for_simulation()
    
    monitor = Monitor(model)
    stepSize = 600 #Seconds 
    startPeriod = datetime.datetime(year=2022, month=10, day=23, hour=0, minute=0, second=0)
    endPeriod = datetime.datetime(year=2022, month=11, day=6, hour=0, minute=0, second=0)
    monitor.monitor(startPeriod=startPeriod,
                    endPeriod=endPeriod,
                    stepSize=stepSize)
    

    # The rest is just formatting the resulting plot
    line_date = datetime.datetime(year=2022, month=10, day=27, hour=8, minute=23, second=0) ## At this time, the supply temperature setpoint is changed to constant 19 Deg 
    id_list = ["Space temperature sensor", "Heat recovery temperature sensor", "Heating coil temperature sensor"]
    for id_ in id_list:
        fig,axes = monitor.plot_dict[id_]
        
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            n = len(l)
            box = ax.get_position()
            ax.set_position([0.12, box.y0, box.width, box.height])
            ax.legend(loc="upper center", bbox_to_anchor=(0.5,1.15), prop={'size': 8}, ncol=n)
            ax.yaxis.label.set_size(15)
            ax.axvline(line_date, color=monitor.colors[3])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

            # df = pd.DataFrame()
            # df.insert(0, "time", monitor.simulator.dateTimeSteps)
            # df = df.set_index("time")
            # ax.set_xticklabels(map(bar_plot_line_format, df.index, [evaluation_metric]*len(df.index)))

    fig,axes = monitor.plot_dict["monitor"]
    for ax in axes:
        ax.axvline(line_date, color=monitor.colors[3])
    monitor.save_plots()
    plt.show()

if __name__ == '__main__':
    test()


