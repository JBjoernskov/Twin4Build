import os
import sys
import datetime
from dateutil.tz import tzutc
import matplotlib.pyplot as plt

###Only for testing before distributing package
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
from twin4build.monitor.monitor import Monitor
from twin4build.model.model import Model
def test():
    model = Model(saveSimulationResult=True)
    model.load_model()
    
    monitor = Monitor(model)
    stepSize = 600 #Seconds 
    startPeriod = datetime.datetime(year=2022, month=10, day=10, hour=0, minute=0, second=0) ## Commissioning case
    endPeriod = datetime.datetime(year=2022, month=11, day=15, hour=0, minute=0, second=0) ## Commissioning case 11 15
    # startPeriod = datetime.datetime(year=2021, month=12, day=20, hour=0, minute=0, second=0) #piecewise 20.5-23
    # endPeriod = datetime.datetime(year=2022, month=1, day=20, hour=0, minute=0, second=0) #piecewise 20.5-23
    # startPeriod = datetime.datetime(year=2022, month=10, day=28, hour=0, minute=0, second=0) #Constant 19
    # endPeriod = datetime.datetime(year=2022, month=12, day=23, hour=0, minute=0, second=0) #Constant 19
    monitor.monitor(startPeriod=startPeriod,
                    endPeriod=endPeriod,
                    stepSize=stepSize)
    
    line_date = datetime.datetime(year=2022, month=10, day=27, hour=8, minute=23, second=0) ## At this time, the supply temperature setpoint is changed to constant 19 Deg C
    fig,axes = monitor.plot_dict["Indoor temperature sensor"]
    axes[0].set_ylim([18,28])
    
    for ax in axes:
        ax.axvline(line_date, color=monitor.colors[3])
    monitor.save_plots()
    plt.show()

if __name__ == '__main__':
    test()



