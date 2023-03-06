import os
import sys
import datetime
from dateutil.tz import tzutc
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
    startPeriod = datetime.datetime(year=2022, month=10, day=1, hour=0, minute=0, second=0)
    endPeriod = datetime.datetime(year=2022, month=12, day=31, hour=0, minute=0, second=0)
    monitor.monitor(startPeriod=startPeriod,
                    endPeriod=endPeriod,
                    stepSize=stepSize)
    


if __name__ == '__main__':
    test()