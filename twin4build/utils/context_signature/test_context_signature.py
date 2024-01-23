import os
import sys
import datetime
from dateutil import tz
if __name__ == '__main__':
    uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
    file_path = uppath(os.path.abspath(__file__), 4)
    sys.path.append(file_path)
import twin4build as tb

def test():
    
    filename = os.path.join(os.path.abspath(uppath(os.path.abspath(__file__), 5)), "Twin4build-Case-Studies", "LBNL", "configuration_template_LBNL.xlsm")
    stepSize = 60
    startTime = datetime.datetime(year=2022, month=2, day=1, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen")) 
    endTime = datetime.datetime(year=2022, month=2, day=2, hour=0, minute=0, second=0, tzinfo=tz.gettz("Europe/Copenhagen"))

    model = tb.Model(id="model", saveSimulationResult=True)
    model.load_model_new(infer_connections=True, semantic_model_filename=filename)

if __name__=="__main__":
    test()