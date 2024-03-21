from twin4build.base import ClassificationAnnController
import sys
import os
import numpy as np
import torch.nn as nn
import torch

uppath = lambda _path,n: os.sep.join(_path.split(os.sep)[:-n])
file_path = uppath(os.path.abspath(__file__), 9)
sys.path.append(file_path)

from twin4build.logger.Logging import Logging

logger = Logging.get_logger("ai_logfile")

logger.info("Controller Model ann classificator File")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')


class room_controller_net(nn.Module):
    def __init__(self, input_size, output_size):
        super(room_controller_net, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 100)
        self.fc3 = nn.Linear(100, output_size)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def load_model(filename, input_size, output_size, device):
    model_state_dict = torch.load(filename, map_location=torch.device('cpu'))
    model = room_controller_net(input_size, output_size).to(device)
    model.load_state_dict(model_state_dict)
    return model

class ClassificationAnnControllerSystem(ClassificationAnnController):
    def __init__(self, 
                room_identifier = None,
                input_size = None,
                **kwargs):
        super().__init__(**kwargs)
        logger.info("[Controller Model Classification Ann] : Entered in Initialise Funtion")
        self.input = {"actualValue": None}
        self.output = {"inputSignal": None}

        

        self.room_identifier = room_identifier
        self.models_path = r"C:\Users\asces\OneDriveUni\Projects\VentilationModel\ann_classification_models"
        self.model_filename = os.path.join(self.models_path, f"room_controller_classification_net_room{self.room_identifier}.pth")
        self.input_size =  input_size
        self.output_size =  20
        self.device =  device
        self.model = load_model(self.model_filename, self.input_size, self.output_size, self.device)

        self._config = {"parameters": ["input_size", "output_size"]}


    @property
    def config(self):
        return self._config

    def cache(self,
            startTime=None,
            endTime=None,
            stepSize=None):
        pass    

    def initialize(self,
                    startTime=None,
                    endTime=None,
                    stepSize=None):
        pass

    def do_step(self, secondTime=None, dateTime=None, stepSize=None):
        
        logger.info("[Controller Model Classification Ann] : Entered in Do Step Funtion")
        #The input of the model is a data vector of 5 elements: Month, day, hour, minute, CO2. Extract the time-related elements from the simulation timestamp
        month = dateTime.month / 12
        day = dateTime.isoweekday() / 7
        hour = dateTime.hour / 24
        minute = dateTime.minute / 60
        actualValue = (self.input["actualValue"] - 300) / (1300 - 300)

        inputs = torch.tensor([month, day, hour, minute, actualValue], dtype=torch.float32).to(self.device)
        self.model.eval()  # Set model to evaluation mode
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs, 0)
        #Make the output signal to be in the range of 0-1
        predicted = predicted / 20

        self.output["inputSignal"] = predicted.item()
        


